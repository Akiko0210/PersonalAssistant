"""Application orchestrator (§3).

Owns the three-state machine and wires every subsystem to it. This is the only place
where audio, models, cloud, feedback, and controls meet; each subsystem stays ignorant
of the others. The app implements :class:`AgentController` so the §8 tools can drive
capture and report state.

State side-effects (registered as machine hooks):

  MUTED      enter: release the mic stream (hard mute, FR-M2) + mute earcon (FR-V3)
  LISTENING  enter: start Pipecat local audio + cloud loop, "Listening"
  CAPTURING  enter: suspend cloud loop (A4), open local router, start VAD recorder

Guards (§4):
  CAPTURING -> MUTED  : auto-stop + save the active session first (FR-M4, R-8)
  LISTENING -> CAPTURING and * -> MUTED : cancel in-flight TTS/LLM (FR-M5)

Resilience: on start, unfinalized sessions from a crash are recovered (R-8). The app
starts MUTED by default (privacy, C7).
"""

from __future__ import annotations

import logging
import sys
import threading
from datetime import datetime
from typing import Callable

from .agent.conversation import ConversationPipeline
from .agent.tools import NoteTools
from .audio import devices
from .audio.router import AudioRouter
from .audio.vad import SpeechProbability
from .capture.recorder import Recorder
from .capture.session import NoteSession, find_unfinalized_sessions
from .capture.transcriber import FasterWhisperTranscriber, Transcribe, TranscriptionWorker
from .config import Config
from .controls.hotkeys import HotkeyActions, HotkeyManager
from .feedback.earcons import SAMPLE_RATE
from .feedback.voice import VoiceFeedback, deepgram_speak_factory
from .paths import Paths
from .state import State, StateMachine, Transition
from .storage.rag import NotesIndex
from .storage.store import Store
from .summarize.summarizer import Summarizer

log = logging.getLogger(__name__)


class App:
    """The foreground app body that owns state, controls, audio, and storage."""

    def __init__(
        self,
        cfg: Config,
        paths: Paths,
        *,
        # Heavy/optional components are injectable so the app is testable without
        # models, microphones, or network. Defaults build the real things lazily.
        prob_fn: SpeechProbability | None = None,
        transcribe: Transcribe | None = None,
        summarizer_client=None,
        conversation: ConversationPipeline | None = None,
        play=None,
        speak=None,
    ) -> None:
        self._cfg = cfg
        self._paths = paths

        # Storage + retrieval + summarization.
        self._store = Store(paths)
        self._index = NotesIndex(cfg, paths)
        self._summarizer = Summarizer(cfg, client=summarizer_client)

        # Feedback (earcons + spoken confirmations). Playback/TTS injectable.
        self._play = play or (lambda audio, sr: devices.play(audio, sr, device=cfg.audio.output_device))
        self._speak = speak or deepgram_speak_factory(cfg, self._play)
        self._feedback = VoiceFeedback(cfg, play=self._play, speak=self._speak)

        # Tools + conversation pipeline (the single LLM agent).
        self._tools = NoteTools(self, self._store, self._index, self._summarizer)
        self._conversation = conversation or ConversationPipeline(
            cfg, self._tools, on_tool_called=self._on_tool_called
        )

        # Audio + capture wiring.
        self._router = AudioRouter(sample_rate=cfg.audio.sample_rate, device=cfg.audio.input_device)
        self._prob_fn = prob_fn
        self._transcribe = transcribe
        self._recorder: Recorder | None = None
        self._active_session: NoteSession | None = None

        # Listening / mute model state (Open Decision O-2).
        self._listening_mode = cfg.mute.default_model

        # State machine.
        self._sm = StateMachine(initial=State.MUTED)
        self._wire_state_machine()

        # Controls.
        self._hotkeys = HotkeyManager(cfg, self._build_hotkey_actions())

        self._lock = threading.RLock()

    # ── public lifecycle ─────────────────────────────────────────────────────
    def start(self) -> None:
        """Recover crashed sessions, register controls, settle into the start state."""
        self._recover_unfinalized()
        self._hotkeys.start()
        # Default state at launch is MUTED (§4 / C7). Fire the enter-hook explicitly so
        # any startup earcon plays; the machine already initialised to MUTED.
        if not self._cfg.mute.start_muted:
            self._sm.unmute()
        log.info("voice notes agent started in state %s", self._sm.state.value)

    def stop(self) -> None:
        """Graceful shutdown: save any active session, tear down controls + cloud loop."""
        if self._sm.state is State.CAPTURING:
            self._sm.mute()  # guard auto-saves
        self._hotkeys.stop()
        self._conversation.stop()
        self._router.close()

    def run_forever(self) -> None:  # pragma: no cover - blocking loop
        self.start()
        try:
            self._console_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    # ── foreground terminal controls ─────────────────────────────────────────
    def _console_loop(self) -> None:  # pragma: no cover - interactive
        """Drive the agent from single keypresses in this terminal window.

        This is the reliable, always-works control surface. Global hotkeys and headset
        media buttons still work when available, but many wired headsets (e.g. Apple
        EarPods) don't emit media keys on Windows, so the terminal keys are the floor.
        """
        try:
            import msvcrt  # Windows: read one keypress at a time, no Enter needed.
        except ImportError:
            msvcrt = None

        self._print_controls()
        if msvcrt is not None:
            while True:
                ch = msvcrt.getwch()
                if ch in ("q", "\x03", "\x1b"):  # q / Ctrl-C / Esc
                    break
                self._handle_console_key(ch)
        else:  # POSIX / non-Windows fallback: line-buffered input.
            for line in sys.stdin:
                ch = line.strip()[:1].lower()
                if ch == "q":
                    break
                if ch:
                    self._handle_console_key(ch)

    def _handle_console_key(self, ch: str) -> None:  # pragma: no cover - interactive
        key = ch.lower()
        if key == "m":
            self.toggle_mute_short()
        elif key == "n":
            self.toggle_notes()
        elif key == "s":
            self._feedback.status(
                f"{self._sm.state.value}"
                + (f", session {self._active_session.id}" if self._active_session else "")
            )
        else:
            return
        print(f"[state] {self._sm.state.value}", flush=True)

    def _print_controls(self) -> None:  # pragma: no cover - interactive
        print(
            "\n=== Voice Notes Agent - terminal controls ===\n"
            "  m : toggle Listening / Muted\n"
            "  n : start / stop note capture\n"
            "  s : speak current status\n"
            "  q : quit\n"
            f"Starting state: {self._sm.state.value}. Keep this window focused.\n"
            "(Global hotkeys Ctrl+Alt+M / Ctrl+Alt+N also work when available.)\n",
            flush=True,
        )

    # ── AgentController protocol (used by tools) ─────────────────────────────
    def start_capture(self) -> str:
        tr = self._sm.start_capture()
        sid = self._active_session.id if self._active_session else ""
        log.info("start_capture -> %s (%s)", sid, tr.trigger)
        return sid

    def stop_capture(self) -> tuple[str, str]:
        session = self._active_session
        self._sm.stop_capture()
        if session is None:
            return "", "no_active_session"
        return session.id, "finalized"

    def active_session_id(self) -> str | None:
        return self._active_session.id if self._active_session else None

    def current_state(self) -> str:
        return self._sm.state.value

    def listening_mode(self) -> str:
        return self._listening_mode

    # ── user-intent handlers (driven by hotkeys / wake words) ────────────────
    def toggle_mute_short(self) -> None:
        """Short press: toggle between MUTED(wake-word sleep) and LISTENING."""
        with self._lock:
            if self._sm.state is State.MUTED:
                self._listening_mode = self._cfg.mute.short_press
                self._sm.unmute()
            else:
                self._listening_mode = self._cfg.mute.short_press
                self._sm.mute()

    def toggle_mute_long(self) -> None:
        """Long press: true mute (only a manual control wakes it, FR-M3)."""
        with self._lock:
            if self._sm.state is State.MUTED:
                self._listening_mode = self._cfg.mute.long_press
                self._sm.unmute()
            else:
                self._listening_mode = self._cfg.mute.long_press
                self._sm.mute()

    def toggle_notes(self) -> None:
        """notes_toggle hotkey: start if LISTENING, stop (+summarize) if CAPTURING."""
        with self._lock:
            if self._sm.state is State.LISTENING:
                self.start_capture()
            elif self._sm.state is State.CAPTURING:
                self.handle_stop_notes()

    def handle_stop_notes(self) -> None:
        """Button / wake-word stop path: finalize then summarize + read back ourselves.

        In CAPTURING the cloud agent is dormant (A4), so the app — not the LLM — must run
        the post-stop summary and read-back (FR-C8, FR-S3).
        """
        session = self._active_session
        sid, _status = self.stop_capture()
        if session is not None:
            threading.Thread(
                target=self._summarize_and_readback,
                args=(session,),
                name="summarize",
                daemon=True,
            ).start()
        else:
            self._start_conversation_pipeline()

    # ── state-machine wiring ─────────────────────────────────────────────────
    def _wire_state_machine(self) -> None:
        sm = self._sm
        sm.on_enter(State.MUTED, self._enter_muted)
        sm.on_enter(State.LISTENING, self._enter_listening)
        sm.on_enter(State.CAPTURING, self._enter_capturing)
        sm.on_exit(State.CAPTURING, self._exit_capturing)

        # Guards (§4).
        sm.add_guard(State.CAPTURING, State.MUTED, self._guard_autosave_capture)
        sm.add_guard(State.LISTENING, State.CAPTURING, self._guard_cancel_inflight)
        sm.add_guard(State.LISTENING, State.MUTED, self._guard_cancel_inflight)

    # -- enter/exit handlers --------------------------------------------------
    def _enter_muted(self, _tr: Transition) -> None:
        self._router.close()  # hard mute: release the mic device (FR-M2)
        self._conversation.stop()  # Pipecat owns LISTENING audio; stop it on hard mute.
        self._feedback.muted()  # earcon only (FR-V3)

    def _enter_listening(self, tr: Transition) -> None:
        # Pipecat's local transport owns mic/speaker I/O in LISTENING. Keep the local
        # AudioRouter closed so it cannot compete for the headset microphone.
        self._router.close()
        if tr.src is State.CAPTURING:
            # Don't start the conversation pipeline yet — _summarize_and_readback
            # needs exclusive audio output first. It starts the pipeline when done.
            return
        self._start_conversation_pipeline()

    def _enter_capturing(self, _tr: Transition) -> None:
        # Suspend the cloud loop; the recorder takes the mic locally (A4, C4).
        self._conversation.suspend()
        session = NoteSession.create(self._paths, sample_rate=self._cfg.audio.sample_rate)
        self._active_session = session
        recorder = self._build_recorder(session)
        self._recorder = recorder
        recorder.start()
        self._router.open(sink=recorder.on_frame)
        self._feedback.recording_notes()

    def _exit_capturing(self, _tr: Transition) -> None:
        recorder = self._recorder
        self._recorder = None
        self._router.close()
        if recorder is not None:
            self._feedback.stopped_summarizing()
            recorder.stop()  # flush + drain transcription + finalize (NFR-3)

    # -- guards ---------------------------------------------------------------
    def _guard_autosave_capture(self, _tr: Transition) -> None:
        """Entering MUTED from CAPTURING must auto-stop and save first (FR-M4)."""
        # The CAPTURING exit handler finalizes the recorder; nothing else needed here.
        # We still kick the summary on a background thread so muting stays instant.
        session = self._active_session
        if session is not None:
            threading.Thread(
                target=self._summarize_and_readback,
                args=(session,),
                name="summarize-on-mute",
                daemon=True,
            ).start()

    def _guard_cancel_inflight(self, _tr: Transition) -> None:
        """Cancel any in-flight TTS/LLM response (§4 entering CAPTURING; FR-M5 muting)."""
        self._conversation.cancel_in_flight()

    # ── capture component construction (lazy heavy deps) ─────────────────────
    def _ensure_prob_fn(self) -> SpeechProbability:
        if self._prob_fn is None:
            from .audio.vad import load_silero_probability

            self._prob_fn = load_silero_probability()
        return self._prob_fn

    def _build_recorder(self, session: NoteSession) -> Recorder:
        prob_fn = self._ensure_prob_fn()
        if self._transcribe is not None:
            transcribe: Transcribe = self._transcribe
            fallback = None
        else:
            fw = FasterWhisperTranscriber(self._cfg.whisper)
            transcribe = fw
            fallback = fw.use_fallback
        worker = TranscriptionWorker(
            transcribe,
            on_text=session.set_segment_text,
            fallback=fallback,
        )
        return Recorder(session, self._cfg.vad, prob_fn, worker)

    # ── post-capture summary + read-back ─────────────────────────────────────
    def _summarize_and_readback(self, session: NoteSession) -> None:
        """Summarize a finished session and read back the spoken summary (FR-S2/S3)."""
        try:
            self._feedback.start_working_cue()  # long-op cue (FR-V4)
            transcript = session.transcript_text()
            speech_sec = session.total_speech_sec
            info = self._store.find(session.id)
            session_dir = info.dir if info else session.dir

            if not self._summarizer.should_summarize(speech_sec) or not transcript.strip():
                # Negligible speech: keep the transcript, skip the full summary (FR-S5).
                self._index.index_session(
                    session_id=session.id,
                    started=session.started.isoformat(),
                    transcript=transcript,
                    summary="",
                )
                self._feedback.status("Saved the note. There was little to summarize.")
                return

            result = self._summarizer.summarize(transcript)
            markdown = result.to_markdown(self._cfg.summary.full_template)
            self._store.save_summary(session_dir, markdown, title=result.title)
            self._index.index_session(
                session_id=session.id,
                started=session.started.isoformat(),
                transcript=transcript,
                summary=markdown,
            )
            self._feedback.summary_ready(result.spoken_summary)
        except Exception as exc:  # FR-V5: announce errors audibly
            log.exception("summarization failed")
            self._feedback.error("I couldn't reach the summarizer.")
            try:
                self._index.index_session(
                    session_id=session.id,
                    started=session.started.isoformat(),
                    transcript=session.transcript_text(),
                    summary="",
                )
            except Exception:
                log.exception("failed to index session after summarization error")
        finally:
            self._feedback.stop_working_cue()
            if self._active_session is session:
                self._active_session = None
            if self._sm.state is State.LISTENING:
                self._start_conversation_pipeline()

    # ── crash recovery (R-8) ─────────────────────────────────────────────────
    def _recover_unfinalized(self) -> None:
        for d in find_unfinalized_sessions(self._paths):
            log.warning("recovering unfinalized session at %s", d)
            # Mark finalized; the audio + any transcript already on disk are preserved.
            info = self._store.info_for(d)
            if info is None:
                continue
            transcript_path = d / "transcript.txt"
            transcript = (
                transcript_path.read_text(encoding="utf-8") if transcript_path.exists() else ""
            )
            try:
                self._index.index_session(
                    session_id=info.session_id,
                    started=info.started,
                    transcript=transcript,
                    summary="",
                )
            except Exception:
                log.exception("failed to index recovered session %s", info.session_id)

    def _start_conversation_pipeline(self) -> None:
        """Start the conversation pipeline and announce LISTENING."""
        self._feedback.listening()
        self._conversation.start()
        self._conversation.resume()

    # ── hooks ────────────────────────────────────────────────────────────────
    def _on_tool_called(self, name: str) -> None:
        # Keep the conversation/capture mutual-exclusion honest when the LLM drives
        # capture: start/stop flow through start_capture/stop_capture already, which
        # transition the machine and suspend/resume the cloud loop.
        log.debug("LLM tool called: %s", name)

    def _build_hotkey_actions(self) -> HotkeyActions:
        return HotkeyActions(
            mute_short=self.toggle_mute_short,
            mute_long=self.toggle_mute_long,
            notes_toggle=self.toggle_notes,
            push_to_talk_down=self.toggle_mute_short,   # PTT: open mic for a turn
            push_to_talk_up=lambda: None,
        )
