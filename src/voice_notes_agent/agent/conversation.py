"""Real-time conversation subsystem (Pipecat) — cloud STT -> Claude(tools) -> cloud TTS.

This is the LISTENING-mode pipeline (§5.6, phases 3-4). It is built on Pipecat, which
provides native turn-taking, barge-in, silence handling, and tool calling (§7). The
pipeline:

    mic frames ─▶ VAD ─▶ Deepgram STT ─▶ Anthropic LLM (with §8 tools) ─▶ Deepgram TTS ─▶ out

Responsibilities mapped to requirements:
  * FR-Q1: real-time streaming pipeline
  * FR-Q2: barge-in — user can interrupt mid-speech; TTS + generation stop promptly
  * FR-Q3: min-duration guard so coughs/backchannels don't falsely interrupt
  * FR-Q4 / R-7: interruption during a tool call must not hang or double-fire. Policy:
    let an in-flight tool finish in the background and surface its result on the next
    turn; only cancel if a new, contradicting intent arrives.
  * A4: when the agent calls ``start_note_session`` the app *suspends* this pipeline so
    note audio is handled locally and never streamed to the cloud.

Pipecat's concrete class names vary across releases, so all Pipecat imports are lazy and
isolated in :meth:`_build`. Treat this module as the integration seam: the surrounding
contract (start/stop/suspend/resume, tool dispatch, barge-in guard) is stable; the inner
wiring is adjusted to whatever Pipecat version is installed.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
from typing import Any, Callable

from .tools import TOOL_SPECS, NoteTools

log = logging.getLogger(__name__)

# Set by the app: when a note tool starts/stops capture, the app must suspend/resume
# this pipeline. The conversation exposes a hook the tool layer triggers indirectly
# through the controller, so no direct coupling is needed here.
ToolDispatch = Callable[[str, dict[str, Any]], dict[str, Any]]


class ConversationPipeline:
    """Owns the Pipecat pipeline and its asyncio loop on a dedicated thread."""

    def __init__(self, cfg, tools: NoteTools, *, on_tool_called: Callable[[str], None] | None = None):
        self._cfg = cfg
        self._tools = tools
        self._on_tool_called = on_tool_called
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._task = None  # Pipecat PipelineTask
        self._runner = None
        self._suspended = False
        self._pending_tool_results: dict[str, dict] = {}  # R-7 bookkeeping

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        """Start the pipeline on its own thread + event loop (LISTENING entered)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._thread_main, name="pipecat", daemon=True)
        self._thread.start()
        log.info("conversation pipeline starting")

    def stop(self) -> None:
        """Tear down the pipeline (leaving LISTENING)."""
        loop = self._loop
        if loop is not None:
            asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        log.info("conversation pipeline stopped")

    def suspend(self) -> None:
        """Pause cloud STT/LLM/TTS while capture runs (A4). Mic is freed for local use."""
        self._suspended = True
        self.cancel_in_flight()
        loop = self._loop
        if loop is not None:  # pragma: no cover - requires pipecat
            asyncio.run_coroutine_threadsafe(self._pause_io(), loop)
        log.info("conversation pipeline suspended (capture active)")

    def resume(self) -> None:
        """Resume the cloud loop after capture ends (back to LISTENING)."""
        self._suspended = False
        loop = self._loop
        if loop is not None:  # pragma: no cover - requires pipecat
            asyncio.run_coroutine_threadsafe(self._resume_io(), loop)
        log.info("conversation pipeline resumed")

    def cancel_in_flight(self) -> None:
        """Cancel any in-flight TTS/LLM response (guard for §4 / FR-M5)."""
        loop = self._loop
        if loop is not None:  # pragma: no cover - requires pipecat
            asyncio.run_coroutine_threadsafe(self._interrupt(), loop)

    # -- tool calls (R-7 policy) ----------------------------------------------
    def handle_tool_call(self, name: str, arguments: dict[str, Any], tool_use_id: str) -> dict:
        """Dispatch a tool call, recording it so interruption can't double-fire (R-7).

        We track the tool_use_id; if the same call is delivered again after a barge-in,
        we return the cached result instead of re-running the side effect.
        """
        if tool_use_id in self._pending_tool_results:
            return self._pending_tool_results[tool_use_id]
        if self._on_tool_called is not None:
            self._on_tool_called(name)
        result = self._tools.dispatch(name, arguments)
        self._pending_tool_results[tool_use_id] = result
        # Cap memory; only the most recent handful matter for the dedupe window.
        if len(self._pending_tool_results) > 32:
            self._pending_tool_results.pop(next(iter(self._pending_tool_results)))
        return result

    # -- thread / loop --------------------------------------------------------
    def _thread_main(self) -> None:  # pragma: no cover - requires pipecat + keys
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        except Exception:
            log.exception("conversation pipeline crashed")
        finally:
            self._loop.close()
            self._loop = None

    async def _run(self) -> None:  # pragma: no cover - requires pipecat + keys
        task = self._build()
        from pipecat.pipeline.runner import PipelineRunner

        self._runner = PipelineRunner(**_pipeline_runner_kwargs(PipelineRunner))
        await self._runner.run(task)

    def _build(self):  # pragma: no cover - requires pipecat + keys
        """Construct the Pipecat pipeline. Lazy imports isolate the optional dep."""
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.services.anthropic.llm import AnthropicLLMService
        from pipecat.services.deepgram.stt import DeepgramSTTService
        from pipecat.services.deepgram.tts import DeepgramTTSService
        from pipecat.transports.local.audio import (
            LocalAudioTransport,
            LocalAudioTransportParams,
        )

        cc = self._cfg.conversation
        # Barge-in min-duration guard (FR-Q3): require sustained speech before interrupt.
        vad = SileroVADAnalyzer(
            params=VADParams(
                start_secs=cc.bargein_min_ms / 1000.0,
                stop_secs=cc.turn_gap_ms / 1000.0,
            )
        )
        transport = LocalAudioTransport(
            LocalAudioTransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=vad,
                **_local_audio_transport_kwargs(LocalAudioTransportParams, cc),
            )
        )
        stt = DeepgramSTTService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            model=self._cfg.providers.stt.model,
        )
        tts = DeepgramTTSService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            voice=self._cfg.providers.tts.voice,
        )
        llm = AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=self._cfg.providers.llm.model,
            system_prompt=cc.system_prompt,
        )
        self._register_tools(llm)

        pipeline = Pipeline([
            transport.input(),
            stt,
            llm,
            tts,
            transport.output(),
        ])
        self._task = PipelineTask(
            pipeline,
            params=PipelineParams(allow_interruptions=True),  # barge-in (FR-Q2)
        )
        return self._task

    def _register_tools(self, llm) -> None:  # pragma: no cover - requires pipecat
        """Register the §8 tools with the Pipecat LLM service.

        Each tool handler defers to :meth:`handle_tool_call`, which enforces the R-7
        idempotency policy so a barge-in during a tool call can't double-fire it.
        """
        for spec in TOOL_SPECS:
            name = spec["name"]

            async def _handler(params, _name=name):
                args = getattr(params, "arguments", {}) or {}
                tool_use_id = getattr(params, "tool_call_id", _name)
                result = self.handle_tool_call(_name, args, tool_use_id)
                await params.result_callback(result)

            llm.register_function(name, _handler)

    # -- pipecat control coroutines (best-effort across versions) -------------
    async def _interrupt(self):  # pragma: no cover - requires pipecat
        if self._task is not None:
            try:
                from pipecat.frames.frames import StopInterruptionFrame  # noqa: F401

                await self._task.queue_frame(_interruption_frame())
            except Exception:
                log.debug("interrupt frame not available in this pipecat version")

    async def _pause_io(self):  # pragma: no cover - requires pipecat
        if self._task is not None:
            try:
                await self._task.pause()  # newer pipecat
            except AttributeError:
                await self._interrupt()

    async def _resume_io(self):  # pragma: no cover - requires pipecat
        if self._task is not None:
            try:
                await self._task.resume()
            except AttributeError:
                pass

    async def _shutdown(self):  # pragma: no cover - requires pipecat
        if self._task is not None:
            try:
                await self._task.cancel()
            except Exception:
                pass


def _interruption_frame():  # pragma: no cover - requires pipecat
    from pipecat.frames.frames import BotInterruptionFrame

    return BotInterruptionFrame()


def _pipeline_runner_kwargs(runner_cls) -> dict[str, Any]:
    """Return runner options that are safe for Pipecat's evolving constructors."""
    try:
        params = inspect.signature(runner_cls).parameters
    except (TypeError, ValueError):
        return {}

    kwargs: dict[str, Any] = {}
    if "handle_sigint" in params:
        # This pipeline runs on a background thread. On Windows, signal.signal()
        # only works in the main thread, so app-level shutdown owns Ctrl-C instead.
        kwargs["handle_sigint"] = False
    return kwargs


def _local_audio_transport_kwargs(params_cls, conversation_cfg) -> dict[str, Any]:
    """Return Pipecat local-audio kwargs supported by the installed transport params."""
    configured = {
        "input_device_index": conversation_cfg.input_device_index,
        "output_device_index": conversation_cfg.output_device_index,
    }
    desired = {key: value for key, value in configured.items() if value is not None}

    supported = _supported_constructor_kwargs(params_cls)
    device_keys = {"input_device_index", "output_device_index"}
    if supported is not None and device_keys.isdisjoint(supported):
        return {}

    if len(desired) < len(device_keys):
        try:
            from ..audio.devices import select_pyaudio_device_indexes

            auto_selected = select_pyaudio_device_indexes(
                input_device_index=conversation_cfg.input_device_index,
                output_device_index=conversation_cfg.output_device_index,
            )
            desired = {**auto_selected, **desired}
        except Exception:
            log.exception("could not auto-select PyAudio devices; using Pipecat defaults")

    if not desired:
        return {}

    if supported is None:
        return desired
    return {key: value for key, value in desired.items() if key in supported}


def _supported_constructor_kwargs(cls) -> set[str] | None:
    """Best-effort field discovery for dataclasses, pydantic models, and plain classes."""
    fields = getattr(cls, "model_fields", None) or getattr(cls, "__fields__", None)
    if isinstance(fields, dict) and fields:
        return set(fields)

    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return None
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return None
    return {name for name in params if name != "self"}
