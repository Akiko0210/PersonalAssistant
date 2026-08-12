"""Shared fakes for the Agent-side (voice_agent) tests. One definition each of
the audio/STT/LLM fakes that had grown incompatible copies per file — notably
FakeAudio, whose three variants disagreed about collect_utterance's signature
(only one accepted the wake= kwarg the real turn loop passes).

test_mute_speech keeps its own agent builder: it exercises say() itself, so it
must not have say faked out; it shares FakeAudio from here.
"""

import logging
import queue
import threading

import numpy as np

from voice_agent import Agent


def audio_chunk(n=4):
    """A stand-in captured utterance (int16 samples, like the mic yields)."""
    return np.ones(n, dtype=np.int16)


class FakeAudio:
    """The AudioEngine surface the turn logic touches, with scripted input:
    `utterances` feed collect_utterance in order (then None = idle wake),
    `poll_frames` feed poll_speech (then None = queue empty)."""

    def __init__(self, utterances=None, poll_frames=None):
        self._utterances = list(utterances or [])
        self._poll = list(poll_frames or [])
        self.muted = threading.Event()
        self.pushed = []
        self.flushed = 0

    def collect_utterance(self, interrupt=None, endpoint_ms=None, wake=None):
        return self._utterances.pop(0) if self._utterances else None

    def poll_speech(self, timeout=0.05, return_frame=False):
        return self._poll.pop(0) if self._poll else None

    def pushback(self, frames):
        self.pushed.append(list(frames))

    def flush(self):
        self.flushed += 1

    def has_buffered_speech(self):
        return False


class FakeSTT:
    def __init__(self, texts=()):
        self._texts = list(texts)

    def transcribe(self, audio):
        return self._texts.pop(0) if self._texts else ""


class FakeLLM:
    """The Claude surface the turn loop touches, echoing replies."""

    active = "alice"  # persona routing needs to know who is active

    def __init__(self):
        self.calls = []
        self.unanswered = []

    def converse(self, text):
        self.calls.append(text)
        return f"reply::{text}"

    def record_unanswered(self, text):
        self.unanswered.append(text)

    def take_pending_note(self):
        return None

    def take_pending_switch(self):
        return None

    def take_pending_delegations(self):
        return []


def make_agent(*, heard=(), utterances=None, poll_frames=None, await_seq=None,
               interrupted_remaining=None):
    """An Agent shell via __new__ — no mic, models, or SAPI. say() is faked to
    capture into .spoken and finish uninterrupted; tests that exercise say()
    itself (test_mute_speech) build their own shell instead."""
    agent = Agent.__new__(Agent)
    agent.log = logging.getLogger("test")
    agent.interrupt = threading.Event()
    # Only note-taking and quit abandon a turn; mute stops the microphone and
    # leaves the question to be answered (see test_mute_speech).
    agent.silence = threading.Event()
    agent.hush = threading.Event()
    agent.resume_speech = threading.Event()
    agent.cmds = queue.Queue()
    agent.interject = threading.Event()
    agent.interjections = queue.Queue()
    agent.typed = queue.Queue()
    agent.status = "conversation_mode"
    agent.audio = FakeAudio(utterances=utterances, poll_frames=poll_frames)
    agent.stt = FakeSTT(heard)
    agent.llm = FakeLLM()
    agent._speaking_voice = ""
    agent._interrupted_reply = "the full reply" if interrupted_remaining else None
    agent._interrupted_remaining = interrupted_remaining
    agent.spoken = []

    def fake_say(text, **kwargs):
        agent.spoken.append(text)
        return False  # finishes uninterrupted

    agent.say = fake_say
    if await_seq is not None:
        answers = iter(await_seq)
        agent._await_continuation = lambda settle: next(answers)
    return agent
