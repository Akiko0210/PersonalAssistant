"""Hands-free, voice-first personal notes assistant.

A single persistent process hosting two mutually-exclusive subsystems:

* a local, VAD-gated **capture** subsystem (note-taking + CPU Whisper), and
* a cloud-backed **conversation** subsystem (streaming STT -> Claude w/ tools -> TTS),

coordinated by a three-state machine (MUTED / LISTENING / CAPTURING). See the design
spec in ``voice-notes-agent-build-plan.md`` and the module map in ``README.md``.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
