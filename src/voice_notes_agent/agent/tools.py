"""Agent tools — LLM function specs and their implementations (§8).

The agent is one Claude-driven voice agent that invokes note-taking and retrieval via
tool calls (§A2). This module is the bridge between those tool calls and the local
subsystems. It deliberately knows nothing about Pipecat or audio — it talks to a small
:class:`AgentController` protocol the app implements, plus the store/index/summarizer.

Tool surface (§8):
    start_note_session()                  -> { session_id }
    stop_note_session(session_id?)        -> { session_id, status }
    summarize_session(session_id)         -> { spoken_summary, full_summary, file_path }
    search_notes(query, k?)               -> { results: [...] }
    list_sessions(date_from?, date_to?)   -> { sessions: [...] }
    get_session_summary(session_id)       -> { full_summary, file_path }
    get_status()                          -> { state, active_session_id?, listening_mode }

``stop_note_session`` is also callable directly by a hardware button / wake word — not
only by the LLM (§8, FR-C8) — so the app calls the same :class:`NoteTools` method.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Protocol

from ..storage.rag import NotesIndex
from ..storage.store import Store
from ..summarize.summarizer import Summarizer

log = logging.getLogger(__name__)


class AgentController(Protocol):
    """What the tools need from the app to drive capture and report state."""

    def start_capture(self) -> str:
        """Enter CAPTURING; return the new session id."""

    def stop_capture(self) -> tuple[str, str]:
        """Leave CAPTURING; finalize. Return (session_id, status)."""

    def active_session_id(self) -> str | None: ...

    def current_state(self) -> str: ...

    def listening_mode(self) -> str: ...


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


class NoteTools:
    """Implements the §8 tools against the app controller + local storage."""

    def __init__(
        self,
        controller: AgentController,
        store: Store,
        index: NotesIndex,
        summarizer: Summarizer,
    ) -> None:
        self._c = controller
        self._store = store
        self._index = index
        self._summarizer = summarizer

    # -- tools ----------------------------------------------------------------
    def start_note_session(self) -> dict[str, Any]:
        session_id = self._c.start_capture()
        return {"session_id": session_id}

    def stop_note_session(self, session_id: str | None = None) -> dict[str, Any]:
        sid, status = self._c.stop_capture()
        return {"session_id": sid, "status": status}

    def summarize_session(self, session_id: str) -> dict[str, Any]:
        """Local transcript -> Claude summary -> save summary.md + index in Chroma (§8)."""
        info = self._store.find(session_id)
        if info is None:
            return {"error": f"session {session_id} not found"}
        transcript = (
            info.transcript_path.read_text(encoding="utf-8")
            if info.transcript_path.exists()
            else ""
        )
        # Compute speech length from transcript length proxy is unreliable; the app
        # passes real seconds when it calls summarize directly. For the tool path we
        # summarize whenever there is any transcript text (FR-S5 handled in app flow).
        if not transcript.strip():
            return {
                "session_id": session_id,
                "spoken_summary": "That session had no speech to summarize.",
                "full_summary": "",
                "file_path": "",
            }
        result = self._summarizer.summarize(transcript)
        markdown = result.to_markdown(self._summarizer._cfg.summary.full_template)
        path = self._store.save_summary(info.dir, markdown, title=result.title)
        self._index.index_session(
            session_id=session_id,
            started=info.started,
            transcript=transcript,
            summary=markdown,
        )
        return {
            "session_id": session_id,
            "spoken_summary": result.spoken_summary,
            "full_summary": markdown,
            "file_path": str(path),
        }

    def search_notes(self, query: str, k: int = 5) -> dict[str, Any]:
        hits = self._index.search(query, k=k)
        return {
            "results": [
                {
                    "text": h.text,
                    "session_id": h.session_id,
                    "timestamp": h.timestamp,
                    "score": round(h.score, 4),
                }
                for h in hits
            ]
        }

    def list_sessions(
        self, date_from: str | None = None, date_to: str | None = None
    ) -> dict[str, Any]:
        sessions = self._store.list_sessions(_parse_date(date_from), _parse_date(date_to))
        return {
            "sessions": [
                {"session_id": s.session_id, "date": s.date, "title": s.title}
                for s in sessions
            ]
        }

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        found = self._store.get_summary(session_id)
        if found is None:
            return {"error": f"no summary for session {session_id}"}
        text, path = found
        return {"full_summary": text, "file_path": str(path)}

    def get_status(self) -> dict[str, Any]:
        return {
            "state": self._c.current_state(),
            "active_session_id": self._c.active_session_id(),
            "listening_mode": self._c.listening_mode(),
        }

    # -- dispatch + specs -----------------------------------------------------
    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke a tool by name with a kwargs dict (used by the conversation loop)."""
        fn = {
            "start_note_session": lambda a: self.start_note_session(),
            "stop_note_session": lambda a: self.stop_note_session(a.get("session_id")),
            "summarize_session": lambda a: self.summarize_session(a["session_id"]),
            "search_notes": lambda a: self.search_notes(a["query"], int(a.get("k", 5))),
            "list_sessions": lambda a: self.list_sessions(a.get("date_from"), a.get("date_to")),
            "get_session_summary": lambda a: self.get_session_summary(a["session_id"]),
            "get_status": lambda a: self.get_status(),
        }.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(arguments)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("tool %s failed", name)
            return {"error": str(exc)}


# Anthropic tool specifications (§8). Shared by the conversation loop and any direct
# Claude calls. Mirrors the function signatures above.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "start_note_session",
        "description": "Begin a local VAD-gated note recording session. Capture is "
        "handled locally until stopped; stay silent after calling this.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "stop_note_session",
        "description": "End the active note recording session and finalize its transcript.",
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "summarize_session",
        "description": "Summarize a finished session's transcript, save it, and index it. "
        "Returns a concise spoken summary to read back, plus the full summary.",
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_notes",
        "description": "Retrieve the most relevant note chunks for a query via local RAG.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_sessions",
        "description": "List past note sessions, optionally filtered by ISO date range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_session_summary",
        "description": "Fetch the full stored summary for a session id.",
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_status",
        "description": "Report the agent's current state, active session, and listening mode.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]
