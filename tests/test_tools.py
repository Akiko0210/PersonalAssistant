"""Tests for the agent tool dispatch layer (§8) with fakes — no models or network."""

from __future__ import annotations

from voice_notes_agent.agent.tools import TOOL_SPECS, NoteTools


class FakeController:
    def __init__(self):
        self.state = "listening"
        self.session = None

    def start_capture(self):
        self.state = "capturing"
        self.session = "sess123"
        return self.session

    def stop_capture(self):
        self.state = "listening"
        sid = self.session or ""
        return sid, "finalized"

    def active_session_id(self):
        return self.session

    def current_state(self):
        return self.state

    def listening_mode(self):
        return "both"


class FakeIndex:
    def __init__(self):
        self.indexed = []

    def index_session(self, **kw):
        self.indexed.append(kw)
        return 1

    def search(self, query, k=5):
        from voice_notes_agent.storage.rag import SearchHit

        return [SearchHit(text=f"hit for {query}", session_id="s1", timestamp="t", score=0.9)]


def test_tool_specs_cover_all_dispatch_names():
    spec_names = {s["name"] for s in TOOL_SPECS}
    expected = {
        "start_note_session",
        "stop_note_session",
        "summarize_session",
        "search_notes",
        "list_sessions",
        "get_session_summary",
        "get_status",
    }
    assert spec_names == expected


def test_start_and_stop_roundtrip():
    ctl = FakeController()
    tools = NoteTools(ctl, store=None, index=FakeIndex(), summarizer=None)  # type: ignore[arg-type]
    started = tools.start_note_session()
    assert started == {"session_id": "sess123"}
    assert ctl.current_state() == "capturing"
    stopped = tools.stop_note_session()
    assert stopped["status"] == "finalized"
    assert ctl.current_state() == "listening"


def test_search_notes_shapes_results():
    tools = NoteTools(FakeController(), store=None, index=FakeIndex(), summarizer=None)  # type: ignore[arg-type]
    out = tools.search_notes("budget", k=3)
    assert out["results"][0]["session_id"] == "s1"
    assert out["results"][0]["text"] == "hit for budget"


def test_get_status_reports_state():
    ctl = FakeController()
    tools = NoteTools(ctl, store=None, index=FakeIndex(), summarizer=None)  # type: ignore[arg-type]
    ctl.start_capture()
    status = tools.get_status()
    assert status == {
        "state": "capturing",
        "active_session_id": "sess123",
        "listening_mode": "both",
    }


def test_dispatch_unknown_tool():
    tools = NoteTools(FakeController(), store=None, index=FakeIndex(), summarizer=None)  # type: ignore[arg-type]
    assert "error" in tools.dispatch("nope", {})
