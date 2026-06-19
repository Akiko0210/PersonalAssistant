"""Tests for storage/store listing, RAG chunking, and the §9 paths layout."""

from __future__ import annotations

import json
from datetime import date, datetime

from voice_notes_agent.paths import Paths
from voice_notes_agent.storage.rag import chunk_text
from voice_notes_agent.storage.store import Store


def _make_session(paths: Paths, session_id: str, started: datetime, *, title: str, summary: str = ""):
    d = paths.session_dir(session_id, started)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "started": started.isoformat(),
                "status": "finalized",
                "title": title,
            }
        ),
        encoding="utf-8",
    )
    if summary:
        (d / "summary.md").write_text(summary, encoding="utf-8")
    return d


def test_paths_layout(tmp_path):
    paths = Paths.resolve(tmp_path)
    assert paths.sessions.exists()
    assert paths.chroma.exists()
    assert paths.logs.exists()


def test_list_and_filter_sessions(tmp_path):
    paths = Paths.resolve(tmp_path)
    store = Store(paths)
    _make_session(paths, "aaa", datetime(2026, 6, 1, 9, 0), title="June one")
    _make_session(paths, "bbb", datetime(2026, 6, 19, 14, 0), title="June nineteen")

    everything = store.list_sessions()
    assert {s.session_id for s in everything} == {"aaa", "bbb"}

    only_19 = store.list_sessions(date_from=date(2026, 6, 10))
    assert [s.session_id for s in only_19] == ["bbb"]


def test_find_and_get_summary(tmp_path):
    paths = Paths.resolve(tmp_path)
    store = Store(paths)
    _make_session(paths, "ccc", datetime(2026, 6, 19, 8, 0), title="With summary", summary="# Hi\n")
    info = store.find("ccc")
    assert info is not None and info.title == "With summary"
    found = store.get_summary("ccc")
    assert found is not None
    text, path = found
    assert text.startswith("# Hi")
    assert path.name == "summary.md"


def test_save_summary_updates_title(tmp_path):
    paths = Paths.resolve(tmp_path)
    store = Store(paths)
    d = _make_session(paths, "ddd", datetime(2026, 6, 19, 8, 0), title="Old")
    store.save_summary(d, "# New\n", title="New title")
    info = store.find("ddd")
    assert info is not None and info.title == "New title"


def test_chunk_text_overlap_and_bounds():
    text = " ".join(f"Sentence number {i}." for i in range(40))
    chunks = chunk_text(text, max_chars=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)  # window + overlap headroom
    assert chunk_text("") == []
