"""Tests for the dashboard's knowledge upload + ingest-job endpoints.

The upload path is the one place the dashboard takes untrusted input and writes
a file, so the filename guards are covered thoroughly. The job side is checked
for its state machine and for refusing to run beside a live agent — the rule
that keeps two processes from writing the same Chroma store.
"""

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config as cfg
from web import server as dashboard
from lib.single_instance import AlreadyRunning


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        patcher = mock.patch.object(cfg, "KNOWLEDGE_DIR", self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def send(self, name, payload=b"data", length=None):
        length = len(payload) if length is None else length
        return dashboard.save_upload(name, io.BytesIO(payload), length)

    # --- happy path ---------------------------------------------------------
    def test_saves_an_mp4(self):
        status, body = self.send("course.mp4", b"\x00fake video bytes")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertTrue(body["media"])
        self.assertEqual((self.dir / "course.mp4").read_bytes(), b"\x00fake video bytes")

    def test_document_is_not_flagged_as_media(self):
        status, body = self.send("book.pdf", b"%PDF-1.7 ...")
        self.assertEqual(status, 200)
        self.assertFalse(body["media"])

    def test_larger_than_one_block_is_streamed_whole(self):
        payload = bytes(range(256)) * 8192  # 2 MiB, spans several read blocks
        status, _ = self.send("big.mp4", payload)
        self.assertEqual(status, 200)
        self.assertEqual((self.dir / "big.mp4").read_bytes(), payload)

    def test_no_part_file_is_left_behind(self):
        self.send("course.mp4", b"bytes")
        self.assertEqual([p.name for p in self.dir.iterdir()], ["course.mp4"])

    # --- filename guards ----------------------------------------------------
    def test_path_traversal_is_refused(self):
        for name in ("../evil.mp4", "..\\evil.mp4", "sub/dir.mp4",
                     "C:\\Windows\\evil.mp4", ".."):
            status, body = self.send(name)
            self.assertEqual(status, 400, name)
            self.assertFalse(body["ok"], name)
        # Nothing escaped into (or landed in) the knowledge folder.
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_unsupported_extension_is_refused(self):
        for name in ("script.exe", "notes.docx", "noextension"):
            status, body = self.send(name)
            self.assertEqual(status, 400, name)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_extension_check_is_case_insensitive(self):
        status, _ = self.send("COURSE.MP4", b"bytes")
        self.assertEqual(status, 200)

    def test_empty_name_is_refused(self):
        self.assertEqual(self.send("")[0], 400)
        self.assertEqual(self.send("   ")[0], 400)

    # --- size / integrity ---------------------------------------------------
    def test_empty_upload_is_refused(self):
        self.assertEqual(self.send("course.mp4", b"", 0)[0], 400)

    def test_oversized_upload_is_refused_before_writing(self):
        status, _ = self.send("huge.mp4", b"x", dashboard.MAX_UPLOAD_BYTES + 1)
        self.assertEqual(status, 413)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_truncated_upload_is_discarded(self):
        # Client promised 500 bytes and sent 5: the partial file must not survive
        # as something an ingest would later treat as a real video.
        status, body = self.send("course.mp4", b"short", 500)
        self.assertEqual(status, 400)
        self.assertIn("early", body["error"])
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_existing_name_is_refused_rather_than_overwritten(self):
        (self.dir / "course.mp4").write_bytes(b"original")
        status, body = self.send("course.mp4", b"replacement")
        self.assertEqual(status, 409)
        self.assertEqual((self.dir / "course.mp4").read_bytes(), b"original")


class RemovePendingTests(unittest.TestCase):
    """Undoing a wrong upload. Only touches files no manifest entry claims."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.manifest = self.dir / "manifest.json"
        for p in (mock.patch.object(cfg, "KNOWLEDGE_DIR", self.dir),
                  mock.patch.object(cfg, "KNOWLEDGE_MANIFEST", self.manifest)):
            p.start()
            self.addCleanup(p.stop)

    def test_removes_a_pending_file(self):
        (self.dir / "wrong.pdf").write_bytes(b"oops")
        status, body = dashboard.remove_pending("wrong.pdf")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertFalse((self.dir / "wrong.pdf").exists())

    def test_refuses_an_already_ingested_file(self):
        # Its chunks are in Chroma; deleting the file alone would leave the
        # agent citing a source that no longer exists.
        (self.dir / "book.pdf").write_bytes(b"real")
        self.manifest.write_text('{"h": {"source": "book.pdf", "title": "book"}}',
                                 encoding="utf-8")
        status, body = dashboard.remove_pending("book.pdf")
        self.assertEqual(status, 409)
        self.assertIn("forget", body["error"])
        self.assertTrue((self.dir / "book.pdf").exists())

    def test_refuses_path_traversal(self):
        outside = self.dir.parent / "outside.pdf"
        outside.write_bytes(b"keep me")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        status, _ = dashboard.remove_pending("../outside.pdf")
        self.assertEqual(status, 400)
        self.assertTrue(outside.exists())

    def test_refuses_non_knowledge_files(self):
        # The manifest lives in the same folder — it must not be deletable.
        self.manifest.write_text("{}", encoding="utf-8")
        status, _ = dashboard.remove_pending("manifest.json")
        self.assertEqual(status, 400)
        self.assertTrue(self.manifest.exists())

    def test_missing_file_reports_cleanly(self):
        status, body = dashboard.remove_pending("never-here.mp4")
        self.assertEqual(status, 404)
        self.assertFalse(body["ok"])

    def test_empty_name_is_refused(self):
        self.assertEqual(dashboard.remove_pending("")[0], 400)


class PendingListingTests(unittest.TestCase):
    """api_knowledge surfaces uploaded-but-not-ingested files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        for p in (mock.patch.object(cfg, "KNOWLEDGE_DIR", self.dir),
                  mock.patch.object(cfg, "KNOWLEDGE_MANIFEST", self.dir / "manifest.json"),
                  mock.patch.object(dashboard, "agent_running", lambda: False)):
            p.start()
            self.addCleanup(p.stop)

    def test_new_media_is_listed_as_pending(self):
        (self.dir / "course.mp4").write_bytes(b"x" * 10)
        out = dashboard.api_knowledge()
        self.assertEqual(len(out["pending"]), 1)
        self.assertEqual(out["pending"][0]["name"], "course.mp4")
        self.assertEqual(out["pending"][0]["bytes"], 10)
        self.assertTrue(out["pending"][0]["media"])

    def test_ingested_file_is_not_pending(self):
        (self.dir / "course.mp4").write_bytes(b"x")
        (self.dir / "manifest.json").write_text(
            '{"abc": {"source": "course.mp4", "title": "course", "chunks": 5}}',
            encoding="utf-8")
        out = dashboard.api_knowledge()
        self.assertEqual(out["pending"], [])
        self.assertEqual(len(out["docs"]), 1)

    def test_manifest_and_partial_uploads_are_not_listed(self):
        (self.dir / "manifest.json").write_text("{}", encoding="utf-8")
        (self.dir / "course.mp4.part").write_bytes(b"half")
        self.assertEqual(dashboard.api_knowledge()["pending"], [])


class IngestJobTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(dashboard._set_job, state="idle", message="", file="",
                        started=None, finished=None)

    def test_refuses_to_start_twice(self):
        dashboard._set_job(state="running")
        result = dashboard.start_ingest()
        self.assertFalse(result["ok"])
        self.assertIn("already running", result["error"])

    def test_will_not_ingest_beside_a_live_agent(self):
        # The guard that matters: embedding writes the same Chroma store the
        # running agent has open.
        with mock.patch.object(dashboard, "SingleInstance") as lock:
            lock.return_value.acquire.side_effect = AlreadyRunning("held")
            dashboard._run_ingest()
        job = dashboard.job_status()
        self.assertEqual(job["state"], "error")
        self.assertIn("voice agent is running", job["message"])

    def test_lock_is_released_when_ingest_fails(self):
        # A crashed ingest must not leave the lock held, or the agent could
        # never start again without killing the dashboard.
        with mock.patch.object(dashboard, "SingleInstance") as lock_cls:
            lock = lock_cls.return_value.acquire.return_value
            with mock.patch.dict("sys.modules", {"stores.knowledge": mock.MagicMock()}) as mods:
                mods["stores.knowledge"].KnowledgeStore.side_effect = RuntimeError("boom")
                dashboard._run_ingest()
            lock.release.assert_called_once()
        job = dashboard.job_status()
        self.assertEqual(job["state"], "error")
        self.assertIn("boom", job["message"])

    def test_successful_run_reports_the_summary_and_releases(self):
        with mock.patch.object(dashboard, "SingleInstance") as lock_cls:
            lock = lock_cls.return_value.acquire.return_value
            store = mock.MagicMock()
            store.ingest_folder.return_value = "Ingested 812 chunk(s) from 1 new file(s)."
            with mock.patch.dict("sys.modules", {"stores.knowledge": mock.MagicMock()}) as mods:
                mods["stores.knowledge"].KnowledgeStore.return_value = store
                dashboard._run_ingest()
            lock.release.assert_called_once()
        job = dashboard.job_status()
        self.assertEqual(job["state"], "done")
        self.assertIn("812 chunk", job["message"])
        self.assertTrue(job["finished"])
        # Media must be included: this is the path that exists to transcribe it.
        self.assertTrue(store.ingest_folder.call_args.kwargs["include_media"])

    def _run_with_progress(self, calls):
        """Drive _run_ingest with a store that replays the given on_progress
        calls, returning the job state seen after each one — sampled mid-run,
        since a finished job deliberately clears the current file."""
        seen = []

        def fake_ingest(include_media, on_progress):
            for args in calls:
                on_progress(*args)
                job = dashboard.job_status()
                seen.append((job["message"], job["file"]))
            return "summary"

        with mock.patch.object(dashboard, "SingleInstance"):
            store = mock.MagicMock()
            store.ingest_folder.side_effect = fake_ingest
            with mock.patch.dict("sys.modules", {"stores.knowledge": mock.MagicMock()}) as mods:
                mods["stores.knowledge"].KnowledgeStore.return_value = store
                dashboard._run_ingest()
        return [msg for msg, _ in seen], [f for _, f in seen]

    def test_progress_names_the_file_being_worked_on(self):
        messages, files = self._run_with_progress([("course.mp4",)])
        self.assertEqual(messages, ["Ingesting course.mp4…"])
        self.assertEqual(files, ["course.mp4"])

    def test_finished_job_clears_the_current_file(self):
        self._run_with_progress([("course.mp4",)])
        job = dashboard.job_status()
        self.assertEqual(job["state"], "done")
        self.assertEqual(job["file"], "")

    def test_progress_reports_transcription_percentage(self):
        # A 74-minute recording is a long silence otherwise; the percentage is
        # what tells the user it's moving.
        messages, _ = self._run_with_progress(
            [("talk.mp4",), ("talk.mp4", 25), ("talk.mp4", 90)])
        self.assertEqual(messages, ["Ingesting talk.mp4…",
                                    "Transcribing talk.mp4 — 25%",
                                    "Transcribing talk.mp4 — 90%"])


if __name__ == "__main__":
    unittest.main()
