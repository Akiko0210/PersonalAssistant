"""Tests for the knowledge base's pure logic — transcript windowing, timestamp
formatting, streamed hashing, and the boot scan's deferred-media check.

Nothing here loads Whisper, Chroma, or the embedding model: the ingest methods
are exercised through their helpers, and stores are built with ``__new__`` to
skip the directory-creating constructor.
"""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import config as cfg
from brain import agents
from stores.knowledge import KnowledgeStore, _collection_name, _hms


def seg(text, start):
    """A stand-in for a faster-whisper segment (only .text/.start are read)."""
    return SimpleNamespace(text=text, start=start)


class FakeCol:
    """A Chroma collection double: records upserts, serves canned hits. Keeps
    these tests loading neither Chroma nor the embedding model."""

    def __init__(self, hits=()):
        self.upserts = []                 # (ids, documents, metadatas)
        self.hits = list(hits)            # (doc, meta, distance)
        self.queries = 0

    def upsert(self, ids, documents, metadatas):
        self.upserts.append((list(ids), list(documents), list(metadatas)))

    def count(self):
        return len(self.hits) or sum(len(u[0]) for u in self.upserts)

    def query(self, query_texts, n_results):
        self.queries += 1
        rows = self.hits[:n_results]
        return {"documents": [[d for d, m, dist in rows]],
                "metadatas": [[m for d, m, dist in rows]],
                "distances": [[dist for d, m, dist in rows]]}


def fake_store(cols):
    """A KnowledgeStore whose _col_for serves from `cols` (name -> FakeCol),
    creating on demand so tests can also assert which names were touched."""
    store = KnowledgeStore.__new__(KnowledgeStore)
    store._whisper = None
    store._cols = {}
    store._col_for = lambda name: cols.setdefault(name, FakeCol())
    return store


class TestHms(unittest.TestCase):
    def test_under_an_hour_omits_the_hour(self):
        self.assertEqual(_hms(0), "0:00")
        self.assertEqual(_hms(9), "0:09")
        self.assertEqual(_hms(872), "14:32")

    def test_past_an_hour_includes_it(self):
        self.assertEqual(_hms(3600), "1:00:00")
        self.assertEqual(_hms(5025), "1:23:45")

    def test_fractional_seconds_truncate(self):
        self.assertEqual(_hms(14.9), "0:14")

    def test_none_and_negative_are_tolerated(self):
        # Whisper can hand back a null start on an empty segment.
        self.assertEqual(_hms(None), "0:00")
        self.assertEqual(_hms(-5), "0:00")


class TestMediaWindows(unittest.TestCase):
    """Whisper emits a few seconds of speech per segment; these get grouped into
    chunk-sized windows so each embedded vector carries real context."""

    def windows(self, segments, chars=20, overlap=5):
        with mock.patch.object(cfg, "KB_CHUNK_CHARS", chars), \
                mock.patch.object(cfg, "KB_CHUNK_OVERLAP", overlap):
            return list(KnowledgeStore._media_windows(segments))

    def test_segments_group_into_windows_with_overlap(self):
        out = self.windows([
            seg("alpha", 0.0), seg("bravo", 5.0), seg("charlie", 10.0),
            seg("delta", 15.0), seg("echo", 20.0),
        ])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0], "alpha bravo charlie")
        # The tail of one window opens the next, so a passage split across the
        # boundary is still retrievable whole.
        self.assertEqual(out[1][0], "charlie delta echo")

    def test_window_is_tagged_with_its_first_segments_start(self):
        out = self.windows([
            seg("alpha", 0.0), seg("bravo", 5.0), seg("charlie", 10.0),
            seg("delta", 15.0), seg("echo", 20.0),
        ])
        self.assertEqual(out[0][1], {"t": 0.0})
        self.assertEqual(out[1][1], {"t": 10.0})  # "charlie", not "delta"

    def test_short_transcript_is_one_window(self):
        out = self.windows([seg("just this", 3.0)])
        self.assertEqual(out, [("just this", {"t": 3.0})])

    def test_blank_segments_are_dropped(self):
        out = self.windows([seg("", 0.0), seg("   ", 1.0), seg("real", 2.0)])
        # A blank leading segment must not become the window's timestamp.
        self.assertEqual(out, [("real", {"t": 2.0})])

    def test_no_segments_yields_nothing(self):
        self.assertEqual(self.windows([]), [])
        self.assertEqual(self.windows([seg("", 0.0)]), [])

    def test_segment_longer_than_a_window_still_emitted(self):
        # Guards the buf-empty check: an oversized segment must not be dropped
        # or loop forever. The character chunker splits it downstream.
        long_text = "x" * 50
        out = self.windows([seg(long_text, 7.0)])
        self.assertEqual(out, [(long_text, {"t": 7.0})])

    def test_start_times_are_floats(self):
        # Chroma metadata must be JSON-scalar; an int start is fine but None
        # would not be, so it is normalised at the boundary.
        out = self.windows([seg("hi", None)])
        self.assertEqual(out[0][1], {"t": 0.0})
        self.assertIsInstance(out[0][1]["t"], float)


class TestTranscriptionProgress(unittest.TestCase):
    """Progress reporting during a long transcription. Without it, a 74-minute
    recording is one log line followed by half an hour of silence."""

    @staticmethod
    def timed(text, start, end):
        return SimpleNamespace(text=text, start=start, end=end)

    def drain(self, segments, total):
        calls = []
        out = list(KnowledgeStore._with_progress(
            segments, total, "talk.mp4", lambda name, pct: calls.append(pct)))
        return out, calls

    def test_segments_pass_through_untouched(self):
        segs = [self.timed("a", 0, 10), self.timed("b", 10, 20)]
        out, _ = self.drain(iter(segs), 100)
        self.assertEqual(out, segs)

    def test_reports_percentage_as_it_advances(self):
        segs = iter([self.timed("x", 0, 6), self.timed("x", 6, 30),
                     self.timed("x", 30, 61), self.timed("x", 61, 100)])
        _, calls = self.drain(segs, 100)
        self.assertEqual(calls, [6, 30, 61, 100])

    def test_does_not_report_the_same_step_twice(self):
        # Many short segments inside one 5% band must not spam the console.
        segs = iter([self.timed("x", i, i + 1) for i in range(0, 20)])
        _, calls = self.drain(segs, 100)
        self.assertEqual(calls, sorted(set(calls)))
        self.assertTrue(all(b - a >= 5 for a, b in zip(calls, calls[1:])))

    def test_unknown_duration_reports_nothing_rather_than_dividing_by_zero(self):
        segs = [self.timed("a", 0, 10)]
        out, calls = self.drain(iter(segs), None)
        self.assertEqual(out, segs)   # still yields every segment
        self.assertEqual(calls, [])

    def test_percentage_never_exceeds_one_hundred(self):
        # Whisper can run a hair past the container's reported duration.
        segs = iter([self.timed("x", 0, 105)])
        _, calls = self.drain(segs, 100)
        self.assertEqual(calls, [100])


class TestFileHash(unittest.TestCase):
    def test_matches_a_whole_file_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            payload = b"\x00\x01video-bytes\xff" * 1000
            path.write_bytes(payload)
            self.assertEqual(KnowledgeStore._file_hash(path),
                             hashlib.sha256(payload).hexdigest())

    def test_reads_in_blocks_rather_than_slurping(self):
        # The point of the helper: a multi-gigabyte video must never be loaded
        # whole just to identify it.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.mp4"
            path.write_bytes(b"z" * (3 * (1 << 20)))
            with mock.patch.object(Path, "read_bytes") as slurp:
                KnowledgeStore._file_hash(path)
            slurp.assert_not_called()


class TestPendingMedia(unittest.TestCase):
    """The boot scan skips transcription but must still notice new video, so a
    dropped-in file is never silently ignored."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        patcher = mock.patch.object(cfg, "KNOWLEDGE_DIR", self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = KnowledgeStore.__new__(KnowledgeStore)

    def touch(self, name):
        (self.dir / name).write_bytes(b"data")

    def test_new_media_is_reported(self):
        self.touch("course.mp4")
        self.touch("podcast.mp3")
        self.assertEqual(self.store._pending_media({}), ["course.mp4", "podcast.mp3"])

    def test_already_ingested_media_is_not_reported(self):
        self.touch("course.mp4")
        manifest = {"abc123": {"source": "course.mp4", "title": "course"}}
        self.assertEqual(self.store._pending_media(manifest), [])

    def test_documents_are_not_reported_as_media(self):
        self.touch("book.pdf")
        self.touch("notes.txt")
        self.assertEqual(self.store._pending_media({}), [])


class TestCitations(unittest.TestCase):
    """search() renders each hit's locator: a page for books, a timestamp for
    video, so the user can jump to the moment."""

    def search_over(self, metas):
        cols = {cfg.KNOWLEDGE_COLLECTION: FakeCol(
            [("some passage", meta, 0.1) for meta in metas])}
        store = fake_store(cols)
        with mock.patch.object(KnowledgeStore, "_load_manifest",
                               return_value={"h": {"source": "x"}}):
            return store.search("anything")

    def test_video_hit_cites_a_timestamp(self):
        out = self.search_over([{"title": "Options Course", "t": 872.0}])
        self.assertIn("[Options Course, 14:32]", out)

    def test_pdf_hit_still_cites_a_page(self):
        out = self.search_over([{"title": "Trading Book", "page": 42}])
        self.assertIn("[Trading Book, p.42]", out)

    def test_zero_timestamp_is_not_mistaken_for_missing(self):
        # t=0.0 is falsy; the opening of a video must still be cited.
        out = self.search_over([{"title": "Intro", "t": 0.0}])
        self.assertIn("[Intro, 0:00]", out)

    def test_plain_text_hit_cites_the_title_alone(self):
        out = self.search_over([{"title": "Scratch Notes"}])
        self.assertIn("[Scratch Notes]", out)


class TestIngestTargeting(unittest.TestCase):
    """A targeted ingest routes NEW files into that agent's private collection
    and records the target in the manifest; a known hash is never re-embedded
    under a second target (one document, one collection)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        for name, value in (("KNOWLEDGE_DIR", self.dir),
                            ("KNOWLEDGE_MANIFEST", self.dir / "manifest.json")):
            p = mock.patch.object(cfg, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.cols = {}
        self.store = fake_store(self.cols)

    def ingest(self, target=cfg.COMMON_COLLECTION):
        return self.store.ingest_folder(include_media=False, target=target)

    def test_private_target_routes_to_the_agents_collection(self):
        (self.dir / "plan.txt").write_text("the trading plan", encoding="utf-8")
        self.ingest(target="tom")
        self.assertIn(cfg.agent_knowledge_collection("tom"), self.cols)
        self.assertNotIn(cfg.KNOWLEDGE_COLLECTION, self.cols)
        manifest = self.store._load_manifest()
        (entry,) = manifest.values()
        self.assertEqual(entry["collection"], "tom")

    def test_default_target_is_common_and_untagged_entries_stay_valid(self):
        (self.dir / "notes.txt").write_text("shared text", encoding="utf-8")
        self.ingest()
        (entry,) = self.store._load_manifest().values()
        self.assertEqual(entry["collection"], cfg.COMMON_COLLECTION)
        self.assertIn(cfg.KNOWLEDGE_COLLECTION, self.cols)

    def test_known_hash_is_not_re_embedded_under_a_new_target(self):
        (self.dir / "plan.txt").write_text("the trading plan", encoding="utf-8")
        self.ingest()                       # lands in common
        summary = self.ingest(target="tom")  # same bytes, new target
        self.assertNotIn(cfg.agent_knowledge_collection("tom"), self.cols)
        self.assertIn("NOT moved", summary)  # honest refusal, not silence
        self.assertIn("forget", summary)

    def test_boot_scan_does_not_nag_about_private_files(self):
        (self.dir / "plan.txt").write_text("private plan", encoding="utf-8")
        self.ingest(target="tom")
        summary = self.ingest()  # the every-boot common scan
        self.assertNotIn("NOT moved", summary)

    def test_unknown_target_is_refused_before_any_work(self):
        with self.assertRaises(ValueError):
            self.ingest(target="mallory")


class TestScopedSearch(unittest.TestCase):
    """Isolation is which collections get queried: a caller reaches common plus
    its own (and granted) private collections, and nothing else."""

    def setUp(self):
        self.cols = {}
        self.store = fake_store(self.cols)

    def manifest(self, *entries):
        return mock.patch.object(
            KnowledgeStore, "_load_manifest",
            return_value={f"h{i}": e for i, e in enumerate(entries)})

    def test_caller_reaches_common_plus_its_own_private(self):
        self.cols[cfg.KNOWLEDGE_COLLECTION] = FakeCol(
            [("common passage", {"title": "Book"}, 0.5)])
        self.cols[cfg.agent_knowledge_collection("tom")] = FakeCol(
            [("private passage", {"title": "Tom Plan"}, 0.1)])
        with self.manifest({"source": "a"}, {"source": "b", "collection": "tom"}):
            out = self.store.search("anything", caller="tom")
        self.assertIn("Tom Plan", out)
        self.assertIn("Book", out)

    def test_other_agents_never_touch_a_private_collection(self):
        self.cols[cfg.KNOWLEDGE_COLLECTION] = FakeCol(
            [("common passage", {"title": "Book"}, 0.5)])
        private = self.cols[cfg.agent_knowledge_collection("tom")] = FakeCol(
            [("private passage", {"title": "Tom Plan"}, 0.1)])
        with self.manifest({"source": "a"}, {"source": "b", "collection": "tom"}):
            out = self.store.search("anything", caller="alice")
        self.assertNotIn("Tom Plan", out)
        # Structural: alice's search never even queried tom's collection.
        self.assertEqual(private.queries, 0)

    def test_a_reads_grant_opens_exactly_that_collection(self):
        self.cols[cfg.agent_knowledge_collection("tom")] = FakeCol(
            [("private passage", {"title": "Tom Plan"}, 0.1)])
        original = agents.AGENTS["alice"]["reads"]
        agents.AGENTS["alice"]["reads"] = ("tom",)
        self.addCleanup(lambda: agents.AGENTS["alice"].__setitem__(
            "reads", original))
        with self.manifest({"source": "b", "collection": "tom"}):
            out = self.store.search("anything", caller="alice")
        self.assertIn("Tom Plan", out)

    def test_results_merge_by_distance_across_collections(self):
        self.cols[cfg.KNOWLEDGE_COLLECTION] = FakeCol(
            [("far common", {"title": "Far"}, 0.9)])
        self.cols[cfg.agent_knowledge_collection("tom")] = FakeCol(
            [("near private", {"title": "Near"}, 0.1)])
        with self.manifest({"source": "a"}, {"source": "b", "collection": "tom"}), \
                mock.patch.object(cfg, "KB_SEARCH_RESULTS", 1):
            out = self.store.search("anything", caller="tom")
        self.assertIn("Near", out)
        self.assertNotIn("Far", out)

    def test_manifest_guard_is_scoped_to_the_caller(self):
        # Only a Tom-private source exists: Tom searches, Alice (and a
        # caller-less probe) get the honest nothing-ingested message.
        self.cols[cfg.agent_knowledge_collection("tom")] = FakeCol(
            [("private passage", {"title": "Tom Plan"}, 0.1)])
        with self.manifest({"source": "b", "collection": "tom"}):
            self.assertIn("Tom Plan", self.store.search("x", caller="tom"))
            self.assertIn("No trading knowledge",
                          self.store.search("x", caller="alice"))
            self.assertIn("No trading knowledge", self.store.search("x"))

    def test_forget_deletes_from_the_entrys_own_collection(self):
        deleted = []
        col = FakeCol()
        col.delete = lambda where: deleted.append(where)
        self.cols[cfg.agent_knowledge_collection("tom")] = col
        manifest = {"h1": {"source": "plan.txt", "title": "Plan",
                           "collection": "tom"}}
        with mock.patch.object(KnowledgeStore, "_load_manifest",
                               return_value=manifest), \
                mock.patch.object(KnowledgeStore, "_save_manifest"):
            self.store.forget("plan")
        self.assertEqual(deleted, [{"source": "plan.txt"}])
        self.assertEqual(set(self.cols),
                         {cfg.agent_knowledge_collection("tom")})


class TestCollectionName(unittest.TestCase):
    def test_common_maps_to_the_shared_collection(self):
        self.assertEqual(_collection_name(cfg.COMMON_COLLECTION),
                         cfg.KNOWLEDGE_COLLECTION)

    def test_agent_key_maps_to_its_private_collection(self):
        self.assertEqual(_collection_name("tom"),
                         cfg.agent_knowledge_collection("tom"))


if __name__ == "__main__":
    unittest.main()
