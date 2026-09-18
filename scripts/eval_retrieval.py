"""Offline retrieval eval: replay logged questions against a COPY of the store
and report where each expected exchange ranked. This is the instrument the
CONTEXT_* thresholds are tuned with — a change to the scorer, the keys or the
gate is judged here, on the failures that actually happened, before it is
judged by ear.

    python -m scripts.eval_retrieval                 # copy data/chroma, run the cases
    python -m scripts.eval_retrieval --backfill      # also run backfill_keys on the copy
    python -m scripts.eval_retrieval --copy DIR      # reuse an earlier copy
    python -m scripts.eval_retrieval --set CONTEXT_CANDIDATES=24   # sweep a knob
    python -m scripts.eval_retrieval --verbose       # the per-candidate tables too

Never opens data/chroma itself (the live agent holds it) and never takes the
single-instance lock: everything happens in the copy, which is kept — a live
Chroma handle blocks deleting it on Windows — and its path printed. Copy
while the agent is idle; a snapshot taken mid-write can be torn.

Cases (scripts/eval_cases.json): {label, owner, query, prev_user?, as_of,
window?: {since, until}, exclude_ts?: [prefix], expect_ts?: [prefix],
expect_ids?: [prefix], expect_text?: [substring], require: "any"|"all",
stretch?: bool}. A short query borrows `prev_user` the way _pull_context
does; `exclude_ts` names the verbatim tail; a stretch case is reported but
never fails the run (it is the trigger for the next lever, not a regression).
"""

import argparse
import json
import logging
import math
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import config as cfg


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default=Path(__file__).with_name("eval_cases.json"))
    ap.add_argument("--src", default=cfg.CHROMA_DIR, help="the live store to copy")
    ap.add_argument("--copy", help="an existing copy to reuse (default: fresh, in TEMP)")
    ap.add_argument("--backfill", action="store_true",
                    help="run backfill_keys on the copy first (evaluates the "
                         "companion keys before the live store gets them)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config knob for this run (any OVERRIDABLE name)")
    ap.add_argument("--owner", help="only this persona's cases")
    ap.add_argument("--verbose", action="store_true", help="context DEBUG tables")
    return ap.parse_args()


def space_probe(col, declared):
    """One stored vector, its nearest neighbour, the cosine between them by
    hand, and the distance Chroma reports: which formula matches is the
    space. Keeps D2 — similarity() and the store agreeing — verified after
    every Chroma upgrade."""
    got = col.get(limit=1, include=["embeddings"])
    if not got["ids"]:
        return "space probe skipped (empty collection)", True
    a = list(got["embeddings"][0])
    res = col.query(query_embeddings=[a], n_results=2, include=["distances"])
    if len(res["ids"][0]) < 2:
        return "space probe skipped (one record)", True
    b_id, d = res["ids"][0][1], res["distances"][0][1]
    b = list(col.get(ids=[b_id], include=["embeddings"])["embeddings"][0])
    cos = sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))
    seen = ("cosine" if abs(d - (1 - cos)) < 1e-3 else
            "l2" if abs(d - (2 - 2 * cos)) < 1e-3 else
            f"unknown (d={d:.4f}, cos={cos:.4f})")
    return f"store space: {seen}; chroma_store.SPACE: {declared}", seen == declared


def main():
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(name)-8s %(message)s")
    if args.verbose:
        logging.getLogger("context").setLevel(logging.DEBUG)
    overrides = dict(kv.split("=", 1) for kv in args.set)
    applied = cfg.apply_overrides(overrides)
    for k in overrides:
        print(f"{'set' if k in applied else 'IGNORED (not overridable)'}: "
              f"{k} = {getattr(cfg, k, overrides[k])!r}")

    copy = (Path(args.copy) if args.copy
            else Path(tempfile.mkdtemp(prefix="voice-agent-eval-")) / "chroma")
    if not copy.exists():
        shutil.copytree(args.src, copy)
    print(f"store copy: {copy}")
    cfg.CHROMA_DIR = copy  # before any store is built; chroma_store reads it per call

    from brain import agents, context
    from brain.memory import ConversationMemory
    from lib.dates import window_epochs
    from stores import chroma_store

    class AsOfMemory(ConversationMemory):
        """The store as it stood when the replayed question was asked. A pull
        must not see the exchanges that came after it — least of all the one
        the question itself produced, stamped the very same second, which
        would sit at rank 1 with a perfect lexical score and deflate every
        other record's relative BM25. Hence strictly-before. The cut is on
        `epoch`, so a persona's seeded day-summaries (no epoch) drop out of
        its dense list; the legacy archive has no epochs and is left whole."""
        now = None

        def _query_archive(self, name, query, rows, where=None):
            if self.now is not None and name != cfg.MEMORY_COLLECTION:
                cut = {"epoch": {"$lt": self.now}}
                where = {"$and": [where, cut]} if where else cut
            return super()._query_archive(name, query, rows, where)

        def _lexical_rows(self, name, query, n, window=None):
            rows = super()._lexical_rows(name, query, 10 * n, window)
            if self.now is not None:
                rows = [h for h in rows
                        if not isinstance(h.meta.get("epoch"), (int, float))
                        or h.meta["epoch"] < self.now]
            return rows[:n]

    memory = AsOfMemory()
    kb = SimpleNamespace(query_rows=lambda *a, **k: [])  # no Whisper, no knowledge load
    failures = 0

    line, ok = space_probe(memory._col_for(cfg.agent_memory_collection("alice")),
                           chroma_store.SPACE)
    print(("PASS " if ok else "FAIL ") + line)
    failures += 0 if ok else 2

    if args.backfill:
        t0 = time.monotonic()
        keyed = [memory.backfill_keys(k) for k in agents.AGENTS]
        added, stamped = (sum(x) for x in zip(*keyed))
        print(f"backfill: {added} companion(s) added, {stamped} record(s) stamped "
              f"({time.monotonic() - t0:.1f}s)")

    stamps = {}  # owner -> [(ts, id)] over parent records, for exclude_ts

    def ids_for(owner, prefixes):
        if owner not in stamps:
            got = memory._col_for(cfg.agent_memory_collection(owner)).get(include=["metadatas"])
            stamps[owner] = [(m.get("ts") or "", i)
                             for i, m in zip(got["ids"], got["metadatas"])
                             if not (m or {}).get("part")]
        return {i for ts, i in stamps[owner] if any(ts.startswith(p) for p in prefixes)}

    def matches(c, case):
        ts = str(c.meta.get("ts") or "")
        return (any(ts.startswith(p) for p in case.get("expect_ts", []))
                or any(str(c.id).startswith(p) for p in case.get("expect_ids", []))
                or any(s in str(c.doc) for s in case.get("expect_text", [])))

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    for case in cases:
        owner = case.get("owner", "alice")
        if args.owner and owner != args.owner:
            continue
        query = case["query"]
        if case.get("prev_user") and len(query.split()) < cfg.CONTEXT_SHORT_QUERY_WORDS:
            query = f"{case['prev_user']} {query}"
        now = memory.now = datetime.fromisoformat(case["as_of"]).timestamp()
        exclude = ids_for(owner, case.get("exclude_ts", []))
        window = case.get("window")
        if window:
            cands, _ = memory.rank(query, caller=owner, exclude_ids=exclude, now=now,
                                   window=window_epochs(since=window.get("since"),
                                                        until=window.get("until")))
            for k, c in enumerate(cands, 1):
                c.kept = k <= cfg.MEMORY_SEARCH_RESULTS  # what the tool would show
        else:
            pull = context.build_context(query, owner=owner, memory=memory, kb=kb,
                                         exclude_ids=exclude, now=now)
            cands = [c for c in pull.candidates if c.source == "conversation"]
        found = [(k, c) for k, c in enumerate(cands, 1) if matches(c, case)]
        expected = (len(case.get("expect_ts", [])) + len(case.get("expect_ids", []))
                    + len(case.get("expect_text", [])))
        kept = [c for _, c in found if c.kept]
        passed = (len(kept) >= max(1, expected) if case.get("require") == "all"
                  else bool(kept))
        tag = "PASS" if passed else ("SOFT" if case.get("stretch") else "FAIL")
        if not passed and not case.get("stretch"):
            failures += 1
        print(f"\n{tag} {case['label']}  ({owner}, {len(cands)} candidates)"
              f"{'  [stretch]' if case.get('stretch') else ''}")
        print(f"     query: {query[:100]!r}")
        if not found:
            print("     target: not among the candidates")
        for k, c in found:
            print(f"     target rank {k:3}: sim={c.sim:.2f} lex={c.lex:.2f} "
                  f"score={c.score:.2f} gate={c.gate!s:5} kept={c.kept!s:5} "
                  f"{str(c.meta.get('ts') or c.meta.get('date') or '')[:19]}")
        for k, c in enumerate(cands[:5], 1):
            print(f"       {k}. {'KEEP' if c.kept else 'drop':4} sim={c.sim:.2f} "
                  f"lex={c.lex:.2f} score={c.score:.2f} "
                  f"{' '.join(str(c.doc).split())[:66]!r}")
    print(f"\n{failures} failure(s). Store copy kept at {copy}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
