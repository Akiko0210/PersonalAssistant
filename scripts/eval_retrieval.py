"""Offline retrieval eval: replay logged questions against a COPY of the store
and report where each expected exchange — or a fact built from it — ranked.
This is the instrument the CONTEXT_* thresholds are tuned with: a change to
the scorer, the keys, the gate, the query rewrite or the extractor is judged
here, on the failures that actually happened, before it is judged by ear.

    python -m scripts.eval_retrieval                  # copy data/chroma, run the cases
    python -m scripts.eval_retrieval --backfill       # also run backfill_keys on the copy
    python -m scripts.eval_retrieval --extract        # also run the fact extractor on the copy (model calls)
    python -m scripts.eval_retrieval --understand     # run the query rewrite per case (model calls)
    python -m scripts.eval_retrieval --copy DIR       # reuse an earlier copy (its facts included)
    python -m scripts.eval_retrieval --set CONTEXT_CANDIDATES=24   # sweep a knob
    python -m scripts.eval_retrieval --verbose        # the per-candidate tables too

Never opens data/chroma itself (the live agent holds it) and never takes the
single-instance lock: everything happens in the copy, which is kept — a live
Chroma handle blocks deleting it on Windows — and its path printed. Copy
while the agent is idle; a snapshot taken mid-write can be torn.

Cases (scripts/eval_cases.json): {label, owner, utterance, recent?: [lines],
query?, entities?: [keys], window?: {since, until}, as_of, exclude_ts?:
[prefix], expect_ts?: [prefix], expect_ids?: [prefix], expect_text?:
[substring], require: "any"|"all", stretch?: bool}. Without --understand the
pull uses `query` (default: the utterance), `entities` and `window` as
pinned; with it, brain/query.understand reads `recent` and the utterance
exactly as a live turn would. An expected exchange counts as found when it
is kept itself or a kept fact names it in `from`; `exclude_ts` names the
verbatim tail. A stretch case is reported but never fails the run.
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

try:
    from dotenv import load_dotenv
    load_dotenv(cfg.BASE_DIR / ".env")  # ANTHROPIC_API_KEY, for --extract / --understand
except ImportError:
    pass  # env vars still work


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default=Path(__file__).with_name("eval_cases.json"))
    ap.add_argument("--src", default=cfg.CHROMA_DIR, help="the live store to copy")
    ap.add_argument("--copy", help="an existing copy to reuse (default: fresh, in TEMP)")
    ap.add_argument("--backfill", action="store_true",
                    help="run backfill_keys on the copy first")
    ap.add_argument("--extract", action="store_true",
                    help="run the fact extractor over the copy's unread exchanges "
                         "first (one FACTS_MODEL call per exchange)")
    ap.add_argument("--understand", action="store_true",
                    help="rewrite each case's utterance with QUERY_MODEL, as a live "
                         "turn does, instead of using the pinned query")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config knob for this run (any OVERRIDABLE name)")
    ap.add_argument("--owner", help="only this persona's cases (and extraction)")
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
    from brain.facts import FactStore, catch_up, extractor
    from brain.memory import ConversationMemory
    from brain.query import Query, rewriter, understand
    from lib.dates import window_epochs
    from stores import chroma_store

    class AsOf:
        """A store as it stood when the replayed question was asked. A pull
        must not see records from after it — least of all the exchange the
        question itself produced, stamped the very same second, which would
        sit at rank 1 with a perfect lexical score and deflate every other
        record's relative BM25. Hence strictly-before. The cut is on `epoch`,
        so a persona's seeded day-summaries (no epoch) drop out of its dense
        list; the legacy archive has no epochs and is left whole."""
        now = None

        def _query_archive(self, name, query, rows, where=None):
            if self.now is not None and name != cfg.MEMORY_COLLECTION:
                cut = {"epoch": {"$lt": self.now}}
                where = {"$and": [where, cut]} if where else cut
            return super()._query_archive(name, query, rows, where)

        def _lexical_rows(self, name, query, n, keep=None):
            rows = super()._lexical_rows(name, query, 10 * n, keep)
            if self.now is not None:
                rows = [h for h in rows
                        if not isinstance(h.meta.get("epoch"), (int, float))
                        or h.meta["epoch"] < self.now]
            return rows[:n]

    class AsOfMemory(AsOf, ConversationMemory):
        pass

    class AsOfFacts(AsOf, FactStore):
        """Current as of `now`: established before it, and either still in
        force or superseded only later."""

        def _current_where(self):
            if self.now is None:
                return {"superseded_by": ""}
            # >=: the question's own exchange may be what superseded a fact,
            # and the pull ran before that exchange existed.
            return {"$and": [{"epoch": {"$lt": self.now}},
                             {"$or": [{"superseded_by": ""},
                                      {"superseded_epoch": {"$gte": self.now}}]}]}

        def _is_current(self, meta):
            if self.now is None:
                return meta.get("superseded_by") == ""
            return ((meta.get("epoch") or math.inf) < self.now
                    and (meta.get("superseded_by") == ""
                         or (meta.get("superseded_epoch") or 0) >= self.now))

        def entities(self, owner):
            self._entities.pop(owner, None)  # the cache cannot outlive `now`
            return super().entities(owner)

    memory, facts = AsOfMemory(), AsOfFacts()
    kb = SimpleNamespace(query_rows=lambda *a, **k: [])  # no Whisper, no knowledge load
    owners = [args.owner] if args.owner else list(agents.AGENTS)
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

    from brain.llm.main import make_client  # routes each model to its provider

    if args.extract:
        ask_facts = extractor(make_client(cfg.FACTS_MODEL))
        for k in owners:
            t0 = time.monotonic()
            read, failed = catch_up(
                ask_facts, memory, facts, k,
                progress=lambda i, n, k=k: print(f"  extract {k}: {i}/{n}", flush=True))
            print(f"extract {k}: {read} exchange(s) read, {failed} failed "
                  f"({time.monotonic() - t0:.0f}s)")
            col = facts._col_for(cfg.agent_facts_collection(k))
            got = col.get(where={"superseded_by": ""}, include=["documents", "metadatas"]) \
                if col.count() else {"ids": []}
            rows = sorted(zip(got.get("documents") or [], got.get("metadatas") or []),
                          key=lambda dm: dm[1].get("epoch") or 0)
            print(f"  {len(rows)} current fact(s) for {k}:")
            for doc, meta in rows[-60:]:
                print(f"    [{str(meta.get('ts'))[:16]}] {' '.join(doc.split())[:110]}")

    ask_query = rewriter(make_client(cfg.QUERY_MODEL)) if args.understand else None

    def clock(epoch):
        return datetime.fromtimestamp(epoch).strftime("%m/%d %H:%M") if epoch > 0 else "start"

    stamps = {}  # owner -> {parent id: ts}

    def ts_of(owner):
        if owner not in stamps:
            got = memory._col_for(cfg.agent_memory_collection(owner)).get(include=["metadatas"])
            stamps[owner] = {i: (m or {}).get("ts") or ""
                             for i, m in zip(got["ids"], got["metadatas"])
                             if not (m or {}).get("part")}
        return stamps[owner]

    def covered(c, case, owner):
        """The expectations candidate `c` satisfies: its own stamp, id or
        text — or, for a fact, the stamps of the exchanges in its `from`."""
        own = [str(c.meta.get("ts") or "")]
        if c.source == "fact":
            own += [ts_of(owner).get(i, "") for i in str(c.meta.get("source_ids", "")).split(",")]
        hits = {("ts", p) for p in case.get("expect_ts", []) if any(s.startswith(p) for s in own)}
        hits |= {("id", p) for p in case.get("expect_ids", []) if str(c.id).startswith(p)}
        hits |= {("text", s) for s in case.get("expect_text", []) if s in str(c.doc)}
        return hits

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    for case in cases:
        owner = case.get("owner", "alice")
        if owner not in owners:
            continue
        now = memory.now = facts.now = datetime.fromisoformat(case["as_of"]).timestamp()
        utterance = case.get("utterance") or case["query"]
        exclude = {i for i, ts in ts_of(owner).items()
                   if any(ts.startswith(p) for p in case.get("exclude_ts", []))}
        if ask_query is not None:
            recent = case.get("recent")
            if recent is None and exclude:
                # The verbatim tail IS what the rewrite read on the live turn.
                got = memory._col_for(cfg.agent_memory_collection(owner)).get(
                    ids=sorted(exclude), include=["documents"])
                docs = sorted(zip((ts_of(owner)[i] for i in got["ids"]), got["documents"]))
                recent = [line for _, d in docs for line in d.splitlines()]
            q = understand(ask_query, utterance, recent or [], now=case["as_of"],
                           known_entities=facts.entities(owner))
        else:
            w = case.get("window") or {}
            q = Query(text=case.get("query") or utterance,
                      window=window_epochs(since=w.get("since"), until=w.get("until")) if w else None,
                      entities=tuple(case.get("entities", [])))
        pull = context.build_context(q, owner=owner, memory=memory, facts=facts, kb=kb,
                                     exclude_ids=exclude, now=now)
        cands = [c for c in pull.candidates if c.source != "knowledge"]
        expected = ({("ts", p) for p in case.get("expect_ts", [])}
                    | {("id", p) for p in case.get("expect_ids", [])}
                    | {("text", s) for s in case.get("expect_text", [])})
        found = [(k, c, covered(c, case, owner)) for k, c in enumerate(cands, 1)]
        found = [(k, c, hit) for k, c, hit in found if hit]
        kept_cover = set().union(*(hit for _, c, hit in found if c.kept)) if found else set()
        passed = (kept_cover >= expected if case.get("require") == "all"
                  else bool(kept_cover))
        tag = "PASS" if passed else ("SOFT" if case.get("stretch") else "FAIL")
        if not passed and not case.get("stretch"):
            failures += 1
        print(f"\n{tag} {case['label']}  ({owner}, {len(cands)} candidates)"
              f"{'  [stretch]' if case.get('stretch') else ''}")
        window = f"{clock(q.window[0])}-{clock(q.window[1])}" if q.window else "no"
        print(f"     query: {q.text[:100]!r}  window={window} "
              f"entities={list(q.entities)} ({q.source})")
        if case.get("require") == "all":
            print(f"     covered {len(kept_cover)}/{len(expected)} expectation(s)")
        if not found:
            print("     target: not among the candidates")
        for k, c, hit in found:
            print(f"     target rank {k:3} {c.source:12} sim={c.sim:.2f} lex={c.lex:.2f} "
                  f"score={c.score:.2f} gate={c.gate!s:5} kept={c.kept!s:5} "
                  f"{str(c.meta.get('ts') or c.meta.get('date') or '')[:19]} "
                  f"covers {len(hit)}")
        for k, c in enumerate(cands[:5], 1):
            print(f"       {k}. {'KEEP' if c.kept else 'drop':4} {c.source[:4]} "
                  f"sim={c.sim:.2f} lex={c.lex:.2f} score={c.score:.2f} "
                  f"{' '.join(str(c.doc).split())[:60]!r}")
    print(f"\n{failures} failure(s). Store copy kept at {copy}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
