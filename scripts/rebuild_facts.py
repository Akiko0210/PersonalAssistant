"""Rebuild a persona's fact store from its exchange index.

This is the property that makes write-time extraction safe to ship: facts
are a derived index over the exchanges, so an extractor bug — or a better
prompt — is fixed by running this, never by editing facts. Drops
facts_<owner>, clears the extractor's read-marks on that persona's
exchanges, and re-reads every exchange oldest first with FACTS_MODEL: one
model call per exchange.

    python -m scripts.rebuild_facts --owner alice          # one persona
    python -m scripts.rebuild_facts --all                  # every persona
    python -m scripts.rebuild_facts --owner alice --dry-run   # count, no changes

Run with the agent OFF — it writes the same Chroma store, and the
single-instance lock refuses otherwise.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
from brain import agents  # noqa: E402
from lib.single_instance.main import AlreadyRunning, SingleInstance  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(cfg.BASE_DIR / ".env")  # ANTHROPIC_API_KEY for the extractor
except ImportError:
    pass  # env vars still work


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--owner", choices=list(agents.AGENTS))
    who.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report counts, change nothing")
    args = ap.parse_args()
    owners = list(agents.AGENTS) if args.all else [args.owner]

    try:
        lock = SingleInstance(cfg.LOCK_PATH).acquire()
    except AlreadyRunning:
        print("A voice agent is running — close it first: the rebuild writes "
              "the same Chroma store.")
        return 1
    try:
        from brain.facts import FactStore, catch_up, extractor
        from brain.llm import anthropic as anthropic_api
        from brain.memory import ConversationMemory
        from stores import chroma_store

        memory, facts = ConversationMemory(), FactStore()
        ask_facts = extractor(anthropic_api.make_client())

        for owner in owners:
            col = memory._col_for(cfg.agent_memory_collection(owner))
            marked = col.get(where={"extracted": True}, include=[]).get("ids") or []
            total = len(memory.unextracted(owner)) + len(marked)
            print(f"{owner}: {total} exchange(s), {len(marked)} already read", flush=True)
            if args.dry_run:
                continue
            name = cfg.agent_facts_collection(owner)
            facts._cols.pop(name, None)
            print(f"  dropped {name}: {chroma_store.drop(name)}", flush=True)
            for i in range(0, len(marked), 500):
                ids = marked[i:i + 500]
                col.update(ids=ids, metadatas=[{"extracted": False}] * len(ids))
            t0 = time.monotonic()
            read, failed = catch_up(
                ask_facts, memory, facts, owner,
                progress=lambda i, n: print(f"  {i}/{n}", flush=True))
            print(f"  {read} read, {failed} failed ({time.monotonic() - t0:.0f}s, "
                  f"{cfg.FACTS_MODEL})", flush=True)
    finally:
        lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
