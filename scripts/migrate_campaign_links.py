"""Turn Tom's old campaign→contract links into positions, once.

Before positions existed, a campaign record listed its option contracts in
working_state.db's `contracts` table, and query_trade_log record=<id> picked
every trade touching one. For each record with links, this creates a
position per account (named after the record, tied to it), assigns it those
same trades, then runs the one-match rule so rolls that were never linked
join too (the 2026-10-15 roll chain). The old table is left in place.

    python -m scripts.migrate_campaign_links --dry-run   # report, no changes
    python -m scripts.migrate_campaign_links

Run with the agent OFF; the single-instance lock refuses otherwise. A record
that already has a position is skipped, so a second run is a no-op.
"""

import argparse
import shutil
import sqlite3
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
from lib.single_instance.main import AlreadyRunning, SingleInstance  # noqa: E402
from trading import config as tcfg  # noqa: E402
from trading import trade_log  # noqa: E402
from trading import trade_positions as tpos  # noqa: E402


def _links():
    """{record id: (title, {account: {(symbol, exp, strike, type)}})}."""
    db = sqlite3.connect(f"file:{cfg.WORKING_STATE_PATH}?mode=ro", uri=True)
    try:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name = "
                          "'contracts'").fetchone():
            return {}
        out = {}
        for rid, title, account, *contract in db.execute(
                "SELECT c.record_id, r.title, c.account, c.symbol, "
                "c.expiration, c.strike, c.option_type FROM contracts c "
                "JOIN records r ON r.id = c.record_id ORDER BY c.record_id"):
            out.setdefault(rid, (title, defaultdict(set)))[1][account].add(
                tuple(contract))
        return out
    finally:
        db.close()


def migrate():
    touched = set()
    for rid, (title, by_account) in _links().items():
        if trade_log.positions(record_id=rid):
            print(f"#{rid} {title!r}: already has a position, skipped")
            continue
        for account, contracts in by_account.items():
            refs = [t["ref"] for t in trade_log.query(account=account)
                    if t["position_id"] is None and contracts & {
                        (t["symbol"], e, k, cp) for e, k, cp, _ in t["legs"]}]
            if not refs:
                continue
            pid = trade_log.create_position(account, title, rid)
            trade_log.assign(account, refs, pid)
            touched.add(account)
            print(f"#{rid} {title!r} -> P{pid} on {account}: {len(refs)} "
                  "linked trades")
    for account in sorted(touched):
        trade_log.auto_assign(account)
    for p in trade_log.positions():
        ts = trade_log.query(positions=[p["id"]])
        legs = tpos.net_legs(ts)
        print(f"P{p['id']} {p['name']!r}: {len(ts)} trades, open legs "
              + (", ".join(f"{q:+d} {e} {k:g}{cp}"
                           for (_, e, k, cp), q in sorted(legs.items()))
                 or "none"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="run on a copy of the trade log and report")
    args = ap.parse_args()
    if args.dry_run:        # a copy: safe beside a running agent
        with tempfile.TemporaryDirectory() as tmp:
            real, tcfg.TRADE_LOG_PATH = (tcfg.TRADE_LOG_PATH,
                                         Path(tmp) / "trade_log.db")
            shutil.copy(real, tcfg.TRADE_LOG_PATH)
            try:
                migrate()
            finally:
                tcfg.TRADE_LOG_PATH = real
        print("(dry run: nothing written)")
        return 0
    try:
        lock = SingleInstance(cfg.LOCK_PATH).acquire()
    except AlreadyRunning:
        print("A voice agent is running — close it first: this writes the "
              "trade log it reads.")
        return 1
    try:
        migrate()
        return 0
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
