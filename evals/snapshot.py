"""Snapshots of Tom's state: capture once, verify, clone per run.

A snapshot is a read-only folder with data/ and knowledge/ shaped like the
repo's (so a clone drops straight into a run folder) and a manifest.json:
every file's size and SHA-256, record counts, the source commit, the capture
time, and what was left out and why.

Capture is safe while the agent runs. SQLite files (working state, trade log,
Chroma's own database) go through the online backup API, which copies a
consistent database even mid-write. Chroma's vector segment files are
copied, then their modification times are checked again; a change during
the copy retries, and if it keeps changing the manifest says so. Missing
vector segments (the legacy archive's lost index) are listed and stay
missing. The point is to reproduce what Tom has, not a repaired version.
"""

import fnmatch
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
from datetime import datetime
from pathlib import Path

# What Tom's turns can read, relative to the production root. Everything
# else is left out: credentials, other personas' notes and threads, the
# media behind the knowledge base (its chunks are in Chroma), logs.
INCLUDE = ("data/memory/working_state.db", "data/memory/chroma/**",
           "data/conversation_history/tom.json", "data/settings/*.json",
           "data/trading/trade_log.db", "data/trading/schwab/accounts.json",
           "data/trading/statements/*", "data/TradingJournals/**",
           "knowledge/manifest.json")
EXCLUDE = ("*token*", "*secret*", "*credential*", "data/settings/agent_state.json")
SQLITE = (".db", ".sqlite3")


def _wanted(rel):
    return (any(fnmatch.fnmatch(rel, p) for p in INCLUDE)
            and not any(fnmatch.fnmatch(rel.lower(), p) for p in EXCLUDE))


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in SQLITE:
        con = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
        out = sqlite3.connect(dst)
        try:
            con.backup(out)
        finally:
            out.close()
            con.close()
    else:
        shutil.copyfile(src, dst)


def _counts(root):
    """Record counts for the manifest. Best effort: a store that can't be
    read is reported as such, never guessed."""
    out = {}

    def q(db, sql):
        con = sqlite3.connect(f"file:{(root / db).as_posix()}?mode=ro", uri=True)
        try:
            return con.execute(sql).fetchall()
        finally:
            con.close()
    try:
        out["records"] = {f"{o}/{s}": n for o, s, n in q(
            "data/memory/working_state.db",
            "SELECT owner, status, count(*) FROM records GROUP BY owner, status")}
        out["entries"] = dict(q("data/memory/working_state.db",
                                "SELECT r.owner, count(*) FROM entries e JOIN records r "
                                "ON r.id = e.record_id GROUP BY r.owner"))
    except Exception as e:  # noqa: BLE001
        out["records"] = f"unreadable: {e}"
    try:
        out["trade_log"] = dict(q("data/trading/trade_log.db",
                                  "SELECT name, 0 FROM sqlite_master WHERE type='table'"))
        for t in list(out["trade_log"]):
            out["trade_log"][t] = q("data/trading/trade_log.db",
                                    f"SELECT count(*) FROM \"{t}\"")[0][0]
    except Exception as e:  # noqa: BLE001
        out["trade_log"] = f"unreadable: {e}"
    try:
        db = "data/memory/chroma/chroma.sqlite3"
        out["chroma"] = dict(q(db, "SELECT c.name, count(e.id) FROM collections c "
                                   "JOIN segments s ON s.collection = c.id "
                                   "JOIN embeddings e ON e.segment_id = s.id "
                                   "GROUP BY c.name"))
        vectors = q(db, "SELECT s.id, c.name FROM segments s JOIN collections c "
                        "ON c.id = s.collection WHERE s.scope = 'VECTOR'")
        chroma = root / "data/memory/chroma"
        out["vector_segments_missing"] = sorted(
            name for sid, name in vectors if not (chroma / sid).is_dir())
    except Exception as e:  # noqa: BLE001
        out["chroma"] = f"unreadable: {e}"
    try:
        thread = json.loads((root / "data/conversation_history/tom.json")
                            .read_text(encoding="utf-8"))
        out["tom_thread_messages"] = len(thread)
    except Exception as e:  # noqa: BLE001
        out["tom_thread_messages"] = f"unreadable: {e}"
    return out


def _files(root):
    return {p.relative_to(root).as_posix(): p
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}


def capture(source_root, dest, *, note=""):
    """Copy Tom's state from `source_root` (a checkout with data/) into the
    new snapshot folder `dest`, write the manifest, make it read-only."""
    source_root, dest = Path(source_root), Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    picked = [rel for rel in (p.relative_to(source_root).as_posix()
                              for p in sorted(source_root.rglob("*")) if p.is_file()
                              and not p.relative_to(source_root).parts[0].startswith("."))
              if _wanted(rel)]
    consistent = False
    for _ in range(3):
        before = {rel: os.stat(source_root / rel).st_mtime_ns for rel in picked}
        for rel in picked:
            _copy(source_root / rel, dest / rel)
        after = {rel: os.stat(source_root / rel).st_mtime_ns for rel in picked}
        changed = [r for r in picked if before[r] != after[r]
                   and Path(r).suffix not in SQLITE]
        if not changed:
            consistent = True
            break
    try:
        commit = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    manifest = {
        "name": dest.name, "note": note,
        "captured": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_commit": commit,
        "agent_lock_present": (source_root / "data/agent.lock").exists(),
        "consistent_copy": consistent,
        "include": INCLUDE, "exclude": EXCLUDE,
        "counts": _counts(dest),
        "files": {rel: {"bytes": p.stat().st_size, "sha256": _sha(p)}
                  for rel, p in _files(dest).items()},
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    for p in dest.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD)
    return manifest


def verify(snapshot):
    """Mismatches between `snapshot` and its manifest: [] means untouched."""
    snapshot = Path(snapshot)
    want = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))["files"]
    have = _files(snapshot)
    bad = [f"missing {r}" for r in want if r not in have]
    bad += [f"extra {r}" for r in have if r not in want]
    bad += [f"changed {r}" for r, meta in want.items()
            if r in have and _sha(have[r]) != meta["sha256"]]
    return bad


def clone(snapshot, run_root):
    """A writable copy of the snapshot's state inside `run_root`."""
    snapshot, run_root = Path(snapshot), Path(run_root)
    for rel, src in _files(snapshot).items():
        dst = run_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)   # copyfile, not copy2: the read-only bit must not follow
    (run_root / "logs").mkdir(exist_ok=True)
    shutil.copyfile(snapshot / "manifest.json", run_root / "snapshot_manifest.json")
