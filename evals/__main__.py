"""python -m evals <command> — see docs/TEST_ENV.md."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from evals import runner, snapshot, testenv_root


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m evals")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="capture Tom's live state (read-only master)")
    s.add_argument("--name", default=f"real-{datetime.now():%Y-%m-%d}")
    s.add_argument("--source", help="checkout holding data/ (default: the main checkout)")
    b = sub.add_parser("synth", help="build a synthetic snapshot from a fixture case")
    b.add_argument("fixture", help="case id whose setup ops build the state")
    sub.add_parser("verify", help="hash-check every master snapshot")
    r = sub.add_parser("run", help="one case on one ref")
    r.add_argument("case")
    r.add_argument("--ref", default=runner.WORKTREE)
    r.add_argument("--no-model", action="store_true")
    r.add_argument("--no-judge", action="store_true")
    c = sub.add_parser("compare", help="cases on baseline vs candidate")
    c.add_argument("cases", nargs="+")
    c.add_argument("--baseline", required=True)
    c.add_argument("--candidate", default=runner.WORKTREE)
    c.add_argument("--repeats", type=int, default=3)
    c.add_argument("--no-model", action="store_true")
    c.add_argument("--no-judge", action="store_true")
    for cmd in (r, c):
        cmd.add_argument("--tom-model", help="answer with this Claude model instead of "
                         "Tom's configured one (or its stand-in)")
    a = p.parse_args(argv)
    if getattr(a, "tom_model", None):
        os.environ["EVAL_TOM_MODEL"] = a.tom_model   # inherited by every worker

    root = testenv_root()
    if a.cmd == "snapshot":
        source = Path(a.source) if a.source else Path(runner.production_roots()[0]).parent
        m = snapshot.capture(source, root / "snapshots" / a.name)
        print(json.dumps({k: m[k] for k in ("name", "captured", "consistent_copy",
                                             "agent_lock_present", "counts")}, indent=1))
    elif a.cmd == "synth":
        case = json.loads(runner.find_case(a.fixture).read_text(encoding="utf-8"))
        build = root / "runs" / f"build_{case['snapshot']}_{datetime.now():%Y%m%d-%H%M%S}"
        runner.export_code("HEAD", build)
        (build / "logs").mkdir()
        proc = runner.subprocess.run(
            [sys.executable, str(runner.HARNESS_DIR / "worker.py"), str(build),
             str(runner.find_case(a.fixture)), "--build"],
            cwd=build, env=runner._worker_env(), capture_output=True, text=True)
        if proc.returncode:
            sys.exit(f"build failed: {(build / 'trace.json').read_text()[-2000:]}")
        m = snapshot.capture(build, root / "snapshots" / case["snapshot"],
                             note=f"synthetic, built from {a.fixture}")
        print(json.dumps(m["counts"], indent=1))
    elif a.cmd == "verify":
        for snap in sorted((root / "snapshots").iterdir()):
            bad = snapshot.verify(snap)
            print(f"{snap.name}: {'unchanged' if not bad else bad}")
    elif a.cmd == "run":
        print(json.dumps(runner.run_one(a.case, a.ref, "single", model=not a.no_model,
                                        judge=not a.no_judge), indent=1))
    elif a.cmd == "compare":
        out = runner.compare(a.cases, a.baseline, a.candidate, a.repeats,
                             model=not a.no_model, judge=not a.no_judge)
        print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
