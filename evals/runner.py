"""Runs cases: export a code version, clone a snapshot beside it, start the
worker, judge the answer, and pair baseline against candidate.

Every run gets a fresh folder under <testenv>/runs/. Nothing is reused
between runs, and the master snapshot is hash-verified after each suite.
The export never contains evals/: the harness under test is always this
checkout's, so baseline and candidate are measured with the same ruler.
"""

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

from evals import HARNESS_DIR, JUDGE_MODEL, REPO_DIR, gateway, sandbox, snapshot, testenv_root

WORKTREE = "WORKTREE"   # ref meaning "this checkout as it is now, uncommitted edits included"


def _git(*args, cwd=REPO_DIR):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout.strip()


def production_roots():
    """The live checkout's state folders. A run may never resolve into them."""
    main = Path(_git("rev-parse", "--path-format=absolute", "--git-common-dir")).parent
    return [str(main / d) for d in ("data", "logs", "knowledge")]


def export_code(ref, dest):
    """The app at `ref` into `dest`, without evals/. Returns the commit
    label recorded in the trace."""
    dest.mkdir(parents=True)
    if ref == WORKTREE:
        files = _git("ls-files", "-co", "--exclude-standard").splitlines()
        for rel in files:
            src = REPO_DIR / rel
            if src.is_file() and not rel.startswith("evals/"):
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest / rel)
        dirty = bool(_git("status", "--porcelain"))
        return _git("rev-parse", "HEAD") + ("+uncommitted" if dirty else "")
    tar = subprocess.run(["git", "archive", "--format=tar", ref], cwd=REPO_DIR,
                         capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(dest, filter="data")
    shutil.rmtree(dest / "evals", ignore_errors=True)
    return _git("rev-parse", ref)


def find_case(case_id):
    for folder in (HARNESS_DIR / "cases", testenv_root() / "cases"):
        p = folder / f"{case_id}.json"
        if p.exists():
            return p
    raise FileNotFoundError(f"no case {case_id} in evals/cases or {testenv_root() / 'cases'}")


def _worker_env():
    env = {k: v for k, v in os.environ.items() if not sandbox._SECRET_ENV.search(k)}
    env["EVAL_PRODUCTION_ROOTS"] = json.dumps(production_roots())
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_one(case_id, ref, label, rep=0, *, model=True, judge=True):
    """One case on one version from a fresh clone; returns the run summary."""
    case_path = find_case(case_id)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    # The random tail keeps suites running side by side from sharing a folder.
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}-{os.urandom(2).hex()}"
    run_dir = testenv_root() / "runs" / f"{stamp}_{case_id}_{label}_{rep}"
    commit = export_code(ref, run_dir)
    snapshot.clone(testenv_root() / "snapshots" / case["snapshot"], run_dir)
    cmd = [sys.executable, str(HARNESS_DIR / "worker.py"), str(run_dir), str(case_path)]
    if not model:
        cmd.append("--no-model")
    proc = subprocess.run(cmd, cwd=run_dir, env=_worker_env(), capture_output=True,
                          text=True, encoding="utf-8")
    trace_path = run_dir / "trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
    trace.update({"ref": ref, "commit": commit, "label": label, "rep": rep,
                  "worker_exit": proc.returncode, "worker_stderr": proc.stderr[-3000:]})
    if judge and model and trace.get("turns") and not trace.get("error"):
        trace["judge"] = grade(case, trace, run_dir)
    trace_path.write_text(json.dumps(trace, indent=1, default=str), encoding="utf-8")
    return summarize(case, trace, run_dir)


# --- judging ------------------------------------------------------------------
_JUDGE_SYSTEM = (
    "You grade one spoken answer from Tom, a voice trading assistant. The user "
    "has tunnel vision and can't check numbers against a screen, so a confident "
    "wrong status does real harm, and a correct answer buried in hedging is "
    "worth less. Grade the substance against the ground truth, not the wording. "
    "Checking a tool or offering to verify is fine when the answer still states "
    "the correct status. Reply with JSON only: "
    '{"verdict": "PASS" | "PARTIAL" | "FAIL", "reason": "<one or two sentences>"}')


def grade(case, trace, run_dir):
    turns = trace["turns"]
    convo = "\n".join(f"User: {t['user']}\nTom: {t['reply']}" for t in turns[:-1])
    prompt = (f"Case: {case['title']}\n\n"
              + (f"Earlier turns:\n{convo}\n\n" if convo else "")
              + f"User's question: {turns[-1]['user']}\n\n"
              f"Tom's answer:\n{turns[-1]['reply']}\n\n"
              f"Ground truth: {case['expected']}\n\n"
              "PASS requires: " + "; ".join(case.get("pass_if", [])) + "\n"
              "FAIL if any: " + "; ".join(case.get("fail_if", [])) + "\n"
              "PARTIAL: correct status present but materially incomplete, or hedged "
              "so the user can't tell what the status is.")
    try:
        text = gateway.ask_text(JUDGE_MODEL, _JUDGE_SYSTEM, prompt, run_dir / "judge")
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {"verdict": "ERROR", "reason": text[:300]}
    except Exception as e:  # noqa: BLE001 - a failed judge is reported, not fatal
        return {"verdict": "ERROR", "reason": str(e)[:300]}


def summarize(case, trace, run_dir):
    calls = trace.get("model_calls", [])
    convo = [c for c in calls if c.get("tools")]
    return {
        "case": case["id"], "label": trace.get("label"), "rep": trace.get("rep"),
        "model": convo[-1]["model"] if convo else None,
        "requested_model": convo[-1].get("requested_model") if convo else None,
        "commit": trace.get("commit"), "run_dir": str(run_dir),
        "refused": trace.get("refused"), "error": (trace.get("error") or "")[-800:] or None,
        "reply": (trace.get("turns") or [{}])[-1].get("reply"),
        "verdict": (trace.get("judge") or {}).get("verdict"),
        "reason": (trace.get("judge") or {}).get("reason"),
        "evidence": trace.get("evidence", []),
        "tools": [c["tool"] for c in trace.get("tool_calls", [])],
        "mutations": trace.get("mutations", []),
        "record_block_chars": len((trace.get("record_blocks") or [{}])[-1].get("text", "")),
        "background_chars": len((trace.get("pulls") or [{}])[-1].get("block", "")),
        "seconds": trace.get("seconds"),
        "list_price_usd": round(sum(c.get("list_price_usd") or 0 for c in calls), 4),
    }


# --- suites -------------------------------------------------------------------
def compare(case_ids, baseline, candidate, repeats=3, *, model=True, judge=True, out=None):
    """Each case `repeats` times on both refs, alternating which goes first.
    Writes report.json and report.md and returns the report's folder."""
    out = Path(out or testenv_root() / "reports" / f"{datetime.now():%Y%m%d-%H%M%S}-{os.urandom(2).hex()}")
    out.mkdir(parents=True, exist_ok=True)
    results, snaps = [], set()
    for case_id in case_ids:
        snaps.add(json.loads(find_case(case_id).read_text(encoding="utf-8"))["snapshot"])
        for rep in range(repeats):
            pair = [("baseline", baseline), ("candidate", candidate)]
            for label, ref in (pair if rep % 2 == 0 else pair[::-1]):
                t0 = time.monotonic()
                r = run_one(case_id, ref, label, rep, model=model, judge=judge)
                results.append(r)
                print(f"{case_id} {label} #{rep}: {r['verdict'] or r['refused'] or ('error' if r['error'] else 'built')}"
                      f" ({time.monotonic() - t0:.0f}s)", flush=True)
                (out / "report.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    integrity = {s: snapshot.verify(testenv_root() / "snapshots" / s) for s in snaps}
    report = {"baseline": baseline, "candidate": candidate, "repeats": repeats,
              "model": model, "snapshots_unchanged": {s: not bad for s, bad in integrity.items()},
              "results": results}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / "report.md").write_text(render_md(report), encoding="utf-8")
    return out


def render_md(report):
    lines = [f"# Paired run: {report['baseline']} vs {report['candidate']}", "",
             f"Repeats per case: {report['repeats']}; model calls: {report['model']}; "
             f"master snapshots unchanged: {report['snapshots_unchanged']}", ""]
    by_case = {}
    for r in report["results"]:
        by_case.setdefault(r["case"], []).append(r)
    lines += ["| Case | Pass (b → c) | Evidence in prompt (b → c) | "
              "Tool calls, mean (b → c) | Seconds, mean (b → c) |",
              "| --- | --- | --- | --- | --- |"]
    for case, rs in by_case.items():
        got = {lab: [r for r in rs if r["label"] == lab] for lab in ("baseline", "candidate")}

        def both(fn):
            return " → ".join(fn(got[lab]) for lab in ("baseline", "candidate"))

        def mean(xs):
            return f"{sum(xs) / len(xs):.1f}" if xs else "-"

        def passed(g):
            return f"{sum(r['verdict'] == 'PASS' for r in g)}/{len(g)}"

        def shown(g):
            return f"{sum(all(e['in_prompt'] for e in r['evidence']) for r in g)}/{len(g)}"
        lines.append(f"| {case} | {both(passed)} | {both(shown)} | "
                     f"{both(lambda g: mean([len(r['tools']) for r in g]))} | "
                     f"{both(lambda g: mean([r['seconds'] or 0 for r in g]))} |")
    for case, rs in by_case.items():
        lines += ["", f"## {case}", ""]
        for r in rs:
            lines += [f"### {r['label']} #{r['rep']}: {r['verdict']}",
                      f"- judge: {r['reason']}",
                      f"- evidence: {r['evidence']}",
                      f"- tools: {r['tools']}; mutations: {len(r['mutations'])}; "
                      f"{r['seconds']}s; list price ${r['list_price_usd']}",
                      f"- run: `{r['run_dir']}`", "", "> " + (r["reply"] or r["error"] or
                                                          r["refused"] or "").replace("\n", "\n> "),
                      ""]
    return "\n".join(lines)
