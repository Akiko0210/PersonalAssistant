"""One case, one code version, one fresh clone. Runs as its own process,
started by the runner with the run folder as its working directory:

    python evals/worker.py <run_dir> <case.json> [--no-model] [--build]

The app is imported from the run folder (sandbox.enter); the harness modules
come from this checkout, which the exported code never contains. Tom runs
through the real engine, brain.llm.main.Claude.converse, so the record block,
the Background, the tool loop and the history are production code. Only the
model client is swapped (gateway.SubscriptionClient) and the tools that
leave the machine (sandbox.install_fixtures).

--no-model never calls a model. The rewrite fails over to the raw words, as
it does live when its model is down, and only the prompt parts are built.
That is the deterministic check for "is the evidence in front of Tom?".
--build runs the case's setup ops and exits (synthetic snapshots).

Writes <run_dir>/trace.json: the inputs and outputs of every stage, and
whether each piece of evidence was stored, shown, or fetched by a tool.
"""

import json
import logging
import os
import sqlite3
import sys
import time
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from evals import gateway, sandbox, snapshot  # noqa: E402

OWNER = "tom"
_CLIP = 6000   # chars of one tool result kept in the trace


class _Stop(Exception):
    """Ends a --no-model turn once its prompt is built."""


class _Off:
    """The client for --no-model. The conversation call (the one with tools)
    is recorded like a real one and then stops the turn. Any other call
    (the rewrite) fails, and the engine falls back to the raw words, as it
    does live when that model is down."""

    def __init__(self):
        self.calls = []
        self.messages = self

    def with_options(self, **_):
        return self

    def create(self, *, model, system, messages, tools=None, **_):
        if not tools:
            raise gateway.GatewayError("model calls are off (--no-model)")
        self.calls.append({"model": model, "system": gateway.system_text(system),
                           "prompt": gateway.render_prompt(messages),
                           "tools": [t["name"] for t in tools], "reply": None})
        raise _Stop()

    def close(self):
        pass


def _records(path):
    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            "SELECT r.id, r.status, r.updated, length(r.body), count(e.id) "
            "FROM records r LEFT JOIN entries e ON e.record_id = r.id "
            "WHERE r.owner = ? GROUP BY r.id", (OWNER,)).fetchall()
    finally:
        con.close()
    return {str(i): {"status": s, "updated": u, "body_chars": b, "entries": n}
            for i, s, u, b, n in rows}


def _stored_text(cfg):
    """Everything Tom's stores hold as text, for "was the evidence stored at
    all?": working records and entries, plus every document in Chroma."""
    out = []
    con = sqlite3.connect(cfg.WORKING_STATE_PATH)
    try:
        out += [r[0] for r in con.execute(
            "SELECT title || ' ' || body FROM records WHERE owner = ?", (OWNER,))]
        out += [r[0] for r in con.execute(
            "SELECT e.text FROM entries e JOIN records r ON r.id = e.record_id "
            "WHERE r.owner = ?", (OWNER,))]
    finally:
        con.close()
    chroma = cfg.CHROMA_DIR / "chroma.sqlite3"
    if chroma.exists():
        con = sqlite3.connect(chroma)
        try:
            out += [r[0] for r in con.execute(
                "SELECT string_value FROM embedding_metadata "
                "WHERE key = 'chroma:document'")]
        except sqlite3.Error:
            pass
        finally:
            con.close()
    return "\n".join(t for t in out if t)


def _has(text, ev):
    low = (text or "").lower()
    return all(s.lower() in low for s in ev["contains"])


def run_setup(ops, cfg):
    """Case setup on the clone: extra records or entries (noise, synthetic
    fixtures), with explicit stamps so packing order is reproducible."""
    from stores import working_state
    from brain import history as hist
    ids = {}
    for op in ops or []:
        if op["op"] == "create_record":
            # `filler` pads the body to a size with plainly neutral text, so a
            # fixture can reproduce an overflow without inventing content.
            body = op.get("body", "")
            i = 0
            while len(body) < op.get("filler", 0):
                i += 1
                body += f" Background detail {i}: unrelated to any open trade."
            rid = working_state.create(op.get("owner", OWNER), op["title"],
                                       kind=op.get("kind", ""), body=body.strip())
            ids[op.get("as", op["title"])] = rid
            if op.get("created") or op.get("updated"):
                con = sqlite3.connect(cfg.WORKING_STATE_PATH)
                with con:
                    con.execute("UPDATE records SET created = ?, updated = ? WHERE id = ?",
                                (op.get("created") or op["updated"],
                                 op.get("updated") or op["created"], rid))
                con.close()
        elif op["op"] == "add_entry":
            rid = ids.get(op["record"], op["record"])
            working_state.add_entry(op.get("owner", OWNER), int(rid), op["text"])
            if op.get("ts"):
                con = sqlite3.connect(cfg.WORKING_STATE_PATH)
                with con:
                    eid = con.execute("SELECT max(id) FROM entries").fetchone()[0]
                    con.execute("UPDATE entries SET ts = ? WHERE id = ?", (op["ts"], eid))
                    con.execute("UPDATE records SET updated = max(updated, ?) WHERE id = ?",
                                (op["ts"], int(rid)))
                con.close()
        elif op["op"] == "close_record":
            working_state.close(op.get("owner", OWNER), int(ids.get(op["record"], op["record"])),
                                op["text"])
        elif op["op"] == "clear_thread":
            hist.save(cfg.history_path(OWNER), [])
        else:
            raise ValueError(f"unknown setup op {op['op']}")
    return ids


def main(run_dir, case_path, *, model_on=True, build=False):
    run_dir = Path(run_dir).resolve()
    case = json.loads(Path(case_path).read_text(encoding="utf-8"))
    prod = json.loads(os.environ.get("EVAL_PRODUCTION_ROOTS", "[]"))
    trace = {"case": case["id"], "model_on": model_on, "started": time.time(),
             "deviations": [
                 "models via claude -p on the subscription (tail as a transcript, "
                 "tools over MCP, CLI-chosen thinking)",
                 "query rewrite and fact extraction on their configured fallback "
                 "models (production default is DeepSeek, unreachable here)",
                 "fact extraction after each turn is off unless the case sets learn",
                 "clock is real time, not frozen"]}
    out_path = run_dir / "trace.json"
    try:
        trace["paths"] = sandbox.enter(run_dir, prod)
        logging.basicConfig(filename=run_dir / "logs" / "eval.log", level=logging.INFO,
                            format="%(asctime)s %(name)s %(levelname)s %(message)s",
                            encoding="utf-8")
        import config as cfg
        if build:
            trace["setup_ids"] = run_setup(case.get("setup"), cfg)
            return trace
        import tools
        from brain.llm import main as engine
        from stores.knowledge import KnowledgeStore

        cfg.QUERY_MODEL = cfg.QUERY_FALLBACK_MODEL
        cfg.FACTS_MODEL = cfg.FACTS_FALLBACK_MODEL
        sandboxed = []
        sandbox.install_fixtures(tools._REGISTRY, case.get("fixtures", {}), sandboxed)
        trace["setup_ids"] = run_setup(case.get("setup"), cfg)

        tool_calls, holder = [], {}

        def handle(name, args):
            t0 = time.monotonic()
            out = tools.dispatch(holder["llm"]._ctx, name, args)
            tool_calls.append({"turn": holder.get("turn"), "tool": name, "args": args,
                               "result": str(out)[:_CLIP],
                               "seconds": round(time.monotonic() - t0, 2)})
            return str(out)

        if model_on:
            gateway.require_subscription()
            client = gateway.SubscriptionClient(run_dir / "cli", tool_handler=handle)
        else:
            client = _Off()
        engine.anthropic_api.make_client = lambda: client
        engine.deepseek.make_client = lambda: client

        pulls, blocks = [], []
        real_build = engine.context.build_context

        def build_context(q, **kw):
            pull = real_build(q, **kw)
            pulls.append({"turn": holder.get("turn"), "query": getattr(q, "text", str(q)),
                          "entities": list(getattr(q, "entities", []) or []),
                          "block": pull.block,
                          "kept": [(c.source, c.id) for c in pull.candidates if c.kept]})
            return pull
        engine.context.build_context = build_context

        llm = engine.Claude(store=None, kb=KnowledgeStore())
        llm._jobs = None          # memory jobs run inline, in order, before the next turn
        holder["llm"] = llm
        real_record = llm._record_block

        def record_block(owner):
            text = real_record(owner)
            blocks.append({"turn": holder.get("turn"), "text": text})
            return text
        llm._record_block = record_block
        if not case.get("learn"):
            llm._learn_exchange = lambda owner, msgs: llm.memory.index_exchanges(
                msgs, owner, skip_existing=False)
        if llm.active != OWNER:
            llm.switch_to(OWNER)

        trace["state_before"] = {"records": _records(cfg.WORKING_STATE_PATH),
                                 "counts": snapshot._counts(run_dir)}
        trace["turns"] = []
        for i, text in enumerate(case["turns"]):
            holder["turn"] = i
            t0 = time.monotonic()
            try:
                reply = llm.converse(text)
                llm.settle_reply(reply, reply)
            except _Stop:
                reply = None      # --no-model: the prompt was built and captured
            trace["turns"].append({"user": text, "reply": reply,
                                   "seconds": round(time.monotonic() - t0, 1)})
        trace["pulls"], trace["record_blocks"], trace["tool_calls"] = pulls, blocks, tool_calls
        trace["sandboxed_calls"] = sandboxed
        trace["model_calls"] = client.calls
        trace["state_after"] = {"records": _records(cfg.WORKING_STATE_PATH),
                                "counts": snapshot._counts(run_dir)}

        # Where each piece of evidence got to: stored → in the prompt → fetched.
        stored = _stored_text(cfg)
        final_turn = len(case["turns"]) - 1
        record = next((b["text"] for b in reversed(blocks) if b["turn"] == final_turn), "")
        background = next((p["block"] for p in reversed(pulls) if p["turn"] == final_turn), "")
        convo = [c for c in client.calls if c["tools"]]
        prompt = (convo[-1]["system"] + "\n" + convo[-1]["prompt"]) if convo else ""
        fetched = "\n".join(c["result"] for c in tool_calls if c["turn"] == final_turn)
        trace["evidence"] = [{
            "id": ev["id"], "stored": _has(stored, ev),
            "in_record_block": _has(record, ev), "in_background": _has(background, ev),
            "in_prompt": _has(prompt, ev),
            "fetched_by_tool": _has(fetched, ev)} for ev in case.get("evidence", [])]
        trace["mutations"] = [c for c in sandboxed if c["mutating"]]
        client.close()
    except sandbox.SandboxError as e:
        trace["refused"] = str(e)
    except Exception:  # noqa: BLE001 - the trace must record how a run died
        trace["error"] = traceback.format_exc()
    finally:
        trace["seconds"] = round(time.time() - trace["started"], 1)
        out_path.write_text(json.dumps(trace, indent=1, default=str), encoding="utf-8")
    return trace


if __name__ == "__main__":
    args = sys.argv[1:]
    t = main(args[0], args[1], model_on="--no-model" not in args, build="--build" in args)
    sys.exit(3 if t.get("refused") else 1 if t.get("error") else 0)
