"""Model calls on the user's Claude subscription, shaped like the anthropic SDK.

The engine (brain/llm/main.py) only ever calls
`client.messages.create(...)`, so SubscriptionClient is a drop-in client: it
runs the same request through `claude -p` (the Claude Code CLI) signed in
with the user's claude.ai account. The project's ANTHROPIC_API_KEY is
stripped from the child's environment, because with a key present the CLI
bills the key and not the subscription, and startup is refused unless
`claude auth status` says claude.ai. The project's credits can't be spent
by accident.

What differs from production, and is recorded in every trace (so a result is
never read as more faithful than it is):
- The verbatim tail goes into the prompt as a labelled transcript. The CLI
  takes one prompt, not prior assistant turns.
- Tools reach the model over MCP (mcp_proxy.py → ToolBridge), so it sees
  them as mcp__tom__<name>. The calls still run through the app's own
  tools.dispatch, on this process's ToolContext.
- The CLI picks the thinking budget, and max_tokens isn't applied.
- The CLI adds a short environment note (working folder, OS) and an account
  reminder to the context. --safe-mode would drop them, but it also drops
  --mcp-config servers (measured 2026-11-12), so Tom would lose his tools.
  The CLI runs in an empty folder with no settings sources and no
  built-in tools.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

# Variables that would make the CLI bill something other than the
# subscription (a key, a token, a proxy endpoint, a cloud provider).
_BILLING_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")
CLI_TIMEOUT_S = 900   # one Tom turn with several tool rounds; a hang must still end the run
# The subscription runs Claude models only. A persona configured for DeepSeek
# (Tom is "deepseek pro" on the dashboard, 2026-11-12) runs on the Claude
# model its registry entry would use, and the trace records both, so a score
# is never read as DeepSeek's. Baseline and candidate always get the same
# stand-in, so the paired comparison stays fair.
STAND_INS = {"deepseek-v4-pro": "claude-sonnet-5-5",
             "deepseek-v4-flash": "claude-haiku-4-5"}


class GatewayError(RuntimeError):
    pass


def claude_cli() -> str:
    """The `claude` executable: $CLAUDE_CLI, PATH, or the default install."""
    found = os.getenv("CLAUDE_CLI") or shutil.which("claude")
    if found:
        return found
    exe = Path.home() / ".local" / "bin" / ("claude.exe" if os.name == "nt" else "claude")
    if exe.exists():
        return str(exe)
    raise GatewayError("claude CLI not found; set CLAUDE_CLI")


def child_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in _BILLING_ENV}


def require_subscription(cli=None) -> dict:
    """`claude auth status` as a dict, or GatewayError unless it is the
    claude.ai subscription."""
    out = subprocess.run([cli or claude_cli(), "auth", "status"], env=child_env(),
                         capture_output=True, text=True, timeout=60)
    try:
        status = json.loads(out.stdout)
    except ValueError:
        raise GatewayError(f"claude auth status unreadable: {out.stdout[:200]!r}")
    if not status.get("loggedIn") or status.get("authMethod") != "claude.ai":
        raise GatewayError("claude CLI is not signed in with the claude.ai "
                           f"subscription (authMethod={status.get('authMethod')!r}); "
                           "refusing to run models on anything else")
    return status


# --- tool calls: CLI → mcp_proxy.py → this bridge → tools.dispatch -------------
class ToolBridge:
    """Loopback HTTP endpoint that runs `handler(name, arguments) -> str`.
    It runs on its own thread because the engine thread is blocked inside
    the CLI call that is waiting for these results."""

    def __init__(self, handler):
        bridge = self

        class _H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                text = bridge.handler(body["name"], body.get("arguments") or {})
                data = json.dumps({"text": text}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.handler = handler
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        threading.Thread(target=self.server.serve_forever, daemon=True,
                         name="tool-bridge").start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/call"

    def close(self):
        self.server.shutdown()


# --- the anthropic-SDK-shaped client ------------------------------------------
class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text

    def model_dump(self, **_):
        return {"type": "text", "text": self.text}


def _flat(content) -> str:
    """A message's content as plain text, tool traffic summarised."""
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        b = b if isinstance(b, dict) else b.model_dump()
        if b.get("type") == "text":
            parts.append(b["text"])
        elif b.get("type") == "tool_use":
            parts.append(f"[tool call {b.get('name')} {json.dumps(b.get('input'))}]")
        elif b.get("type") == "tool_result":
            out = b.get("content")
            out = out if isinstance(out, str) else _flat(out)
            parts.append(f"[tool result: {out}]")
    return "\n".join(parts)


def render_prompt(messages) -> str:
    """The conversation as one CLI prompt: earlier turns as a labelled
    transcript, then the current user message on its own."""
    *earlier, last = messages
    lines = [f"{'User' if m['role'] == 'user' else 'You (assistant)'}: {_flat(m['content'])}"
             for m in earlier]
    head = ("[The conversation so far, verbatim, oldest first]\n"
            + "\n\n".join(lines) + "\n[End of conversation so far]\n\n") if lines else ""
    return head + _flat(last["content"])


def system_text(system) -> str:
    if isinstance(system, str):
        return system
    return "\n\n".join(b["text"] for b in system if b.get("type") == "text")


class SubscriptionClient:
    """Drop-in for anthropic.Anthropic(): `.messages.create` and
    `.with_options`. Every call is appended to `self.calls` (the exact
    system text, prompt, tools offered, reply, usage). The worker writes that
    into the trace."""

    def __init__(self, workdir, tool_handler=None, cli=None):
        self.cli = cli or claude_cli()
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.bridge = ToolBridge(tool_handler) if tool_handler else None
        self.messages = self

    def with_options(self, **_):
        return self   # timeouts: the CLI's start-up alone exceeds the rewrite's 8 s

    def create(self, *, model, system, messages, tools=None, **_):
        n = len(self.calls)
        requested, model = model, STAND_INS.get(model, model)
        if tools and os.getenv("EVAL_TOM_MODEL"):   # the conversation call only; rewrite keeps its own
            model = os.environ["EVAL_TOM_MODEL"]
        sys_path = self.workdir / f"call{n:03d}.system.txt"
        sys_path.write_text(system_text(system), encoding="utf-8")
        prompt = render_prompt(messages)
        cmd = [self.cli, "-p", "--model", model,
               "--system-prompt-file", str(sys_path), "--tools", "",
               "--setting-sources", "", "--strict-mcp-config",
               "--no-session-persistence", "--output-format", "json"]
        if tools and self.bridge:
            tools_path = self.workdir / f"call{n:03d}.tools.json"
            tools_path.write_text(json.dumps(tools), encoding="utf-8")
            mcp = {"mcpServers": {"tom": {
                "command": sys.executable,
                "args": [str(Path(__file__).with_name("mcp_proxy.py")),
                         str(tools_path), self.bridge.url]}}}
            mcp_path = self.workdir / f"call{n:03d}.mcp.json"
            mcp_path.write_text(json.dumps(mcp), encoding="utf-8")
            cmd += ["--mcp-config", str(mcp_path), "--allowedTools", "mcp__tom"]
        t0 = time.monotonic()
        out = subprocess.run(cmd, input=prompt, env=child_env(), cwd=self.workdir,
                             capture_output=True, text=True, encoding="utf-8",
                             timeout=CLI_TIMEOUT_S)
        secs = time.monotonic() - t0
        try:
            res = json.loads(out.stdout)
        except ValueError:
            raise GatewayError(f"claude -p failed ({out.returncode}): "
                               f"{(out.stderr or out.stdout)[:500]}")
        if res.get("is_error"):
            raise GatewayError(f"claude -p error: {res.get('result') or res}")
        text = res.get("result") or ""
        usage = res.get("usage") or {}
        self.calls.append({
            "model": model, "requested_model": requested,
            "system": sys_path.read_text(encoding="utf-8"),
            "prompt": prompt, "tools": [t["name"] for t in tools or []],
            "reply": text, "seconds": round(secs, 1),
            "turns": res.get("num_turns"),
            "permission_denials": res.get("permission_denials"),
            # The CLI's list-price figure; on a subscription nothing is billed per call.
            "list_price_usd": res.get("total_cost_usd"),
            "usage": {k: usage.get(k) for k in ("input_tokens", "output_tokens",
                                                "cache_read_input_tokens",
                                                "cache_creation_input_tokens")}})
        return SimpleNamespace(
            content=[_Text(text)], stop_reason="end_turn",
            usage=SimpleNamespace(**self.calls[-1]["usage"]))

    def close(self):
        if self.bridge:
            self.bridge.close()


def ask_text(model, system, prompt, workdir) -> str:
    """One tool-free call on the subscription (the judge uses it)."""
    client = SubscriptionClient(workdir)
    resp = client.create(model=model, system=system,
                         messages=[{"role": "user", "content": prompt}])
    return resp.content[0].text
