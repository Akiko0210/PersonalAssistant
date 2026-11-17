"""Isolation for one eval run. Everything here happens before Tom starts, and
any doubt refuses the run.

A run folder holds an exported code version, with the cloned state beside it
in data/, knowledge/ and logs/. config.BASE_DIR is the folder the code runs
from, so every state path the app derives lands in the clone. The production
code needs no path seam. The checks below make sure that held:

- the layout (data/ etc.) resolves inside the run folder, junctions and
  symlinks followed, before any app module is imported (no write can happen
  first);
- after import, every *_PATH / *_DIR the app configured resolves inside the
  run folder and none into production. Settings overrides and .env are
  applied by then, so a dashboard override can't slip a real path in;
- secrets are removed from the environment, and models are off the
  network: the CLI child is the only process that talks to Anthropic, and
  this process may only open loopback sockets;
- every tool that could leave the machine (brokers, mail, Discord) is
  replaced by a fixture. A missing fixture is an error the model sees, never
  a live fallback. A mutating call is recorded and rejected.
"""

import os
import re
import socket
import sys
from pathlib import Path

# Tools that only touch the cloned stores or this turn's ToolContext. Every
# other registered tool is replaced, so a tool added later is sandboxed by
# default (deny-by-default, not a list of dangerous ones).
LOCAL_TOOLS = frozenset({
    "recall", "get_current_time", "get_current_model", "set_conversation_model",
    "switch_agent", "ask_agent",                      # deferred slots; the harness never runs them
    "update_record", "add_entry", "close_record", "read_record",
    "set_focus", "clear_focus", "get_focus", "focus_project", "describe_project",
    "query_trade_log", "trade_log_positions", "assign_trades", "log_trade",
    "import_statement",                               # reads the cloned statements folder
})
# Calls that would change something outside the machine if they were live.
MUTATING = frozenset({"submit_order", "cancel_order", "send_email",
                      "create_email_draft", "sync_trades"})
# Configured paths that may lie outside the run folder: read-only, and every
# tool that reads them is fixtured.
OUTSIDE_OK = frozenset({"PIPER_VOICE_DIR", "DISCORD_DIR", "DISCORD_LOG_PATH",
                        "DISCORD_TRADES_PATH"})
_SECRET_ENV = re.compile(r"API_KEY|SECRET|TOKEN|PASSWORD|SCHWAB|TASTY|DEEPSEEK|"
                         r"ANTHROPIC|GMAIL|DISCORD|OPENAI", re.I)
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class SandboxError(RuntimeError):
    pass


def _inside(path, root) -> bool:
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(root))
        return True
    except ValueError:
        return False


def check_layout(run_root, production_roots=()):
    """Before import: the state folders resolve inside the run folder, and
    the run folder itself is not inside production."""
    run_root = Path(run_root)
    for prod in production_roots:
        if _inside(run_root, prod) or _inside(prod, run_root):
            raise SandboxError(f"run folder {run_root} overlaps production {prod}")
    for sub in ("data", "logs", "knowledge"):
        p = run_root / sub
        if os.path.lexists(p) and not _inside(p, run_root):
            raise SandboxError(f"{p} resolves to {os.path.realpath(p)}, "
                               "outside the run folder (junction or symlink)")


def check_paths(run_root, modules, production_roots=()):
    """After import: every configured Path attribute stays in the run
    folder; returns {name: resolved} for the trace."""
    seen = {}
    for mod in modules:
        for name, value in vars(mod).items():
            if not isinstance(value, Path) or not re.search(r"(_PATH|_DIR)$", name):
                continue
            real = os.path.realpath(value)
            seen[f"{mod.__name__}.{name}"] = real
            if any(_inside(real, p) for p in production_roots):
                raise SandboxError(f"{mod.__name__}.{name} resolves into "
                                   f"production: {real}")
            if name not in OUTSIDE_OK and not _inside(real, run_root):
                raise SandboxError(f"{mod.__name__}.{name} resolves outside the "
                                   f"run folder: {real}")
    return seen


def scrub_env(run_root):
    """No credentials reach the app, and nothing the app reads from the
    environment points back at production."""
    for key in [k for k in os.environ if _SECRET_ENV.search(k)]:
        del os.environ[key]
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",   # embeddings from the local cache only
        "ANONYMIZED_TELEMETRY": "False",
        "TRADE_STATEMENTS_DIR": str(Path(run_root) / "data" / "trading" / "statements"),
    })


def guard_network():
    """This process may only open loopback sockets (the tool bridge); any
    other connect or DNS lookup raises, so a missed fixture fails loudly."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_gai = socket.getaddrinfo

    def _check(address):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and (host in _LOOPBACK or host.startswith("127.")):
            return
        raise PermissionError(f"sandbox: network blocked ({host})")

    def connect(self, address):
        _check(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        _check(address)
        return real_connect_ex(self, address)

    def getaddrinfo(host, *a, **kw):
        _check((host,))
        return real_gai(host, *a, **kw)

    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex
    socket.getaddrinfo = getaddrinfo


def install_fixtures(registry, fixtures, calls):
    """Replace every non-local tool's handler in tools._REGISTRY. `fixtures`
    maps tool name -> canned text; `calls` (a list) receives one dict per
    sandboxed call, so a test can assert no mutation was attempted."""
    for name, (schema, _) in list(registry.items()):
        if name in LOCAL_TOOLS:
            continue

        def handler(ctx, args, name=name):
            mutating = name in MUTATING
            calls.append({"tool": name, "args": args, "mutating": mutating,
                          "fixture": name in fixtures})
            if mutating:
                return (f"(test environment) {name} was recorded and rejected: "
                        "nothing was sent to the broker or mail server.")
            if name in fixtures:
                return fixtures[name]
            raise RuntimeError(f"the test environment has no fixture for {name}; "
                               "it never reaches a broker, mail or Discord")
        registry[name] = (schema, handler)


def enter(run_root, production_roots=()):
    """Make this process a sandbox for `run_root`, and import the app from
    it. Returns the resolved configured paths."""
    run_root = Path(run_root).resolve()
    check_layout(run_root, production_roots)
    scrub_env(run_root)
    guard_network()
    sys.path.insert(0, str(run_root))
    os.chdir(run_root)
    import config
    import trading.config as tcfg
    if Path(config.BASE_DIR).resolve() != run_root:
        raise SandboxError(f"app imported from {config.BASE_DIR}, not {run_root}")
    return check_paths(run_root, (config, tcfg), production_roots)
