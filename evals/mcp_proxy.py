"""Stdio MCP server that hands Tom's tool calls back to the eval worker.

`claude -p` runs the model on the user's subscription, but the model's tool
calls must run inside the sandboxed worker: there the ToolContext, the cloned
stores and the broker fixtures live, and that is where every call gets
traced. This process owns none of that. It lists the schemas the worker
wrote and forwards each call to the worker's loopback bridge
(gateway.ToolBridge). Stdlib only, so it starts in milliseconds; the CLI
spawns a fresh one for every model call.

Usage (written into the MCP config by gateway.SubscriptionClient):
    python mcp_proxy.py <tools.json> <bridge-url>
"""

import json
import sys
import urllib.request


def _reply(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def _call(bridge, name, arguments):
    req = urllib.request.Request(
        bridge, data=json.dumps({"name": name, "arguments": arguments}).encode(),
        headers={"Content-Type": "application/json"})
    # Generous: a first recall loads the embedding model inside the worker.
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["text"]


def main(tools_path, bridge):
    with open(tools_path, encoding="utf-8") as f:
        schemas = json.load(f)
    listed = [{"name": s["name"], "description": s.get("description", ""),
               "inputSchema": s.get("input_schema") or {"type": "object"}}
              for s in schemas]
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        method, msg_id = msg.get("method"), msg.get("id")
        if msg_id is None:  # a notification (initialized, cancelled): no answer
            continue
        if method == "initialize":
            _reply(msg_id, {
                "protocolVersion": msg.get("params", {}).get("protocolVersion",
                                                             "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "tom", "version": "1"}})
        elif method == "tools/list":
            _reply(msg_id, {"tools": listed})
        elif method == "tools/call":
            p = msg.get("params", {})
            try:
                text = _call(bridge, p.get("name"), p.get("arguments") or {})
                _reply(msg_id, {"content": [{"type": "text", "text": text}]})
            except Exception as e:  # noqa: BLE001 - the model sees the failure, as dispatch() does
                _reply(msg_id, {"content": [{"type": "text",
                                             "text": f"Tool error: {e}"}],
                                "isError": True})
        elif method == "ping":
            _reply(msg_id, {})
        else:
            _reply(msg_id, error={"code": -32601, "message": f"no method {method}"})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
