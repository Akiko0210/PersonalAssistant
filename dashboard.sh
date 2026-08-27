#!/usr/bin/env bash
cd "$(dirname "$0")"
# Same interpreter pick as run.sh: the venv when present, else PATH python3.
PY="python3"
[ -x .venv/bin/python ] && PY=".venv/bin/python"
"$PY" -m web.server "$@"
