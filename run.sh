#!/usr/bin/env bash
# Launch the voice agent (macOS/Linux twin of run.bat). Run directly, or pass
# args, e.g.:
#   ./run.sh --selftest
#   ./run.sh --miccheck
#   ./run.sh --ingest     (embed new files in the knowledge/ folder, then exit)
#   ./run.sh --kb-list    (list ingested knowledge sources, then exit)
cd "$(dirname "$0")"

# Prefer the project venv when one exists — on macOS/Linux the dependencies
# live there rather than in the system Python, so a plain ./run.sh works
# without activating anything first.
PY="python3"
[ -x .venv/bin/python ] && PY=".venv/bin/python"

# A plain launch absorbs anything new in knowledge/ FIRST -- video included --
# so the agent comes up with it already searchable. This runs as its own
# process, which matters two ways: the console shows transcription progress
# rather than a silent stall, and the single-instance lock is released before
# the agent asks for it. A scan with nothing new costs a couple of seconds.
# Skipped when arguments are given, so --ingest / --selftest / --kb-list
# don't run the scan twice.
if [ $# -eq 0 ]; then
    echo "Checking the knowledge folder for new material..."
    echo "Video is transcribed here, once per file - about 20 minutes per hour"
    echo "of recording, with progress shown. Leave this window open; the agent"
    echo "starts as soon as it finishes."
    echo
    "$PY" voice_agent.py --ingest
    echo
fi

"$PY" voice_agent.py "$@"
echo
echo "Agent exited."
