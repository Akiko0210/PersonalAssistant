@echo off
REM Launch the voice agent. Double-click to run, or pass args, e.g.:
REM   run.bat --selftest
REM   run.bat --miccheck
REM   run.bat --ingest     (embed new files in the knowledge\ folder, then exit)
REM   run.bat --kb-list    (list ingested knowledge sources, then exit)
cd /d "%~dp0"

REM A plain launch absorbs anything new in knowledge\ FIRST -- video included --
REM so the agent comes up with it already searchable. This runs as its own
REM process, which matters two ways: the console shows transcription progress
REM rather than a silent stall, and the single-instance lock is released before
REM the agent asks for it. A scan with nothing new costs a couple of seconds.
REM Skipped when arguments are given, so --ingest / --selftest / --kb-list
REM don't run the scan twice.
if "%~1"=="" (
    echo Checking the knowledge folder for new material...
    echo Video is transcribed here, once per file - about 20 minutes per hour
    echo of recording, with progress shown. Leave this window open; the agent
    echo starts as soon as it finishes.
    echo.
    python voice_agent.py --ingest
    echo.
)

python voice_agent.py %*

REM A plain launch also hands the session just ended to the claude CLI for a
REM write-up (logs\session_<date>_<HHMM>_analyzed.md). It takes minutes, so it
REM runs in its own minimized window: this one can close without killing it.
if "%~1"=="" start "Session analysis" /min python -m scripts.analyze_session
echo.
echo Agent exited.
pause
