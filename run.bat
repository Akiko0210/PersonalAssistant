@echo off
REM Launch the voice agent. Double-click to run, or pass args, e.g.:
REM   run.bat --selftest
REM   run.bat --miccheck
REM   run.bat --ingest     (embed new files in the knowledge\ folder, then exit)
REM   run.bat --kb-list    (list ingested knowledge sources, then exit)
cd /d "%~dp0"
python voice_agent.py %*
echo.
echo Agent exited.
pause
