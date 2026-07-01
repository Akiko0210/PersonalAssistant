@echo off
REM Launch the voice agent. Double-click to run, or pass args, e.g.:
REM   run.bat --selftest
REM   run.bat --miccheck
cd /d "%~dp0"
python voice_agent.py %*
echo.
echo Agent exited.
pause
