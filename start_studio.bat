@echo off
cd /d "%~dp0"
echo Starting PanoGS Studio...
".\.venv\Scripts\python.exe" -u -m panogs.apps.cli studio --port 8085
pause
