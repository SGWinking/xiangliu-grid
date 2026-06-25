@echo off
cd /d "%~dp0"
echo Xiangliu Grid v0.3.0: http://127.0.0.1:8765
echo Keep this window open while using the tool.
"C:\Users\ZH\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "%~dp0server.py" 8765
pause
