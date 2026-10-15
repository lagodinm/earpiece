@echo off
chcp 65001 >nul
title earpiece
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" earpiece.py %*
pause
