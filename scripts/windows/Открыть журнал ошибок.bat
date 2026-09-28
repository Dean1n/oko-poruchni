@echo off
cd /d "%~dp0..\.."
if not exist "logs\app.log" (
  echo Журнал пока не создан. Сначала запустите программу через Запуск.bat.
  pause
  exit /b 0
)
start "" notepad.exe "%CD%\logs\app.log"
