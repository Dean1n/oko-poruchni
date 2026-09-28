@echo off
cd /d "%~dp0..\.."
if not exist "data\violations" mkdir "data\violations"
start "" explorer.exe "%CD%\data\violations"
