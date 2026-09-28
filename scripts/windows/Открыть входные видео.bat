@echo off
cd /d "%~dp0..\.."
if not exist "data\videos" mkdir "data\videos"
start "" explorer.exe "%CD%\data\videos"
