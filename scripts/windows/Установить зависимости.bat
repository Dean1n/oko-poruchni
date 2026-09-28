@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0..\.."

set "WHEELS=%CD%\installers\wheels-win-py312"
set "PYTHON_EXE="

py -3.12 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 set "PYTHON_EXE=py -3.12"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"

if not defined PYTHON_EXE (
  echo Python 3.12 x64 не найден.
  echo Сначала запустите УСТАНОВИТЬ_ОФЛАЙН.bat.
  pause
  exit /b 1
)

if not exist "%WHEELS%\*.whl" (
  echo Не найдена папка с автономными пакетами:
  echo %WHEELS%
  pause
  exit /b 1
)

echo Установка библиотек без интернета...
%PYTHON_EXE% -m pip install --no-index --find-links "%WHEELS%" -r requirements.txt
if errorlevel 1 goto error

echo.
echo Библиотеки установлены успешно.
pause
exit /b 0

:error
echo.
echo Ошибка установки. Проверьте, что используется Windows 10/11 x64 и Python 3.12 x64.
pause
exit /b 1
