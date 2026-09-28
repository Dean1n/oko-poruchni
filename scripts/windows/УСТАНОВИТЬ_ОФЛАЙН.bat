@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0..\.."

set "INSTALLER=%CD%\installers\python-3.12.10-amd64.exe"
set "VC_RUNTIME=%CD%\installers\vc_redist.x64.exe"
set "WHEELS=%CD%\installers\wheels-win-py312"
set "PYTHON_EXE="

if exist "%VC_RUNTIME%" (
  echo Установка системных библиотек Microsoft Visual C++...
  start /wait "" "%VC_RUNTIME%" /install /quiet /norestart
)

py -3.12 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 set "PYTHON_EXE=py -3.12"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"

if not defined PYTHON_EXE (
  if not exist "%INSTALLER%" (
    echo Не найден установщик Python: %INSTALLER%
    pause
    exit /b 1
  )
  echo Установка Python 3.12 x64...
  start /wait "" "%INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1
  if errorlevel 1 goto error
  if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
)

if not defined PYTHON_EXE (
  echo Python установлен, но python.exe не найден. Перезагрузите компьютер и повторите запуск.
  pause
  exit /b 1
)

if not exist "%WHEELS%\*.whl" (
  echo Не найдены автономные пакеты: %WHEELS%
  pause
  exit /b 1
)

echo Установка библиотек с флешки. Интернет не используется...
%PYTHON_EXE% -m pip install --no-index --find-links "%WHEELS%" -r requirements.txt
if errorlevel 1 goto error

echo Проверка библиотек...
%PYTHON_EXE% -c "import cv2,numpy,torch,torchvision,ultralytics,PyQt5,lapx; print('Python и библиотеки установлены успешно'); print('PyTorch:',torch.__version__); print('CUDA:',torch.cuda.is_available())"
if errorlevel 1 goto error

echo.
echo ОФЛАЙН-УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО.
echo Теперь запускайте файл Запуск.bat.
pause
exit /b 0

:error
echo.
echo Установка завершилась с ошибкой.
echo Требуется Windows 10/11 x64 и Python 3.12 x64.
pause
exit /b 1
