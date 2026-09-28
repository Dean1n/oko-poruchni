@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."
python -c "import cv2, numpy, torch, ultralytics, PyQt5; print('Python:', __import__('sys').version); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Все основные библиотеки установлены')"
if errorlevel 1 (
  echo Проверка не пройдена. Запустите scripts\windows\Установить зависимости.bat.
)
pause
