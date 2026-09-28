#!/usr/bin/env python3
"""
Главный модуль запуска системы видеоаналитики безопасного передвижения по лестницам

Система контролирует соблюдение правил безопасности:
- Персонал должен держаться кистью руки за поручень при движении по лестницам
- Автоматическое обнаружение и запись нарушений
- Аналитика и отчетность
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# Добавляем текущую директорию в путь для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(Path(__file__).resolve().parent)

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QLocale, QTranslator, Qt
from PyQt5.QtGui import QIcon

from config.app_config import AppConfig
from gui.main_window import MainWindow
from utils.services import initialize_services


def setup_logging(config: AppConfig) -> bool:
    """
    Настройка системы логирования
    
    Args:
        config: Объект конфигурации приложения
        
    Returns:
        True если настройка успешна
    """
    try:
        # Получаем настройки логирования
        log_level = config.get("application", "log_level", "INFO")
        log_file = config.get("application", "log_file", "logs/app.log")
        debug_mode = config.get("application", "debug_mode", False)
        
        # Создаем директорию для логов
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Настройка уровня логирования
        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        level = log_level_map.get(log_level.upper(), logging.INFO)
        
        # Форматирование логов
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Настройка корневого логгера
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Очистка существующих обработчиков
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Обработчик для файла
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Не удалось создать файловый логгер: {e}")
        
        # Обработчик для консоли (только если режим отладки или нет файлового логгера)
        if debug_mode or not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Подавление избыточных логов от некоторых библиотек
        logging.getLogger('ultralytics').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        
        logging.info("Система логирования настроена")
        return True
        
    except Exception as e:
        print(f"Ошибка настройки логирования: {e}")
        traceback.print_exc()
        return False


def setup_application() -> QApplication:
    """
    Настройка Qt приложения
    
    Returns:
        Настроенное приложение QApplication
    """
    # Включаем высокое DPI
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Создаем приложение
    app = QApplication(sys.argv)
    
    # Настройка приложения
    app.setApplicationName("Система видеоаналитики безопасности")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Safety Analytics")
    app.setOrganizationDomain("safety-analytics.local")
    
    # Установка иконки приложения (если есть)
    icon_path = "resources/icons/app_icon.ico"
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # Настройка локализации
    locale = QLocale.system().name()
    translator = QTranslator()
    
    # Попытка загрузки переводов
    translations_path = "resources/translations"
    if os.path.exists(translations_path):
        if translator.load(locale, translations_path):
            app.installTranslator(translator)
    
    return app


def check_dependencies() -> tuple[bool, list]:
    """
    Проверка зависимостей системы
    
    Returns:
        Tuple[bool, list]: (все_зависимости_доступны, список_ошибок)
    """
    errors = []
    
    # Проверка критических библиотек
    critical_imports = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('ultralytics', 'Ultralytics YOLO'),
        ('PyQt5', 'PyQt5')
    ]
    
    for module_name, display_name in critical_imports:
        try:
            __import__(module_name)
        except ImportError as e:
            errors.append(f"Не удалось импортировать {display_name}: {e}")
    
    # Проверка доступности CUDA (предупреждение, не критическая ошибка)
    try:
        import torch
        if not torch.cuda.is_available():
            logging.warning("CUDA недоступна. Будет использоваться CPU для обработки.")
    except Exception as e:
        logging.warning(f"Не удалось проверить доступность CUDA: {e}")
    
    return len(errors) == 0, errors


def show_error_dialog(title: str, message: str, details: str = None):
    """
    Показ диалога с ошибкой
    
    Args:
        title: Заголовок диалога
        message: Основное сообщение
        details: Дополнительные детали ошибки
    """
    try:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        
        if details:
            msg_box.setDetailedText(details)
        
        msg_box.exec_()
    except Exception:
        # Если не удается показать Qt диалог, выводим в консоль
        print(f"ОШИБКА: {title}")
        print(f"Сообщение: {message}")
        if details:
            print(f"Детали: {details}")


def main():
    """Главная функция запуска приложения"""
    exit_code = 0
    
    try:
        # Проверка зависимостей
        deps_ok, dep_errors = check_dependencies()
        if not deps_ok:
            error_msg = "Не все зависимости установлены:\n" + "\n".join(dep_errors)
            print(f"КРИТИЧЕСКАЯ ОШИБКА: {error_msg}")
            return 1
        
        # Загрузка конфигурации
        try:
            config = AppConfig()
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            return 1
        
        # Настройка логирования
        if not setup_logging(config):
            print("Продолжение работы без полного логирования")
        
        logging.info("Запуск приложения")
        
        # Инициализация сервисов
        try:
            services_status = initialize_services()
            if not services_status.get("database", False):
                logging.error("Критическая ошибка инициализации базы данных")
                return 1
        except Exception as e:
            logging.error(f"Ошибка инициализации сервисов: {e}")
            return 1
        
        # Создание Qt приложения
        app = setup_application()
        
        # Создание и показ главного окна
        try:
            main_window = MainWindow()
            main_window.show()
            
            # Запуск цикла обработки событий
            exit_code = app.exec_()
            
        except Exception as e:
            logging.error(f"Ошибка создания главного окна: {e}")
            show_error_dialog(
                "Ошибка запуска",
                "Не удалось создать главное окно приложения",
                str(e)
            )
            exit_code = 1
            
    except KeyboardInterrupt:
        logging.info("Приложение прервано пользователем")
        exit_code = 0
        
    except Exception as e:
        error_msg = f"Критическая ошибка: {e}"
        logging.error(error_msg)
        print(error_msg)
        traceback.print_exc()
        exit_code = 1
        
    finally:
        logging.info(f"Завершение работы приложения с кодом {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
