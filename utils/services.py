import os
import sys
import sqlite3
import logging
import torch
import time
from typing import Dict, Any, Optional

def check_cuda_availability() -> Dict[str, Any]:
    """
    Проверка доступности CUDA и информации о GPU
    
    Returns:
        Словарь с информацией о CUDA и GPU
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
        "current_device": None,
        "device_name": None
    }
    
    if info["cuda_available"]:
        info["device_count"] = torch.cuda.device_count()
        
        for i in range(info["device_count"]):
            device_info = {
                "idx": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(i),
                "memory_reserved": torch.cuda.memory_reserved(i)
            }
            info["devices"].append(device_info)
        
        if info["device_count"] > 0:
            info["current_device"] = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(info["current_device"])
    
    return info

def init_database(db_path: str = "data/analytics.db") -> bool:
    try:
        from core.data_manager import DataManager
        from config.camera_config import CameraConfig
        manager = DataManager(db_path)
        for camera in CameraConfig().get_all_cameras():
            manager.add_camera(camera["camera_id"], camera.get("location", ""),
                               camera.get("logo_path", ""), camera.get("description", ""))
        return True
    except Exception:
        logging.exception("Ошибка инициализации базы данных")
        return False

def check_directories(paths: Dict[str, str]) -> Dict[str, bool]:
    """
    Проверка и создание необходимых директорий
    
    Args:
        paths: Словарь с путями к директориям
        
    Returns:
        Словарь с результатами проверки
    """
    results = {}
    
    for name, path in paths.items():
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
                results[name] = True
            except Exception as e:
                logging.error(f"Ошибка при создании директории {name} ({path}): {e}")
                results[name] = False
        else:
            results[name] = True
    
    return results

def check_model_files(model_paths: Dict[str, str]) -> Dict[str, bool]:
    """
    Проверка наличия файлов моделей
    
    Args:
        model_paths: Словарь с путями к моделям
        
    Returns:
        Словарь с результатами проверки
    """
    results = {}
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            results[name] = True
        else:
            logging.warning(f"Файл модели {name} не найден: {path}")
            results[name] = False
    
    return results

def preload_yolo_model(model_path: str) -> Optional[Any]:
    """
    Предварительная загрузка модели YOLO для ускорения первого запуска
    
    Args:
        model_path: Путь к файлу модели
        
    Returns:
        Загруженная модель или None в случае ошибки
    """
    try:
        from ultralytics import YOLO
        
        start_time = time.time()
        model = YOLO(model_path)
        load_time = time.time() - start_time
        
        logging.info(f"Модель YOLO загружена за {load_time:.2f} секунд: {model_path}")
        return model
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели YOLO: {e}")
        return None

def initialize_services(config_path: str = "config/app_config.json") -> Dict[str, Any]:
    """
    Инициализация всех необходимых сервисов при запуске приложения
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Словарь с информацией о статусе сервисов
    """
    results = {
        "cuda": None,
        "database": False,
        "directories": {},
        "models": {},
        "yolo_model": None
    }
    
    try:
        from config.app_config import AppConfig
        
        # Загружаем конфигурацию
        app_config = AppConfig(config_path)
        
        # Проверяем CUDA
        results["cuda"] = check_cuda_availability()
        
        # Инициализируем базу данных
        db_path = app_config.get("application", "database_path", "data/analytics.db")
        results["database"] = init_database(db_path)
        
        # Проверяем и создаем директории
        paths = {
            "videos": app_config.get_path("default_videos_dir", "data/videos"),
            "violations": app_config.get_path("default_violations_dir", "data/violations"),
            "exports": app_config.get_path("exports_dir", "exports"),
            "logs": os.path.dirname(app_config.get("application", "log_file", "logs/app.log")),
            "temp": app_config.get("application", "temp_dir", "temp")
        }
        results["directories"] = check_directories(paths)
        
        # Проверяем наличие файлов моделей
        model_paths = {
            "yolo_pose": app_config.get_model_path("yolo_pose_path", "models/yolov8x-pose.pt"),
            "tracker_config": app_config.get_model_path("tracker_config", "models/bytetrack.yaml")
        }
        results["models"] = check_model_files(model_paths)
        
        # Предварительно загружаем модель YOLO, если включена поддержка CUDA
        if results["cuda"]["cuda_available"] and app_config.get("cuda", "enabled", True):
            # Устанавливаем устройство CUDA
            device = app_config.get("cuda", "device", 0)
            torch.cuda.set_device(device)
            
            # Загружаем модель
            yolo_path = model_paths["yolo_pose"]
            if results["models"]["yolo_pose"]:
                results["yolo_model"] = preload_yolo_model(yolo_path)
    
    except Exception as e:
        logging.error(f"Ошибка при инициализации сервисов: {e}")
    
    return results

if __name__ == "__main__":
    """
    Запуск сервисов из командной строки
    """
    import argparse
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Разбор аргументов командной строки
    parser = argparse.ArgumentParser(description='Инициализация сервисов для видеоаналитики')
    parser.add_argument('--config', help='Путь к файлу конфигурации', default='config/app_config.json')
    
    args = parser.parse_args()
    
    # Инициализация сервисов
    results = initialize_services(args.config)
    
    # Вывод результатов
    logging.info("Инициализация сервисов завершена")
    logging.info(f"CUDA: {'Доступна' if results['cuda']['cuda_available'] else 'Недоступна'}")
    logging.info(f"База данных: {'Инициализирована' if results['database'] else 'Ошибка инициализации'}")
    
    logging.info("Директории:")
    for name, status in results["directories"].items():
        logging.info(f"  {name}: {'OK' if status else 'Ошибка'}")
    
    logging.info("Модели:")
    for name, status in results["models"].items():
        logging.info(f"  {name}: {'Найдена' if status else 'Не найдена'}")
    
    logging.info(f"Модель YOLO: {'Загружена' if results['yolo_model'] is not None else 'Не загружена'}")
