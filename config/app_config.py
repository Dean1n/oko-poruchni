import os
import json
from utils.paths import project_path
from typing import Dict, Any, Optional

class AppConfig:
    """Класс для управления конфигурацией приложения"""
    
    def __init__(self, config_path: str = "config/app_config.json"):
        """
        Инициализация с указанием пути к файлу конфигурации
        
        Args:
            config_path: Путь к JSON-файлу с конфигурацией
        """
        self.config_path = project_path(config_path)
        self.default_config = {
            "application": {
                "name": "Система видеоаналитики безопасного передвижения по лестницам",
                "version": "1.0.0",
                "debug_mode": False,
                "log_level": "INFO",
                "log_file": "logs/app.log",
                "database_path": "data/analytics.db",
                "temp_dir": "temp"
            },
            "cuda": {
                "enabled": True,
                "device": 0
            },
            "models": {
                "yolo_pose_path": "models/model.pt",
                "tracker_config": "models/bytetrack.yaml"
            },
            "video_processing": {
                "default_violation_threshold": 3.0,
                "motion_detection": {
                    "threshold": 25,
                    "min_area": 500,
                    "blur_size": 21,
                    "dilate_iterations": 2
                },
                "visualization": {
                    "roi_color": [255, 0, 0],
                    "roi_alpha": 0.3,
                    "stair_color": [128, 128, 128],
                    "stair_alpha": 0.2,
                    "handrail_color": [0, 255, 0],
                    "handrail_alpha": 0.4,
                    "holding_box_color": [0, 255, 0],
                    "violation_box_color": [255, 0, 0]
                }
            },
            "ui": {
                "window_width": 1280,
                "window_height": 800,
                "theme": "default",
                "language": "ru",
                "font_size": 10
            },
            "paths": {
                "cameras_config": "config/cameras.json",
                "default_videos_dir": "data/videos",
                "default_violations_dir": "data/violations",
                "exports_dir": "exports"
            }
        }
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Загрузка конфигурации из файла
        
        Returns:
            Словарь с настройками
        """
        if not os.path.exists(self.config_path):
            # Создаем директорию, если она не существует
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Создаем файл с настройками по умолчанию
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.default_config, f, ensure_ascii=False, indent=2)
            
            return self.default_config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
                # Объединяем загруженную конфигурацию с настройками по умолчанию
                return self._merge_configs(self.default_config, config)
        except Exception as e:
            print(f"Ошибка при загрузке конфигурации: {e}")
            return self.default_config
    
    def _merge_configs(self, default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
        """
        Объединение настроек по умолчанию с пользовательскими
        
        Args:
            default: Настройки по умолчанию
            custom: Пользовательские настройки
            
        Returns:
            Объединенный словарь с настройками
        """
        result = default.copy()
        
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, section: str, key: Optional[str] = None, default_value: Any = None) -> Any:
        """
        Получение значения настройки
        
        Args:
            section: Раздел настроек
            key: Ключ настройки (None для получения всего раздела)
            default_value: Значение по умолчанию
            
        Returns:
            Значение настройки или раздел целиком
        """
        if section not in self.config:
            return default_value
        
        if key is None:
            return self.config[section]
        
        if key not in self.config[section]:
            return default_value
        
        value = self.config[section][key]
        if section in ("paths", "models") or (section == "application" and key in ("database_path", "log_file", "temp_dir")):
            return project_path(value)
        return value
    
    def set(self, section: str, key: str, value: Any) -> bool:
        """
        Установка значения настройки
        
        Args:
            section: Раздел настроек
            key: Ключ настройки
            value: Значение настройки
            
        Returns:
            True, если значение успешно установлено
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
        
        return self.save()
    
    def save(self) -> bool:
        """
        Сохранение конфигурации в файл
        
        Returns:
            True, если сохранение успешно
        """
        try:
            # Создаем директорию, если она не существует
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Ошибка при сохранении конфигурации: {e}")
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """
        Получение всей конфигурации
        
        Returns:
            Словарь со всеми настройками
        """
        return self.config
    
    def get_path(self, key: str, default_value: str = "") -> str:
        """
        Получение пути из конфигурации
        
        Args:
            key: Ключ пути в разделе "paths"
            default_value: Значение по умолчанию
            
        Returns:
            Путь из конфигурации
        """
        return self.get("paths", key, default_value)
    
    def get_cuda_settings(self) -> Dict[str, Any]:
        """
        Получение настроек CUDA
        
        Returns:
            Словарь с настройками CUDA
        """
        return self.get("cuda", None, {"enabled": False, "device": 0})
    
    def get_model_path(self, model_key: str, default_value: str = "") -> str:
        """
        Получение пути к модели
        
        Args:
            model_key: Ключ модели в разделе "models"
            default_value: Значение по умолчанию
            
        Returns:
            Путь к модели
        """
        return self.get("models", model_key, default_value)
    
    def get_ui_settings(self) -> Dict[str, Any]:
        """
        Получение настроек интерфейса
        
        Returns:
            Словарь с настройками интерфейса
        """
        return self.get("ui", None, {})
    
    def get_processing_settings(self) -> Dict[str, Any]:
        """
        Получение настроек обработки видео
        
        Returns:
            Словарь с настройками обработки видео
        """
        return self.get("video_processing", None, {})
    
    def reset_to_defaults(self) -> bool:
        """
        Сброс настроек к значениям по умолчанию
        
        Returns:
            True, если сброс успешен
        """
        self.config = self.default_config.copy()
        return self.save()
