import os
import json
from utils.paths import project_path
from typing import Dict, List, Any, Optional

class CameraConfig:
    """Класс для управления конфигурацией камер"""
    
    def __init__(self, config_path: str = "config/cameras.json"):
        """
        Инициализация с указанием пути к файлу конфигурации
        
        Args:
            config_path: Путь к JSON-файлу с конфигурацией камер
        """
        self.config_path = project_path(config_path)
        self._ensure_config_file()
    
    def _ensure_config_file(self):
        """Создание файла конфигурации, если он не существует"""
        # Создаем директорию для конфига, если она не существует
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Проверяем существование файла
        if not os.path.exists(self.config_path):
            # Создаем пустой файл конфигурации
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump({"cameras": []}, f, ensure_ascii=False, indent=2)
    
    def get_all_cameras(self) -> List[Dict[str, Any]]:
        """
        Получение списка всех камер
        
        Returns:
            Список камер с их настройками
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                cameras = config.get("cameras", [])
                for camera in cameras:
                    for key in ("video_folder_path", "violations_folder_path", "yolo_model_path", "logo_path"):
                        if camera.get(key):
                            camera[key] = project_path(camera[key])
                return cameras
        except Exception as e:
            print(f"Ошибка при чтении конфигурации: {e}")
            return []
    
    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение настроек конкретной камеры по ID
        
        Args:
            camera_id: ID камеры
            
        Returns:
            Словарь с настройками камеры или None, если камера не найдена
        """
        cameras = self.get_all_cameras()
        for camera in cameras:
            if camera.get("camera_id") == camera_id:
                return camera
        return None
    
    def add_camera(self, camera_config: Dict[str, Any]):
        """
        Добавление или обновление настроек камеры
        
        Args:
            camera_config: Словарь с настройками камеры
        """
        camera_config = camera_config.copy()
        from utils.paths import PROJECT_ROOT
        from pathlib import Path
        for key in ("video_folder_path", "violations_folder_path", "yolo_model_path", "logo_path"):
            if camera_config.get(key):
                try:
                    camera_config[key] = Path(camera_config[key]).relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    pass
        camera_id = camera_config.get("camera_id")
        if not camera_id:
            raise ValueError("ID камеры должен быть указан")
        
        # Получаем текущий список камер
        cameras = self.get_all_cameras()
        
        # Проверяем, есть ли уже такая камера
        for i, camera in enumerate(cameras):
            if camera.get("camera_id") == camera_id:
                # Обновляем настройки существующей камеры
                cameras[i] = camera_config
                break
        else:
            # Добавляем новую камеру
            cameras.append(camera_config)
        
        # Сохраняем обновленный список
        self._save_cameras(cameras)
    
    def remove_camera(self, camera_id: str) -> bool:
        """
        Удаление камеры из конфигурации
        
        Args:
            camera_id: ID камеры
            
        Returns:
            True, если камера удалена, False - если не найдена
        """
        cameras = self.get_all_cameras()
        for i, camera in enumerate(cameras):
            if camera.get("camera_id") == camera_id:
                # Удаляем камеру
                del cameras[i]
                self._save_cameras(cameras)
                return True
        return False
    
    def _save_cameras(self, cameras: List[Dict[str, Any]]):
        """
        Сохранение списка камер в файл конфигурации
        
        Args:
            cameras: Список с настройками камер
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                from pathlib import Path
                from utils.paths import PROJECT_ROOT
                portable = []
                for camera in cameras:
                    item = camera.copy()
                    for key in ("video_folder_path", "violations_folder_path", "yolo_model_path", "logo_path"):
                        if item.get(key):
                            try:
                                item[key] = Path(item[key]).relative_to(PROJECT_ROOT).as_posix()
                            except ValueError:
                                pass
                    portable.append(item)
                json.dump({"cameras": portable}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка при сохранении конфигурации: {e}")
    
    def validate_camera_config(self, camera_config: Dict[str, Any]) -> List[str]:
        """
        Проверка корректности настроек камеры
        
        Args:
            camera_config: Словарь с настройками камеры
            
        Returns:
            Список ошибок или пустой список, если ошибок нет
        """
        errors = []
        
        # Проверяем наличие обязательных полей
        required_fields = ["camera_id", "location", "video_folder_path", 
                           "violations_folder_path", "roi_zone"]
        
        for field in required_fields:
            if field not in camera_config:
                errors.append(f"Отсутствует обязательное поле: {field}")
        
        # Проверяем зону интереса
        roi_zone = camera_config.get("roi_zone", [])
        if len(roi_zone) < 3:
            errors.append("Зона интереса должна иметь не менее 3 точек")
        
        # Проверяем зоны лестниц
        stair_zones = camera_config.get("stair_zones", [])
        for i, stair in enumerate(stair_zones):
            if "coordinates" not in stair:
                errors.append(f"Отсутствуют координаты для лестницы {i+1}")
            
            if "handrails" not in stair:
                errors.append(f"Отсутствуют поручни для лестницы {i+1}")
            else:
                handrails = stair.get("handrails", [])
                if len(handrails) < 1:
                    errors.append(f"Лестница {i+1} должна иметь хотя бы 1 поручень")
                
                for j, handrail in enumerate(handrails):
                    if "coordinates" not in handrail:
                        errors.append(f"Отсутствуют координаты для поручня {j+1} лестницы {i+1}")
        
        return errors
