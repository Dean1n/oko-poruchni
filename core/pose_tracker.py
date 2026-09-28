import cv2
import numpy as np
import time
import logging
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any, Optional
from threading import Lock
from utils.paths import project_path
import torch


class PoseTracker:
    """Улучшенный трекер позы с детекцией нарушений безопасности на лестницах"""
    
    # Индексы ключевых точек YOLO Pose
    LEFT_WRIST_IDX = 9
    RIGHT_WRIST_IDX = 10
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Инициализация трекера позы
        
        Args:
            model_path: Путь к модели YOLO Pose
            device: Устройство для вычислений ('auto', 'cpu', 'cuda')
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.model = None
        self.device = self._setup_device(device)
        self.model_lock = Lock()
        
        # Инициализация модели
        self._load_model()
        
        # Исправленные параметры трекера для новой версии ultralytics
        self.track_params = {
            "tracker": project_path("models/bytetrack.yaml"), 
            "classes": [0],  # Только люди
            "verbose": False,
            "persist": True,
            "conf": 0.3,  # Повышен порог уверенности
            "iou": 0.4,   # Повышен IOU порог
            "imgsz": 640,  # Размер изображения
            "device": self.device
        }
        
        # Зоны и состояние
        self.stair_zones = []
        self.handrail_zones = []
        self.tracked_persons = {}
        self.current_track_ids = set()
        self.last_video_time = None
        
        # Статистика и метрики
        self.stats = {
            'total_detections': 0,
            'total_violations': 0,
            'processing_time': 0,
            'frame_count': 0
        }
        
    def _setup_device(self, device: str) -> str:
        """Настройка устройства для вычислений"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self.logger.info(f"Используется CUDA: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                self.logger.info("Используется CPU")
        return device
        
    def _load_model(self):
        """Загрузка модели YOLO с обработкой ошибок"""
        try:
            with self.model_lock:
                self.model = YOLO(self.model_path)
                if self.device == 'cuda':
                    self.model.to('cuda')
                self.logger.info(f"Модель YOLO загружена: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели YOLO: {e}")
            raise
    
    def set_zones(self, stair_zones: List[Dict[str, Any]]) -> bool:
        """
        Установка зон лестниц и поручней с валидацией
        
        Args:
            stair_zones: Список зон лестниц с поручнями
            
        Returns:
            True если зоны установлены успешно
        """
        try:
            self.stair_zones = []
            self.handrail_zones = []
            
            for stair in stair_zones:
                if not self._validate_stair_zone(stair):
                    self.logger.warning(f"Некорректная зона лестницы: {stair}")
                    continue
                    
                stair_id = stair["id"]
                stair_coords = np.array(stair["coordinates"], dtype=np.int32)
                
                self.stair_zones.append({
                    "id": stair_id,
                    "polygon": stair_coords
                })
                
                # Обработка поручней
                for handrail in stair.get("handrails", []):
                    if not self._validate_handrail_zone(handrail):
                        self.logger.warning(f"Некорректный поручень: {handrail}")
                        continue
                        
                    handrail_id = handrail["id"]
                    handrail_coords = np.array(handrail["coordinates"], dtype=np.int32)
                    
                    self.handrail_zones.append({
                        "stair_id": stair_id,
                        "id": handrail_id,
                        "polygon": handrail_coords
                    })
            
            self.logger.info(f"Установлено зон: лестниц - {len(self.stair_zones)}, поручней - {len(self.handrail_zones)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при установке зон: {e}")
            return False
    
    def _validate_stair_zone(self, stair: Dict[str, Any]) -> bool:
        """Валидация зоны лестницы"""
        required_fields = ["id", "coordinates"]
        for field in required_fields:
            if field not in stair:
                return False
        
        coords = stair["coordinates"]
        return isinstance(coords, list) and len(coords) >= 3
    
    def _validate_handrail_zone(self, handrail: Dict[str, Any]) -> bool:
        """Валидация зоны поручня"""
        required_fields = ["id", "coordinates"]
        for field in required_fields:
            if field not in handrail:
                return False
        
        coords = handrail["coordinates"]
        return isinstance(coords, list) and len(coords) >= 3
    
    def process(self, frame: np.ndarray, violation_threshold: float, video_time_s: float) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Обработка кадра с улучшенной логикой и обработкой ошибок
        
        Args:
            frame: Кадр видео
            violation_threshold: Порог времени для нарушения (в секундах)
            video_time_s: Время текущего кадра от начала видео, в секундах
            
        Returns:
            Tuple[List[Dict], np.ndarray]: (список_нарушений, визуализация)
        """
        start_time = time.time()
        current_time = float(video_time_s)
        if not np.isfinite(current_time) or current_time < 0:
            raise ValueError("Некорректная временная метка кадра")
        if self.last_video_time is not None and current_time < self.last_video_time:
            self.reset_tracking()
        self.last_video_time = current_time
        self.current_track_ids = set()
        violations = []
        
        try:
            # Валидация входных данных
            if frame is None or frame.size == 0:
                self.logger.warning("Получен пустой кадр")
                return violations, frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            
            visualization = frame.copy()
            
            # Запуск трекинга с обработкой ошибок
            with self.model_lock:
                if self.model is None:
                    raise RuntimeError("Модель YOLO не загружена")
                
                # Исправленный вызов трекинга без недоступных параметров
                results = self.model.track(
                    frame,
                    tracker=self.track_params["tracker"],
                    classes=self.track_params["classes"],
                    verbose=self.track_params["verbose"],
                    persist=self.track_params["persist"],
                    conf=self.track_params["conf"],
                    iou=self.track_params["iou"],
                    imgsz=self.track_params["imgsz"],
                    device=self.track_params["device"]
                )
            
            # Проверка результатов
            if not results or len(results) == 0:
                self._remove_inactive_tracks(set(), current_time)
                self._clean_old_tracks(current_time)
                self._draw_zones(visualization)
                return violations, visualization
            
            result = results[0]
            
            # Проверяем наличие треков
            if result.boxes is None or result.boxes.id is None:
                self._remove_inactive_tracks(set(), current_time)
                self._clean_old_tracks(current_time)
                self._draw_zones(visualization)
                return violations, visualization
            
            # Получение ID треков
            track_ids = result.boxes.id.int().cpu().tolist()
            self.current_track_ids = set(track_ids)
            self.stats['total_detections'] += len(track_ids)
            
            # Отображение зон
            self._draw_zones(visualization)
            
            # Обработка треков
            violations = self._process_tracks(results, track_ids, current_time, violation_threshold, visualization)
            
            # Очистка старых треков
            self._clean_old_tracks(current_time)
            
            # Обновление статистики
            processing_time = time.time() - start_time
            self.stats['processing_time'] += processing_time
            self.stats['frame_count'] += 1
            self.stats['total_violations'] += len(violations)
            
            return violations, visualization
            
        except Exception:
            self.logger.exception("Ошибка при обработке кадра")
            raise

    def _process_tracks(self, results, track_ids, current_time, violation_threshold, visualization):
        """Обработка треков с улучшенной логикой"""
        violations = []
        current_tracks = set(track_ids)
        
        for i, track_id in enumerate(track_ids):
            try:
                # Получение данных трека
                bbox = results[0].boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                # Инициализация трека если необходимо
                if track_id not in self.tracked_persons:
                    self.tracked_persons[track_id] = {
                        'violation_timer': 0.0,
                        'last_seen': current_time,
                        'bbox': (x1, y1, x2, y2),
                        'in_stair_zone': False,
                        'holding_handrail': False,
                        'consecutive_violations': 0,
                        'detection_confidence': 0.0,
                        'contact_state': 'unknown',
                        'stair_zone_id': None
                    }
                
                # Обновление bbox
                self.tracked_persons[track_id]['bbox'] = (x1, y1, x2, y2)
                self.tracked_persons[track_id]['detection_confidence'] = float(results[0].boxes.conf[i].cpu())
                
                # Получение ключевых точек
                keypoints = None
                if results[0].keypoints is not None and i < len(results[0].keypoints):
                    keypoints = results[0].keypoints[i].data[0].cpu().numpy()
                
                # Проверка нахождения в зоне лестницы
                person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                stair_zone = next((zone for zone in self.stair_zones
                                   if self._is_point_in_any_zone(person_center, [zone])), None)
                in_stair_zone = stair_zone is not None
                stair_id = stair_zone["id"] if stair_zone else None
                if self.tracked_persons[track_id].get("stair_zone_id") != stair_id:
                    self._reset_person_state(track_id, current_time)
                self.tracked_persons[track_id]["stair_zone_id"] = stair_id
                
                self.tracked_persons[track_id]['in_stair_zone'] = in_stair_zone
                
                if in_stair_zone:
                    # Обработка нарушений в зоне лестницы
                    violation = self._process_stair_zone_violation(
                        track_id, keypoints, current_time, violation_threshold
                    )
                    
                    if violation:
                        violations.append(violation)
                    
                    # Визуализация
                    self._draw_person_visualization(visualization, track_id, x1, y1, x2, y2, keypoints, violation is not None)
                
                else:
                    # Сброс состояния если вне зоны лестницы
                    self._reset_person_state(track_id, current_time)
                    
                    # Простая визуализация для персонала вне зоны
                    cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 255, 0), 2)
                
            except Exception as e:
                self.logger.exception(f"Ошибка при обработке трека {track_id}")
                raise
        
        # Удаление неактивных треков
        self._remove_inactive_tracks(current_tracks, current_time)
        
        return violations
    
    def _process_stair_zone_violation(self, track_id, keypoints, current_time, violation_threshold):
        person = self.tracked_persons[track_id]
        rails = [zone for zone in self.handrail_zones
                 if zone["stair_id"] == person.get("stair_zone_id")]
        wrists = []
        for index in (self.LEFT_WRIST_IDX, self.RIGHT_WRIST_IDX):
            wrist = None
            if keypoints is not None and index < len(keypoints):
                point = keypoints[index]
                if (len(point) >= 3 and np.isfinite(point[:3]).all()
                        and point[2] > 0.5 and point[0] > 0 and point[1] > 0):
                    wrist = (float(point[0]), float(point[1]))
            wrists.append(wrist)
        holding = any(w is not None and self._is_point_in_any_zone(w, rails) for w in wrists)
        state = ('holding' if holding else 'not_holding'
                 if rails and all(w is not None for w in wrists) else 'unknown')
        previous_state = person.get('contact_state', 'unknown')
        time_delta = max(0.0, current_time - person['last_seen'])
        person['last_seen'] = current_time
        person['holding_handrail'] = holding
        person['contact_state'] = state
        if state == 'not_holding':
            # Count only intervals bounded by two valid observations of non-contact.
            if previous_state == 'not_holding':
                person['violation_timer'] += time_delta
                person['consecutive_violations'] += 1
            else:
                person['violation_timer'] = 0.0
                person['consecutive_violations'] = 1
        else:
            person['violation_timer'] = 0.0
            person['consecutive_violations'] = 0
        if (person['violation_timer'] > violation_threshold
                and person['consecutive_violations'] >= 3):
            return {'track_id': track_id, 'bbox': person['bbox'],
                    'violation_time': person['violation_timer'],
                    'stair_zone_id': person.get('stair_zone_id'),
                    'confidence': person.get('detection_confidence', 0.0)}
        return None
    
    def _reset_person_state(self, track_id, current_time):
        """Сброс состояния персонала вне зоны лестницы"""
        if track_id in self.tracked_persons:
            self.tracked_persons[track_id]['last_seen'] = current_time
            self.tracked_persons[track_id]['violation_timer'] = 0.0
            self.tracked_persons[track_id]['holding_handrail'] = False
            self.tracked_persons[track_id]['consecutive_violations'] = 0
            self.tracked_persons[track_id]['contact_state'] = 'unknown'
    
    def _remove_inactive_tracks(self, current_tracks, current_time):
        """Удаление неактивных треков"""
        inactive_tracks = []
        for track_id in self.tracked_persons:
            if track_id not in current_tracks:
                self.tracked_persons[track_id]['contact_state'] = 'unknown'
                self.tracked_persons[track_id]['violation_timer'] = 0.0
                self.tracked_persons[track_id]['consecutive_violations'] = 0
                time_since_last_seen = current_time - self.tracked_persons[track_id]['last_seen']
                if time_since_last_seen > 2.0:  # 2 секунды тайм-аут
                    inactive_tracks.append(track_id)
        
        for track_id in inactive_tracks:
            del self.tracked_persons[track_id]
    
    def _clean_old_tracks(self, current_time, timeout=5.0):
        """Очистка старых треков"""
        to_delete = []
        for track_id, info in self.tracked_persons.items():
            if current_time - info['last_seen'] > timeout:
                to_delete.append(track_id)
        
        for track_id in to_delete:
            del self.tracked_persons[track_id]
    
    def _is_point_in_any_zone(self, point, zones):
        """Проверка нахождения точки в любой из зон"""
        try:
            for zone in zones:
                if cv2.pointPolygonTest(zone["polygon"], (float(point[0]), float(point[1])), False) >= 0:
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Ошибка при проверке точки в зоне: {e}")
            return False
    
    def _draw_zones(self, frame):
        """Отрисовка зон лестниц и поручней"""
        try:
            # Лестницы
            for stair in self.stair_zones:
                cv2.polylines(frame, [stair["polygon"]], True, (128, 128, 128), 2)
            
            # Поручни
            for handrail in self.handrail_zones:
                cv2.polylines(frame, [handrail["polygon"]], True, (0, 255, 0), 2)
        except Exception as e:
            self.logger.error(f"Ошибка при отрисовке зон: {e}")
    
    def _draw_person_visualization(self, frame, track_id, x1, y1, x2, y2, keypoints, is_violation):
        """Отрисовка визуализации персонала"""
        try:
            # Цвет рамки в зависимости от нарушения
            unknown = self.tracked_persons[track_id].get('contact_state') == 'unknown'
            box_color = (0, 0, 255) if is_violation else (0, 180, 255) if unknown else (0, 255, 0)
            
            # Рамка
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Информация о таймере
            violation_timer = self.tracked_persons[track_id]['violation_timer']
            status_text = f"ID: {track_id}, Time: {violation_timer:.1f}s, {self.tracked_persons[track_id].get('contact_state', 'unknown')}"
            
            cv2.putText(frame, status_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # Ключевые точки
            if keypoints is not None:
                self._draw_keypoints(frame, keypoints)
                
        except Exception as e:
            self.logger.error(f"Ошибка при отрисовке персонала {track_id}: {e}")
    
    def _draw_keypoints(self, frame, keypoints):
        """Отрисовка ключевых точек позы"""
        try:
            # Цвета для разных частей тела
            colors = {
                'nose': (255, 0, 0),
                'left_arm': (0, 255, 0),
                'right_arm': (0, 0, 255),
                'body': (255, 255, 0),
                'legs': (255, 0, 255)
            }
            
            # Рисуем точки
            for i, point in enumerate(keypoints):
                if len(point) >= 3:
                    x, y, conf = point[:3]
                    if conf > 0.5:
                        if i == 0:  # Нос
                            color = colors['nose']
                        elif i in [5, 7, 9]:  # Левая рука
                            color = colors['left_arm']
                        elif i in [6, 8, 10]:  # Правая рука
                            color = colors['right_arm']
                        elif i in [1, 2, 3, 4]:  # Тело
                            color = colors['body']
                        else:  # Ноги
                            color = colors['legs']
                        
                        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
            
            # Соединения скелета
            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4), (1, 2), (5, 6),
                (5, 7), (7, 9), (6, 8), (8, 10),
                (11, 13), (13, 15), (12, 14), (14, 16),
                (5, 11), (6, 12), (1, 5), (2, 6), (11, 12)
            ]
            
            for pair in skeleton:
                if pair[0] < len(keypoints) and pair[1] < len(keypoints):
                    pt1 = keypoints[pair[0]]
                    pt2 = keypoints[pair[1]]
                    
                    if (len(pt1) >= 3 and len(pt2) >= 3 and 
                        pt1[2] > 0.5 and pt2[2] > 0.5):
                        cv2.line(frame, 
                               (int(pt1[0]), int(pt1[1])), 
                               (int(pt2[0]), int(pt2[1])), 
                               (255, 255, 255), 1)
        except Exception as e:
            self.logger.error(f"Ошибка при отрисовке ключевых точек: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики работы трекера"""
        avg_processing_time = (
            self.stats['processing_time'] / self.stats['frame_count'] 
            if self.stats['frame_count'] > 0 else 0
        )
        
        return {
            'total_detections': self.stats['total_detections'],
            'total_violations': self.stats['total_violations'],
            'active_tracks': len(self.tracked_persons),
            'avg_processing_time': avg_processing_time,
            'frame_count': self.stats['frame_count'],
            'zones_configured': len(self.stair_zones) > 0
        }
    
    def reset_statistics(self):
        """Сброс статистики"""
        self.stats = {
            'total_detections': 0,
            'total_violations': 0,
            'processing_time': 0,
            'frame_count': 0
        }
    
    def cleanup(self):
        """Очистка ресурсов"""
        try:
            with self.model_lock:
                if self.model is not None:
                    del self.model
                    self.model = None
            
            self.tracked_persons.clear()
            self.stair_zones.clear()
            self.handrail_zones.clear()
            
            # Очистка GPU памяти если используется CUDA
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.logger.info("Ресурсы трекера очищены")
            
        except Exception as e:
            self.logger.error(f"Ошибка при очистке ресурсов: {e}")

    def reset_tracking(self):
        self.tracked_persons.clear()
        self.current_track_ids.clear()
        self.last_video_time = None
        self.reset_statistics()
        predictor = getattr(self.model, 'predictor', None)
        for tracker in getattr(predictor, 'trackers', []):
            tracker.reset()
