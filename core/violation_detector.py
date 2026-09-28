import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional
from utils.geometry import point_in_polygon

class ViolationDetector:
    """
    Модуль для определения нарушений правил безопасности на лестницах
    (отслеживает держание за поручень)
    """
    
    # Индексы ключевых точек YOLO Pose
    LEFT_WRIST_IDX = 9
    RIGHT_WRIST_IDX = 10
    
    def __init__(self, violation_threshold: float = 3.0):
        """
        Инициализация детектора нарушений
        
        Args:
            violation_threshold: Порог времени для фиксации нарушения (в секундах)
        """
        self.violation_threshold = violation_threshold
        self.tracked_persons = {}  # {track_id: {'violation_timer': float, 'last_seen': float}}
        self.stair_zones = []
        self.handrail_zones = []
    
    def set_zones(self, stair_zones: List[Dict[str, Any]]):
        """
        Установка зон лестниц и поручней
        
        Args:
            stair_zones: Список зон лестниц с поручнями из конфигурации
        """
        self.stair_zones = []
        self.handrail_zones = []
        
        for stair in stair_zones:
            stair_id = stair["id"]
            stair_coords = stair["coordinates"]
            self.stair_zones.append({
                "id": stair_id,
                "polygon": stair_coords
            })
            
            for handrail in stair.get("handrails", []):
                handrail_id = handrail["id"]
                handrail_coords = handrail["coordinates"]
                self.handrail_zones.append({
                    "stair_id": stair_id,
                    "id": handrail_id,
                    "polygon": handrail_coords
                })
    
    def check_violations(self, 
                        results: Any, 
                        current_time: float) -> List[Dict[str, Any]]:
        """
        Проверка наличия нарушений на основе результатов отслеживания позы
        
        Args:
            results: Результаты трекинга от YOLO Pose
            current_time: Текущее время для обновления таймеров
            
        Returns:
            Список обнаруженных нарушений
        """
        violations = []
        
        # Отслеживаем текущие треки для последующей очистки
        current_tracks = set()
        
        # Проверяем, есть ли результаты трекинга
        if results[0].boxes.id is None:
            # Очищаем персонал, который давно не видели
            self._clean_old_tracks(current_time)
            return violations
        
        # Получаем ID отслеживаемых объектов
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Обработка каждого отслеживаемого человека
        for i, track_id in enumerate(track_ids):
            current_tracks.add(track_id)
            
            # Получаем ограничивающую рамку
            bbox = boxes[i]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Получаем ключевые точки позы
            keypoints = None
            if results[0].keypoints is not None and i < len(results[0].keypoints):
                keypoints = results[0].keypoints[i].data[0].cpu().numpy()
            
            # Проверяем, находится ли человек в зоне лестницы
            person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            stair_zone = self._get_stair_zone_for_point(person_center)
            
            if stair_zone is not None:
                # Инициализируем трекер для этого человека, если еще нет
                if track_id not in self.tracked_persons:
                    self.tracked_persons[track_id] = {
                        'violation_timer': 0.0,
                        'last_seen': current_time,
                        'in_stair_zone': True,
                        'holding_handrail': False,
                        'stair_zone_id': stair_zone["id"],
                        'bbox': (x1, y1, x2, y2)
                    }
                else:
                    # Обновляем информацию о bbox и стаир зоне, но НЕ last_seen!
                    self.tracked_persons[track_id]['in_stair_zone'] = True
                    self.tracked_persons[track_id]['stair_zone_id'] = stair_zone["id"]
                    self.tracked_persons[track_id]['bbox'] = (x1, y1, x2, y2)
                
                # Проверяем, держится ли человек за поручень
                holding_handrail = False
                
                if keypoints is not None:
                    left_wrist = keypoints[self.LEFT_WRIST_IDX][:2] if self.LEFT_WRIST_IDX < len(keypoints) and keypoints[self.LEFT_WRIST_IDX][2] > 0.5 else None
                    right_wrist = keypoints[self.RIGHT_WRIST_IDX][:2] if self.RIGHT_WRIST_IDX < len(keypoints) and keypoints[self.RIGHT_WRIST_IDX][2] > 0.5 else None
                    
                    # Проверяем находится ли левая или правая кисть руки в зоне поручня
                    if left_wrist is not None and left_wrist[0] > 0 and left_wrist[1] > 0:
                        if self._is_point_in_handrail_zone(left_wrist, stair_zone["id"]):
                            holding_handrail = True
                            
                    if not holding_handrail and right_wrist is not None and right_wrist[0] > 0 and right_wrist[1] > 0:
                        if self._is_point_in_handrail_zone(right_wrist, stair_zone["id"]):
                            holding_handrail = True
                
                # Обновляем статус держания за поручень
                self.tracked_persons[track_id]['holding_handrail'] = holding_handrail
                
                # Обновляем таймер нарушения - СНАЧАЛА вычисляем delta
                time_delta = current_time - self.tracked_persons[track_id]['last_seen']
                
                # ЗАТЕМ обновляем last_seen
                self.tracked_persons[track_id]['last_seen'] = current_time
                
                if not holding_handrail:
                    self.tracked_persons[track_id]['violation_timer'] += time_delta
                    print(f"Track {track_id}: Not holding handrail! Timer increased to {self.tracked_persons[track_id]['violation_timer']:.2f}s")
                else:
                    # Сбрасываем таймер, если человек держится за поручень
                    self.tracked_persons[track_id]['violation_timer'] = 0.0
                    print(f"Track {track_id}: Holding handrail. Timer reset to 0.")
                
                # Проверяем, является ли это нарушением
                if self.tracked_persons[track_id]['violation_timer'] > self.violation_threshold:
                    violations.append({
                        'track_id': track_id,
                        'bbox': (x1, y1, x2, y2),
                        'violation_time': self.tracked_persons[track_id]['violation_timer'],
                        'stair_zone_id': stair_zone["id"]
                    })
                    print(f"Track {track_id}: VIOLATION DETECTED! Time: {self.tracked_persons[track_id]['violation_timer']:.2f}s")
            else:
                # Если персонал находится вне зоны лестницы
                if track_id in self.tracked_persons:
                    self.tracked_persons[track_id]['in_stair_zone'] = False
                    self.tracked_persons[track_id]['last_seen'] = current_time
                    # Сбрасываем таймер, если вне зоны лестницы
                    self.tracked_persons[track_id]['violation_timer'] = 0.0
        
        # Очищаем треки, которых нет в текущем кадре
        for track_id in list(self.tracked_persons.keys()):
            if track_id not in current_tracks:
                time_since_last_seen = current_time - self.tracked_persons[track_id]['last_seen']
                if time_since_last_seen > 1.0:  # Удаляем через 1 секунду неактивности
                    del self.tracked_persons[track_id]
        
        return violations
    
    def _clean_old_tracks(self, current_time: float, timeout: float = 1.0):
        """
        Удаление устаревших треков персонала
        
        Args:
            current_time: Текущее время
            timeout: Тайм-аут в секундах для удаления трека
        """
        to_delete = []
        for track_id, track_info in self.tracked_persons.items():
            if current_time - track_info['last_seen'] > timeout:
                to_delete.append(track_id)
        
        for track_id in to_delete:
            del self.tracked_persons[track_id]
    
    def _get_stair_zone_for_point(self, point: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """
        Определение зоны лестницы, в которой находится точка
        
        Args:
            point: Координаты точки (x, y)
            
        Returns:
            Словарь с информацией о зоне лестницы или None
        """
        for zone in self.stair_zones:
            if point_in_polygon(point, zone["polygon"]):
                return zone
        return None
    
    def _is_point_in_handrail_zone(self, point: Tuple[float, float], stair_id: int) -> bool:
        """
        Проверка, находится ли точка в зоне поручня для указанной лестницы
        
        Args:
            point: Координаты точки (x, y)
            stair_id: ID лестницы
            
        Returns:
            True, если точка находится в зоне поручня
        """
        for handrail in self.handrail_zones:
            if handrail["stair_id"] == stair_id:
                if point_in_polygon(point, handrail["polygon"]):
                    return True
        return False
    
    def get_violation_status(self, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Получение информации о статусе нарушения для конкретного трека
        
        Args:
            track_id: ID трека
            
        Returns:
            Словарь с информацией о статусе нарушения или None
        """
        if track_id in self.tracked_persons:
            info = self.tracked_persons[track_id]
            is_violation = info['violation_timer'] > self.violation_threshold
            
            return {
                'track_id': track_id,
                'in_stair_zone': info['in_stair_zone'],
                'holding_handrail': info['holding_handrail'],
                'violation_timer': info['violation_timer'],
                'is_violation': is_violation,
                'stair_zone_id': info.get('stair_zone_id'),
                'bbox': info.get('bbox')
            }
        return None
    
    def get_all_tracked_persons(self) -> List[Dict[str, Any]]:
        """
        Получение информации обо всех отслеживаемых людях
        
        Returns:
            Список словарей с информацией о персонале
        """
        result = []
        for track_id, info in self.tracked_persons.items():
            is_violation = info['violation_timer'] > self.violation_threshold
            
            result.append({
                'track_id': track_id,
                'in_stair_zone': info['in_stair_zone'],
                'holding_handrail': info['holding_handrail'],
                'violation_timer': info['violation_timer'],
                'is_violation': is_violation,
                'stair_zone_id': info.get('stair_zone_id'),
                'bbox': info.get('bbox')
            })
        return result