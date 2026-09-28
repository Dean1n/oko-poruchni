import cv2
import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Any, Optional
from collections import deque


class MotionDetector:
    """Улучшенный детектор движения с адаптивными параметрами"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация детектора движения
        
        Args:
            config: Конфигурация детектора движения
        """
        self.logger = logging.getLogger(__name__)
        
        # Основные параметры
        self.threshold = config.get("threshold", 25)
        self.min_area = config.get("min_area", 500)
        self.blur_size = config.get("blur_size", 21)
        self.dilate_iterations = config.get("dilate_iterations", 2)
        
        # Адаптивные параметры
        self.min_total_area_percentage = config.get("min_total_area_percentage", 0.005)
        self.learning_rate = config.get("learning_rate", 0.05)
        self.noise_filter_size = config.get("noise_filter_size", 3)
        self.history_size = config.get("history_size", 5)
        
        # Параметры для адаптации к освещению
        self.brightness_adaptation = config.get("brightness_adaptation", True)
        self.brightness_threshold = config.get("brightness_threshold", 30)
        self.brightness_history_size = config.get("brightness_history_size", 10)
        
        # Состояние детектора
        self.background = None
        self.roi_polygon = None
        self.roi_mask = None
        
        # История кадров для стабилизации
        self.frame_history = deque(maxlen=self.history_size)
        self.brightness_history = deque(maxlen=self.brightness_history_size)
        
        # Статистика и метрики
        self.stats = {
            'frames_processed': 0,
            'motion_detected_frames': 0,
            'avg_motion_area': 0.0,
            'avg_brightness': 0.0,
            'last_motion_time': 0.0,
            'background_updates': 0
        }
        
        # Временные переменные для оптимизации
        self.last_frame_time = 0.0
        self.frame_skip_counter = 0
        self.frame_skip_threshold = config.get("frame_skip", 1)  # Обрабатывать каждый N-й кадр
        
        self.logger.info("Детектор движения инициализирован")
    
    def set_roi(self, roi_points: List[Tuple[int, int]]):
        """
        Установка полигона зоны интереса с валидацией
        
        Args:
            roi_points: Список точек зоны интереса [(x1, y1), (x2, y2), ...]
        """
        try:
            if not roi_points or len(roi_points) < 3:
                self.logger.warning("ROI не установлена или содержит менее 3 точек")
                self.roi_polygon = None
                self.roi_mask = None
                return False
            
            self.roi_polygon = np.array([roi_points], dtype=np.int32)
            self.roi_mask = None  # Будет создана при первом кадре
            
            self.logger.info(f"Установлена ROI с {len(roi_points)} точками")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при установке ROI: {e}")
            return False
    
    def create_roi_mask(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Создание маски зоны интереса
        
        Args:
            frame_shape: Форма кадра (высота, ширина)
        
        Returns:
            Маска зоны интереса
        """
        if self.roi_polygon is None:
            return np.ones(frame_shape, dtype=np.uint8) * 255
        
        mask = np.zeros(frame_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_polygon], 255)
        return mask
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Обнаружение движения в кадре с улучшенной логикой
        
        Args:
            frame: Кадр видео
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (движение_обнаружено, визуализация)
        """
        try:
            # Валидация входного кадра
            if frame is None or frame.size == 0:
                self.logger.warning("Получен пустой кадр")
                return False, None
            
            current_time = time.time()
            
            # Пропуск кадров для оптимизации
            if self.frame_skip_threshold > 1:
                self.frame_skip_counter += 1
                if self.frame_skip_counter < self.frame_skip_threshold:
                    return False, frame.copy()
                self.frame_skip_counter = 0
            
            # Создание маски ROI если необходимо
            if self.roi_mask is None:
                self.roi_mask = self.create_roi_mask(frame.shape[:2])
            
            # Предобработка кадра
            processed_frame = self._preprocess_frame(frame)
            
            # Инициализация фона
            if self.background is None:
                self._initialize_background(processed_frame)
                return False, self._create_visualization(frame, False, [], 0.0)
            
            # Адаптация к изменениям освещения
            if self.brightness_adaptation:
                self._adapt_to_brightness_changes(processed_frame)
            
            # Детекция движения
            motion_detected, contours, total_area = self._detect_motion(processed_frame, frame.shape)
            
            # Обновление статистики
            self._update_statistics(motion_detected, total_area, processed_frame, current_time)
            
            # Обновление фона
            self._update_background(processed_frame, motion_detected)
            
            # Создание визуализации
            visualization = self._create_visualization(frame, motion_detected, contours, total_area)
            
            return motion_detected, visualization
            
        except Exception as e:
            self.logger.error(f"Ошибка при детекции движения: {e}")
            return False, frame.copy() if frame is not None else None
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Предобработка кадра"""
        try:
            # Преобразование в оттенки серого
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Применение медианного фильтра для удаления шума
            if self.noise_filter_size > 1:
                gray = cv2.medianBlur(gray, self.noise_filter_size)
            
            # Гауссово размытие
            if self.blur_size > 1:
                gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
            
            # Применение маски ROI
            gray_roi = cv2.bitwise_and(gray, gray, mask=self.roi_mask)
            
            # Добавление в историю кадров
            self.frame_history.append(gray_roi.copy())
            
            return gray_roi
            
        except Exception as e:
            self.logger.error(f"Ошибка предобработки кадра: {e}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def _initialize_background(self, processed_frame: np.ndarray):
        """Инициализация фона"""
        try:
            self.background = processed_frame.copy().astype(np.float32)
            self.logger.info("Фон инициализирован")
        except Exception as e:
            self.logger.error(f"Ошибка инициализации фона: {e}")
    
    def _adapt_to_brightness_changes(self, processed_frame: np.ndarray):
        """Адаптация к изменениям освещения"""
        try:
            # Вычисляем среднюю яркость
            avg_brightness = np.mean(processed_frame[self.roi_mask > 0])
            self.brightness_history.append(avg_brightness)
            
            # Проверяем резкие изменения яркости
            if len(self.brightness_history) >= 2:
                brightness_change = abs(avg_brightness - self.brightness_history[-2])
                
                if brightness_change > self.brightness_threshold:
                    # Резкое изменение освещения - ускоряем адаптацию фона
                    self.learning_rate = min(0.3, self.learning_rate * 3)
                    self.stats['background_updates'] += 1
                    self.logger.debug(f"Обнаружено изменение освещения: {brightness_change:.2f}")
                else:
                    # Восстанавливаем нормальную скорость обучения
                    self.learning_rate = max(0.01, self.learning_rate * 0.95)
                    
        except Exception as e:
            self.logger.error(f"Ошибка адаптации к освещению: {e}")
    
    def _detect_motion(self, processed_frame: np.ndarray, original_shape: Tuple[int, int, int]) -> Tuple[bool, List, float]:
        """Основная логика детекции движения"""
        try:
            # Проверка совместимости с фоном
            if self.background.shape != processed_frame.shape:
                self._initialize_background(processed_frame)
                return False, [], 0.0
            
            # Вычисление разности с фоном
            frame_delta = cv2.absdiff(self.background.astype(np.uint8), processed_frame)
            
            # Пороговая обработка с адаптивным порогом
            adaptive_threshold = self._calculate_adaptive_threshold(frame_delta)
            thresh = cv2.threshold(frame_delta, adaptive_threshold, 255, cv2.THRESH_BINARY)[1]
            
            # Морфологические операции для уменьшения шума
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.dilate(thresh, kernel, iterations=self.dilate_iterations)
            
            # Поиск контуров
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Фильтрация контуров и подсчет общей площади
            valid_contours = []
            total_motion_area = 0.0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.min_area:
                    # Дополнительная фильтрация по форме
                    if self._is_valid_motion_contour(contour):
                        valid_contours.append(contour)
                        total_motion_area += area
            
            # Определение наличия движения
            frame_area = original_shape[0] * original_shape[1]
            area_percentage = total_motion_area / frame_area
            
            motion_detected = (
                len(valid_contours) > 0 and
                total_motion_area >= self.min_area and
                area_percentage >= self.min_total_area_percentage
            )
            
            return motion_detected, valid_contours, total_motion_area
            
        except Exception as e:
            self.logger.error(f"Ошибка детекции движения: {e}")
            return False, [], 0.0
    
    def _calculate_adaptive_threshold(self, frame_delta: np.ndarray) -> int:
        """Вычисление адаптивного порога"""
        try:
            # Базовый порог
            base_threshold = self.threshold
            
            # Адаптация на основе статистики кадра
            mean_val = np.mean(frame_delta[self.roi_mask > 0])
            std_val = np.std(frame_delta[self.roi_mask > 0])
            
            # Адаптивный порог на основе стандартного отклонения
            adaptive_factor = min(2.0, max(0.5, std_val / 30.0))
            adaptive_threshold = int(base_threshold * adaptive_factor)
            
            # Ограничиваем диапазон
            adaptive_threshold = max(10, min(80, adaptive_threshold))
            
            return adaptive_threshold
            
        except Exception as e:
            self.logger.error(f"Ошибка вычисления адаптивного порога: {e}")
            return self.threshold
    
    def _is_valid_motion_contour(self, contour: np.ndarray) -> bool:
        """Проверка валидности контура движения"""
        try:
            # Проверка соотношения сторон
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Фильтруем слишком вытянутые объекты (могут быть артефактами)
            if aspect_ratio > 5.0 or aspect_ratio < 0.2:
                return False
            
            # Проверка заполненности контура
            contour_area = cv2.contourArea(contour)
            bounding_area = w * h
            
            if bounding_area > 0:
                fill_ratio = contour_area / bounding_area
                # Фильтруем слишком разреженные объекты
                if fill_ratio < 0.3:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации контура: {e}")
            return True  # В случае ошибки считаем контур валидным
    
    def _update_background(self, processed_frame: np.ndarray, motion_detected: bool):
        """Обновление модели фона"""
        try:
            if motion_detected:
                # При обнаружении движения замедляем обучение
                learning_rate = self.learning_rate * 0.5
            else:
                # При отсутствии движения ускоряем обучение
                learning_rate = self.learning_rate * 1.5
            
            # Ограничиваем скорость обучения
            learning_rate = max(0.001, min(0.5, learning_rate))
            
            # Обновляем фон
            cv2.accumulateWeighted(
                processed_frame.astype(np.float32), 
                self.background, 
                learning_rate
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления фона: {e}")
    
    def _update_statistics(self, motion_detected: bool, total_area: float, processed_frame: np.ndarray, current_time: float):
        """Обновление статистики"""
        try:
            self.stats['frames_processed'] += 1
            
            if motion_detected:
                self.stats['motion_detected_frames'] += 1
                self.stats['last_motion_time'] = current_time
            
            # Скользящее среднее для площади движения
            alpha = 0.1
            self.stats['avg_motion_area'] = (
                alpha * total_area + 
                (1 - alpha) * self.stats['avg_motion_area']
            )
            
            # Средняя яркость
            if len(self.brightness_history) > 0:
                self.stats['avg_brightness'] = np.mean(self.brightness_history)
                
        except Exception as e:
            self.logger.error(f"Ошибка обновления статистики: {e}")
    
    def _create_visualization(self, frame: np.ndarray, motion_detected: bool, contours: List, total_area: float) -> np.ndarray:
        """Создание визуализации результатов детекции"""
        try:
            visualization = frame.copy()
            
            # Отрисовка контуров движения
            if contours:
                for contour in contours:
                    # Ограничивающий прямоугольник
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Контур
                    cv2.drawContours(visualization, [contour], -1, (0, 255, 0), 1)
            
            # Отрисовка ROI
            if self.roi_polygon is not None:
                cv2.polylines(visualization, [self.roi_polygon], True, (255, 0, 0), 2)
                
                # Полупрозрачная заливка ROI
                overlay = visualization.copy()
                cv2.fillPoly(overlay, [self.roi_polygon], (255, 0, 0))
                cv2.addWeighted(overlay, 0.1, visualization, 0.9, 0, visualization)
            
            # Информационная панель
            self._draw_info_panel(visualization, motion_detected, len(contours), total_area)
            
            return visualization
            
        except Exception as e:
            self.logger.error(f"Ошибка создания визуализации: {e}")
            return frame.copy()
    
    def _draw_info_panel(self, frame: np.ndarray, motion_detected: bool, contour_count: int, total_area: float):
        """Отрисовка информационной панели"""
        try:
            # Фон для панели
            panel_height = 120
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Текстовая информация
            y_offset = 30
            line_height = 20
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (255, 255, 255)
            thickness = 1
            
            # Статус движения
            status = "ДВИЖЕНИЕ ОБНАРУЖЕНО" if motion_detected else "Движение не обнаружено"
            status_color = (0, 255, 0) if motion_detected else (128, 128, 128)
            cv2.putText(frame, f"Статус: {status}", (15, y_offset), font, font_scale, status_color, thickness)
            
            # Количество объектов
            y_offset += line_height
            cv2.putText(frame, f"Объектов движения: {contour_count}", (15, y_offset), font, font_scale, color, thickness)
            
            # Общая площадь движения
            y_offset += line_height
            area_percent = (total_area / (frame.shape[0] * frame.shape[1])) * 100
            cv2.putText(frame, f"Площадь движения: {area_percent:.2f}%", (15, y_offset), font, font_scale, color, thickness)
            
            # Средняя яркость
            y_offset += line_height
            cv2.putText(frame, f"Средняя яркость: {self.stats['avg_brightness']:.1f}", (15, y_offset), font, font_scale, color, thickness)
            
            # Частота детекции
            y_offset += line_height
            if self.stats['frames_processed'] > 0:
                detection_rate = (self.stats['motion_detected_frames'] / self.stats['frames_processed']) * 100
                cv2.putText(frame, f"Частота детекции: {detection_rate:.1f}%", (15, y_offset), font, font_scale, color, thickness)
            
        except Exception as e:
            self.logger.error(f"Ошибка отрисовки информационной панели: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики работы детектора"""
        detection_rate = 0.0
        if self.stats['frames_processed'] > 0:
            detection_rate = (self.stats['motion_detected_frames'] / self.stats['frames_processed']) * 100
        
        return {
            'frames_processed': self.stats['frames_processed'],
            'motion_detected_frames': self.stats['motion_detected_frames'],
            'detection_rate_percent': detection_rate,
            'avg_motion_area': self.stats['avg_motion_area'],
            'avg_brightness': self.stats['avg_brightness'],
            'last_motion_time': self.stats['last_motion_time'],
            'background_updates': self.stats['background_updates'],
            'roi_configured': self.roi_polygon is not None
        }
    
    def reset_statistics(self):
        """Сброс статистики"""
        self.stats = {
            'frames_processed': 0,
            'motion_detected_frames': 0,
            'avg_motion_area': 0.0,
            'avg_brightness': 0.0,
            'last_motion_time': 0.0,
            'background_updates': 0
        }
        self.logger.info("Статистика детектора движения сброшена")
    
    def reset_background(self):
        """Сброс модели фона"""
        self.background = None
        self.frame_history.clear()
        self.brightness_history.clear()
        self.logger.info("Модель фона сброшена")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Обновление конфигурации на лету"""
        try:
            # Обновляем параметры
            self.threshold = new_config.get("threshold", self.threshold)
            self.min_area = new_config.get("min_area", self.min_area)
            self.blur_size = new_config.get("blur_size", self.blur_size)
            self.dilate_iterations = new_config.get("dilate_iterations", self.dilate_iterations)
            
            self.min_total_area_percentage = new_config.get("min_total_area_percentage", self.min_total_area_percentage)
            self.learning_rate = new_config.get("learning_rate", self.learning_rate)
            self.noise_filter_size = new_config.get("noise_filter_size", self.noise_filter_size)
            
            self.brightness_adaptation = new_config.get("brightness_adaptation", self.brightness_adaptation)
            self.brightness_threshold = new_config.get("brightness_threshold", self.brightness_threshold)
            
            self.frame_skip_threshold = new_config.get("frame_skip", self.frame_skip_threshold)
            
            self.logger.info("Конфигурация детектора движения обновлена")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления конфигурации: {e}")
            return False
    
    def cleanup(self):
        """Очистка ресурсов детектора"""
        try:
            self.background = None
            self.roi_mask = None
            self.frame_history.clear()
            self.brightness_history.clear()
            self.reset_statistics()
            
            self.logger.info("Ресурсы детектора движения очищены")
            
        except Exception as e:
            self.logger.error(f"Ошибка при очистке ресурсов: {e}")
