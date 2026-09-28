import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QMessageBox, QFileDialog, QListWidget, QGroupBox, QScrollArea, QWidget,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QListWidgetItem,
    QSlider, QSpinBox, QCheckBox, QTabWidget, QFrame, QTextEdit, QProgressBar,
    QFormLayout  # Добавлен недостающий импорт
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint, QRect, QThread
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QMouseEvent, QFont


def get_frame_at_position(video_path: str, position: int = 0, position_type: str = "frame") -> Optional[np.ndarray]:
    """
    Получение кадра из видео на указанной позиции с улучшенной обработкой ошибок
    
    Args:
        video_path: Путь к видеофайлу
        position: Позиция (номер кадра или время в секундах)
        position_type: Тип позиции ("frame" или "time")
        
    Returns:
        Кадр в виде numpy массива или None в случае ошибки
    """
    if not os.path.exists(video_path):
        return None
    
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        if position_type == "time":
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                return None
            position = int(position * fps)
        
        # Установка позиции с проверкой границ
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        position = max(0, min(position, frame_count - 1))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
            
    except Exception as e:
        logging.error(f"Ошибка при получении кадра: {e}")
    finally:
        if cap is not None:
            cap.release()
    
    return None


class FrameLoaderThread(QThread):
    """Поток для загрузки кадров из видео"""
    
    frame_loaded = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_path: str, position: int = 0):
        super().__init__()
        self.video_path = video_path
        self.position = position
    
    def run(self):
        """Загрузка кадра в отдельном потоке"""
        try:
            frame = get_frame_at_position(self.video_path, self.position)
            if frame is not None:
                self.frame_loaded.emit(frame)
            else:
                self.error_occurred.emit("Не удалось загрузить кадр")
        except Exception as e:
            self.error_occurred.emit(f"Ошибка загрузки кадра: {str(e)}")


class ZoneEditorWidget(QWidget):
    """Улучшенный виджет для редактирования зон на изображении"""
    
    zone_updated = pyqtSignal(str, list)
    point_added = pyqtSignal(int, int)
    point_removed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        # Изображение и отображение
        self.original_image = None
        self.displayed_image = None
        self.scale_factor = 1.0
        self.image_offset = QPoint(0, 0)
        
        # Текущие точки и режимы
        self.current_points = []
        self.editing_mode = "roi"
        self.stair_id = 1
        self.handrail_id = 1
        
        # Зоны из конфига
        self.roi_zone = []
        self.stair_zones = []
        
        # Состояние редактирования
        self.current_stair_idx = -1
        self.current_handrail_idx = -1
        self.is_drawing = False
        self.selected_point_idx = -1
        
        # Настройки отображения
        self.show_roi = True
        self.show_stairs = True
        self.show_handrails = True
        self.show_grid = False  # Добавлено
        self.point_radius = 6
        self.line_thickness = 2
        
        # Настройки цветов
        self.colors = {
            "roi": (255, 0, 0),         # Красный
            "stair": (128, 128, 128),   # Серый
            "handrail": (0, 255, 0),    # Зеленый
            "current": (0, 0, 255),     # Синий для текущей зоны
            "selected": (255, 255, 0),  # Желтый для выбранной точки
            "grid": (200, 200, 200)     # Серый для сетки
        }
        
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)
        
        # Настройка контекстного меню
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
    
    def set_image(self, image: np.ndarray):
        """Установка изображения для редактирования с проверкой"""
        try:
            if image is not None and image.size > 0:
                self.original_image = image.copy()
                self.scale_factor = 1.0
                self.image_offset = QPoint(0, 0)
                self.update_display()
                self.logger.info(f"Изображение установлено: {image.shape}")
            else:
                self.logger.warning("Попытка установить пустое изображение")
        except Exception as e:
            self.logger.error(f"Ошибка установки изображения: {e}")
    
    def set_zones(self, roi_zone: List, stair_zones: List):
        """Установка существующих зон с валидацией"""
        try:
            self.roi_zone = roi_zone if roi_zone else []
            self.stair_zones = stair_zones if stair_zones else []
            
            # Валидация зон
            self._validate_zones()
            
            self.update_display()
            self.logger.info(f"Зоны установлены: ROI - {len(self.roi_zone)} точек, "
                           f"Лестниц - {len(self.stair_zones)}")
                           
        except Exception as e:
            self.logger.error(f"Ошибка установки зон: {e}")
    
    def _validate_zones(self):
        """Валидация зон на корректность"""
        # Проверка ROI
        if self.roi_zone:
            if len(self.roi_zone) < 3:
                self.logger.warning("ROI зона содержит менее 3 точек")
                self.roi_zone = []
        
        # Проверка зон лестниц
        valid_stairs = []
        for stair in self.stair_zones:
            if self._validate_stair_zone(stair):
                valid_stairs.append(stair)
            else:
                self.logger.warning(f"Некорректная зона лестницы: {stair.get('id', 'неизвестно')}")
        
        self.stair_zones = valid_stairs
    
    def _validate_stair_zone(self, stair: Dict[str, Any]) -> bool:
        """Валидация зоны лестницы"""
        if not isinstance(stair, dict):
            return False
        
        if "id" not in stair or "coordinates" not in stair:
            return False
        
        coords = stair["coordinates"]
        if not isinstance(coords, list) or len(coords) < 3:
            return False
        
        # Проверка поручней
        if "handrails" in stair:
            handrails = stair["handrails"]
            if isinstance(handrails, list):
                valid_handrails = []
                for handrail in handrails:
                    if self._validate_handrail_zone(handrail):
                        valid_handrails.append(handrail)
                stair["handrails"] = valid_handrails
        
        return True
    
    def _validate_handrail_zone(self, handrail: Dict[str, Any]) -> bool:
        """Валидация зоны поручня"""
        if not isinstance(handrail, dict):
            return False
        
        if "id" not in handrail or "coordinates" not in handrail:
            return False
        
        coords = handrail["coordinates"]
        return isinstance(coords, list) and len(coords) >= 3
    
    def set_editing_mode(self, mode: str, stair_id: int = 1, handrail_id: int = 1):
        """Установка режима редактирования с безопасностью"""
        try:
            if mode == self.editing_mode and stair_id == self.stair_id and handrail_id == self.handrail_id:
                return
            
            self.editing_mode = mode
            self.stair_id = stair_id
            self.handrail_id = handrail_id
            
            # Сброс состояния
            self.current_points = []
            self.current_stair_idx = -1
            self.current_handrail_idx = -1
            self.selected_point_idx = -1
            
            # Загрузка существующих точек
            self._load_existing_points()
            
            self.update_display()
            self.logger.info(f"Режим изменен: {mode}, лестница: {stair_id}, поручень: {handrail_id}")
            
        except Exception as e:
            self.logger.error(f"Ошибка изменения режима: {e}")
    
    def _load_existing_points(self):
        """Загрузка существующих точек для текущего режима"""
        try:
            if self.editing_mode == "roi" and self.roi_zone:
                self.current_points = [list(point) for point in self.roi_zone]
                
            elif self.editing_mode == "stair":
                for idx, stair in enumerate(self.stair_zones):
                    if stair.get("id") == self.stair_id:
                        self.current_stair_idx = idx
                        if "coordinates" in stair:
                            self.current_points = [list(point) for point in stair["coordinates"]]
                        break
                        
            elif self.editing_mode == "handrail":
                for stair_idx, stair in enumerate(self.stair_zones):
                    if stair.get("id") == self.stair_id:
                        self.current_stair_idx = stair_idx
                        handrails = stair.get("handrails", [])
                        for handrail_idx, handrail in enumerate(handrails):
                            if handrail.get("id") == self.handrail_id:
                                self.current_handrail_idx = handrail_idx
                                if "coordinates" in handrail:
                                    self.current_points = [list(point) for point in handrail["coordinates"]]
                                break
                        break
                        
        except Exception as e:
            self.logger.error(f"Ошибка загрузки точек: {e}")
            self.current_points = []
    
    def update_display(self):
        """Обновление отображения с улучшенной графикой"""
        if self.original_image is None:
            return
        
        try:
            # Создание отображаемого изображения
            display_img = self.original_image.copy()
            
            # Отрисовка сетки если нужно
            if hasattr(self, 'show_grid') and self.show_grid:
                self._draw_grid(display_img)
            
            # Отрисовка существующих зон
            self._draw_existing_zones(display_img)
            
            # Отрисовка текущей редактируемой зоны
            self._draw_current_zone(display_img)
            
            # Добавление информации
            #self._draw_info_panel(display_img)
            
            self.displayed_image = display_img
            self.update()
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления отображения: {e}")
    
    def _draw_grid(self, image: np.ndarray):
        """Отрисовка вспомогательной сетки"""
        h, w = image.shape[:2]
        grid_step = 50
        color = self.colors["grid"]
        
        # Вертикальные линии
        for x in range(0, w, grid_step):
            cv2.line(image, (x, 0), (x, h), color, 1)
        
        # Горизонтальные линии
        for y in range(0, h, grid_step):
            cv2.line(image, (0, y), (w, y), color, 1)
    
    def _draw_existing_zones(self, image: np.ndarray):
        """Отрисовка существующих зон"""
        try:
            # ROI зона
            if self.show_roi and self.roi_zone and len(self.roi_zone) >= 3:
                pts = np.array([self.roi_zone], dtype=np.int32)
                cv2.polylines(image, pts, True, self.colors["roi"], self.line_thickness)
                
                # Полупрозрачная заливка
                overlay = image.copy()
                cv2.fillPoly(overlay, pts, self.colors["roi"])
                cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
                
                # Подпись
                if len(self.roi_zone) > 0:
                    center = self._calculate_polygon_center(self.roi_zone)
                    cv2.putText(image, "ROI", center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors["roi"], 2)
            
            # Зоны лестниц
            if self.show_stairs:
                for stair in self.stair_zones:
                    if "coordinates" in stair and len(stair["coordinates"]) >= 3:
                        pts = np.array([stair["coordinates"]], dtype=np.int32)
                        cv2.polylines(image, pts, True, self.colors["stair"], self.line_thickness)
                        
                        # Полупрозрачная заливка
                        overlay = image.copy()
                        cv2.fillPoly(overlay, pts, self.colors["stair"])
                        cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)
                        
                        # Подпись
                        center = self._calculate_polygon_center(stair["coordinates"])
                        stair_id = stair.get("id", "?")
                        cv2.putText(image, f"Stair {stair_id}", center, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["stair"], 2)
            
            # Поручни
            if self.show_handrails:
                for stair in self.stair_zones:
                    if "handrails" in stair:
                        for handrail in stair["handrails"]:
                            if "coordinates" in handrail and len(handrail["coordinates"]) >= 3:
                                pts = np.array([handrail["coordinates"]], dtype=np.int32)
                                cv2.polylines(image, pts, True, self.colors["handrail"], self.line_thickness)
                                
                                # Полупрозрачная заливка
                                overlay = image.copy()
                                cv2.fillPoly(overlay, pts, self.colors["handrail"])
                                cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
                                
                                # Подпись
                                center = self._calculate_polygon_center(handrail["coordinates"])
                                handrail_id = handrail.get("id", "?")
                                cv2.putText(image, f"H{handrail_id}", center, 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["handrail"], 1)
                                           
        except Exception as e:
            self.logger.error(f"Ошибка отрисовки существующих зон: {e}")
    
    def _draw_current_zone(self, image: np.ndarray):
        """Отрисовка текущей редактируемой зоны"""
        if not self.current_points:
            return
        
        try:
            color = self.colors["current"]
            
            # Линии между точками
            for i in range(len(self.current_points)):
                pt1 = tuple(map(int, self.current_points[i]))
                pt2 = tuple(map(int, self.current_points[(i + 1) % len(self.current_points)]))
                cv2.line(image, pt1, pt2, color, self.line_thickness + 1)
            
            # Точки
            for i, point in enumerate(self.current_points):
                pt = tuple(map(int, point))
                point_color = self.colors["selected"] if i == self.selected_point_idx else color
                cv2.circle(image, pt, self.point_radius, point_color, -1)
                cv2.circle(image, pt, self.point_radius + 2, (255, 255, 255), 1)
                
                # Номер точки
                cv2.putText(image, str(i + 1), (pt[0] + 8, pt[1] - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                           
        except Exception as e:
            self.logger.error(f"Ошибка отрисовки текущей зоны: {e}")
    
    def _draw_info_panel(self, image: np.ndarray):
        """Отрисовка информационной панели"""
        try:
            # Фон для панели
            panel_height = 100
            overlay = image.copy()
            cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Информация о режиме
            mode_text = f"Режим: {self.editing_mode}"
            if self.editing_mode == "stair":
                mode_text += f", Лестница: {self.stair_id}"
            elif self.editing_mode == "handrail":
                mode_text += f", Лестница: {self.stair_id}, Поручень: {self.handrail_id}"
            
            cv2.putText(image, mode_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Количество точек
            points_text = f"Точек: {len(self.current_points)}"
            cv2.putText(image, points_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Подсказки
            help_text = "ЛКМ: добавить точку, ПКМ: удалить последнюю, Ctrl+ЛКМ: выбрать точку"
            cv2.putText(image, help_text, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
        except Exception as e:
            self.logger.error(f"Ошибка отрисовки информационной панели: {e}")
    
    def _calculate_polygon_center(self, points: List) -> Tuple[int, int]:
        """Вычисление центра полигона"""
        try:
            if not points:
                return (0, 0)
            
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            center_x = int(sum(x_coords) / len(x_coords))
            center_y = int(sum(y_coords) / len(y_coords))
            
            return (center_x, center_y)
            
        except Exception as e:
            self.logger.error(f"Ошибка вычисления центра полигона: {e}")
            return (0, 0)
    
    def save_current_zone(self) -> bool:
        """Сохранение текущей зоны с валидацией"""
        try:
            if not self.current_points or len(self.current_points) < 3:
                QMessageBox.warning(self, "Предупреждение", 
                                  "Необходимо задать минимум 3 точки для создания зоны")
                return False
            
            # Создание копии точек
            points_copy = [list(point) for point in self.current_points]
            
            if self.editing_mode == "roi":
                self.roi_zone = points_copy
                self.zone_updated.emit("roi", self.roi_zone)
                
            elif self.editing_mode == "stair":
                self._save_stair_zone(points_copy)
                
            elif self.editing_mode == "handrail":
                if not self._save_handrail_zone(points_copy):
                    return False
            
            self.update_display()
            self.logger.info(f"Зона сохранена: {self.editing_mode}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения зоны: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения зоны: {str(e)}")
            return False
    
    def _save_stair_zone(self, points: List):
        """Сохранение зоны лестницы"""
        found_idx = -1
        for idx, stair in enumerate(self.stair_zones):
            if stair.get("id") == self.stair_id:
                found_idx = idx
                break
        
        if found_idx >= 0:
            self.stair_zones[found_idx]["coordinates"] = points
        else:
            self.stair_zones.append({
                "id": self.stair_id,
                "coordinates": points,
                "handrails": []
            })
        
        self.zone_updated.emit("stair", self.stair_zones)
    
    def _save_handrail_zone(self, points: List) -> bool:
        """Сохранение зоны поручня"""
        stair_idx = -1
        for idx, stair in enumerate(self.stair_zones):
            if stair.get("id") == self.stair_id:
                stair_idx = idx
                break
        
        if stair_idx < 0:
            QMessageBox.warning(self, "Предупреждение", 
                              f"Лестница с ID {self.stair_id} не найдена. Сначала создайте лестницу.")
            return False
        
        handrail_idx = -1
        handrails = self.stair_zones[stair_idx].get("handrails", [])
        for idx, handrail in enumerate(handrails):
            if handrail.get("id") == self.handrail_id:
                handrail_idx = idx
                break
        
        if handrail_idx >= 0:
            self.stair_zones[stair_idx]["handrails"][handrail_idx]["coordinates"] = points
        else:
            if "handrails" not in self.stair_zones[stair_idx]:
                self.stair_zones[stair_idx]["handrails"] = []
            
            self.stair_zones[stair_idx]["handrails"].append({
                "id": self.handrail_id,
                "coordinates": points
            })
        
        self.zone_updated.emit("handrail", self.stair_zones)
        return True
    
    def clear_current_points(self):
        """Очистка текущих точек"""
        self.current_points = []
        self.selected_point_idx = -1
        self.update_display()
    
    def delete_selected_zone(self):
        """Удаление выбранной зоны"""
        try:
            if self.editing_mode == "roi":
                self.roi_zone = []
                self.zone_updated.emit("roi", self.roi_zone)
                
            elif self.editing_mode == "stair":
                self.stair_zones = [s for s in self.stair_zones if s.get("id") != self.stair_id]
                self.zone_updated.emit("stair", self.stair_zones)
                
            elif self.editing_mode == "handrail":
                for stair in self.stair_zones:
                    if stair.get("id") == self.stair_id and "handrails" in stair:
                        stair["handrails"] = [h for h in stair["handrails"] if h.get("id") != self.handrail_id]
                self.zone_updated.emit("handrail", self.stair_zones)
            
            self.current_points = []
            self.update_display()
            
        except Exception as e:
            self.logger.error(f"Ошибка удаления зоны: {e}")
    
    def mousePressEvent(self, event):
        """Обработка нажатия мыши с расширенным функционалом"""
        if self.original_image is None:
            return
        
        try:
            x, y = self._get_image_coordinates(event.pos())
            if x is None or y is None:
                return
            
            if event.button() == Qt.LeftButton:
                # Проверка модификаторов
                if event.modifiers() & Qt.ControlModifier:
                    # Ctrl + ЛКМ - выбор точки для редактирования
                    self._select_nearest_point(x, y)
                else:
                    # ЛКМ - добавление новой точки
                    self.current_points.append([x, y])
                    self.point_added.emit(x, y)
                    
                self.update_display()
                
            elif event.button() == Qt.RightButton:
                # ПКМ - удаление последней точки или контекстное меню
                if self.current_points:
                    self.current_points.pop()
                    self.point_removed.emit()
                    self.update_display()
                else:
                    self.show_context_menu(event.pos())
                    
        except Exception as e:
            self.logger.error(f"Ошибка обработки нажатия мыши: {e}")
    
    def _select_nearest_point(self, x: int, y: int):
        """Выбор ближайшей точки для редактирования"""
        if not self.current_points:
            return
        
        min_distance = float('inf')
        nearest_idx = -1
        
        for i, point in enumerate(self.current_points):
            distance = ((point[0] - x) ** 2 + (point[1] - y) ** 2) ** 0.5
            if distance < min_distance and distance <= self.point_radius * 2:
                min_distance = distance
                nearest_idx = i
        
        self.selected_point_idx = nearest_idx if nearest_idx >= 0 else -1
    
    def mouseMoveEvent(self, event):
        """Обработка движения мыши для перетаскивания точек"""
        if self.original_image is None or self.selected_point_idx < 0:
            return
        
        try:
            if event.buttons() & Qt.LeftButton:
                x, y = self._get_image_coordinates(event.pos())
                if x is not None and y is not None:
                    self.current_points[self.selected_point_idx] = [x, y]
                    self.update_display()
                    
        except Exception as e:
            self.logger.error(f"Ошибка перетаскивания точки: {e}")
    
    def mouseReleaseEvent(self, event):
        """Обработка отпускания мыши"""
        if event.button() == Qt.LeftButton:
            self.selected_point_idx = -1
    
    def show_context_menu(self, position):
        """Показ контекстного меню"""
        from PyQt5.QtWidgets import QMenu, QAction
        
        menu = QMenu(self)
        
        # Действия для текущей зоны
        save_action = QAction("Сохранить зону", self)
        save_action.triggered.connect(self.save_current_zone)
        menu.addAction(save_action)
        
        clear_action = QAction("Очистить точки", self)
        clear_action.triggered.connect(self.clear_current_points)
        menu.addAction(clear_action)
        
        delete_action = QAction("Удалить зону", self)
        delete_action.triggered.connect(self.delete_selected_zone)
        menu.addAction(delete_action)
        
        menu.addSeparator()
        
        # Действия для отображения
        toggle_roi_action = QAction("Показать ROI", self)
        toggle_roi_action.setCheckable(True)
        toggle_roi_action.setChecked(self.show_roi)
        toggle_roi_action.triggered.connect(self.toggle_roi_visibility)
        menu.addAction(toggle_roi_action)
        
        toggle_stairs_action = QAction("Показать лестницы", self)
        toggle_stairs_action.setCheckable(True)
        toggle_stairs_action.setChecked(self.show_stairs)
        toggle_stairs_action.triggered.connect(self.toggle_stairs_visibility)
        menu.addAction(toggle_stairs_action)
        
        toggle_handrails_action = QAction("Показать поручни", self)
        toggle_handrails_action.setCheckable(True)
        toggle_handrails_action.setChecked(self.show_handrails)
        toggle_handrails_action.triggered.connect(self.toggle_handrails_visibility)
        menu.addAction(toggle_handrails_action)
        
        menu.exec_(self.mapToGlobal(position))
    
    def toggle_roi_visibility(self):
        """Переключение видимости ROI"""
        self.show_roi = not self.show_roi
        self.update_display()
    
    def toggle_stairs_visibility(self):
        """Переключение видимости лестниц"""
        self.show_stairs = not self.show_stairs
        self.update_display()
    
    def toggle_handrails_visibility(self):
        """Переключение видимости поручней"""
        self.show_handrails = not self.show_handrails
        self.update_display()
    
    def paintEvent(self, event):
        """Отрисовка виджета с масштабированием"""
        super().paintEvent(event)
        
        if self.displayed_image is not None:
            try:
                height, width, channel = self.displayed_image.shape
                bytes_per_line = 3 * width
                q_img = QImage(self.displayed_image.data, width, height, 
                              bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                
                pixmap = QPixmap.fromImage(q_img)
                
                # Масштабирование с сохранением пропорций
                scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                # Центрирование изображения
                x = (self.width() - scaled_pixmap.width()) // 2
                y = (self.height() - scaled_pixmap.height()) // 2
                
                painter = QPainter(self)
                painter.drawPixmap(x, y, scaled_pixmap)
                painter.end()
                
                # Обновление параметров масштабирования
                self.scale_factor = scaled_pixmap.width() / width
                self.image_offset = QPoint(x, y)
                
            except Exception as e:
                self.logger.error(f"Ошибка отрисовки: {e}")
    
    def _get_image_coordinates(self, widget_pos: QPoint) -> Tuple[Optional[int], Optional[int]]:
        """Преобразование координат виджета в координаты изображения"""
        if self.original_image is None or self.scale_factor == 0:
            return None, None
        
        try:
            # Относительные координаты в масштабированном изображении
            rel_x = widget_pos.x() - self.image_offset.x()
            rel_y = widget_pos.y() - self.image_offset.y()
            
            # Проверка границ
            if rel_x < 0 or rel_y < 0:
                return None, None
            
            # Преобразование в координаты оригинального изображения
            orig_x = int(rel_x / self.scale_factor)
            orig_y = int(rel_y / self.scale_factor)
            
            # Проверка границ оригинального изображения
            img_height, img_width = self.original_image.shape[:2]
            if orig_x >= img_width or orig_y >= img_height:
                return None, None
            
            return orig_x, orig_y
            
        except Exception as e:
            self.logger.error(f"Ошибка преобразования координат: {e}")
            return None, None


class ZoneEditorDialog(QDialog):
    """Улучшенный диалог редактора зон с расширенным функционалом"""
    
    def __init__(self, parent=None, camera_config=None):
        super().__init__(parent)
        self.setWindowTitle("Редактор зон камеры")
        self.setMinimumSize(1200, 800)
        
        self.camera_config = camera_config or {}
        self.roi_zone = self.camera_config.get("roi_zone", [])
        self.stair_zones = self.camera_config.get("stair_zones", [])
        
        self.frame = None
        self.video_path = ""
        self.frame_loader_thread = None
        
        self.logger = logging.getLogger(__name__)
        
        self.init_ui()
        self.load_first_frame()
    
    def init_ui(self):
        """Инициализация улучшенного интерфейса"""
        main_layout = QVBoxLayout()
        
        # Панель инструментов
        toolbar = self._create_toolbar()
        main_layout.addWidget(toolbar)
        
        # Основной контент в виде вкладок
        content_tabs = QTabWidget()
        
        # Вкладка редактирования
        edit_tab = self._create_edit_tab()
        content_tabs.addTab(edit_tab, "Редактирование зон")
        
        # Вкладка управления зонами
        manage_tab = self._create_management_tab()
        content_tabs.addTab(manage_tab, "Управление зонами")
        
        # Вкладка настроек
        settings_tab = self._create_settings_tab()
        content_tabs.addTab(settings_tab, "Настройки")
        
        main_layout.addWidget(content_tabs)
        
        # Нижняя панель с кнопками
        bottom_panel = self._create_bottom_panel()
        main_layout.addWidget(bottom_panel)
        
        self.setLayout(main_layout)
        
        # Инициализация после создания интерфейса
        QTimer.singleShot(100, self._post_init)
    
    def _create_toolbar(self) -> QWidget:
        """Создание панели инструментов"""
        toolbar = QFrame()
        toolbar.setFrameStyle(QFrame.Box)
        layout = QHBoxLayout()
        
        # Выбор режима
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Зона интереса (ROI)", "roi")
        self.mode_combo.addItem("Лестница", "stair")
        self.mode_combo.addItem("Поручень", "handrail")
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        
        # Выбор ID лестницы
        self.stair_id_combo = QComboBox()
        self.stair_id_combo.setEnabled(False)
        self.stair_id_combo.currentIndexChanged.connect(self.on_stair_id_changed)
        
        # Выбор ID поручня
        self.handrail_id_combo = QComboBox()
        self.handrail_id_combo.setEnabled(False)
        self.handrail_id_combo.currentIndexChanged.connect(self.on_handrail_id_changed)
        
        # Кнопки действий
        self.save_zone_btn = QPushButton("Сохранить зону")
        self.save_zone_btn.clicked.connect(self.save_current_zone)
        
        self.clear_points_btn = QPushButton("Очистить точки")
        self.clear_points_btn.clicked.connect(self.clear_current_points)
        
        self.load_frame_btn = QPushButton("Загрузить кадр")
        self.load_frame_btn.clicked.connect(self.load_frame_dialog)
        
        # Добавление виджетов
        layout.addWidget(QLabel("Режим:"))
        layout.addWidget(self.mode_combo)
        layout.addWidget(QLabel("Лестница:"))
        layout.addWidget(self.stair_id_combo)
        layout.addWidget(QLabel("Поручень:"))
        layout.addWidget(self.handrail_id_combo)
        layout.addStretch()
        layout.addWidget(self.save_zone_btn)
        layout.addWidget(self.clear_points_btn)
        layout.addWidget(self.load_frame_btn)
        
        toolbar.setLayout(layout)
        return toolbar
    
    def _create_edit_tab(self) -> QWidget:
        """Создание вкладки редактирования"""
        tab = QWidget()
        layout = QHBoxLayout()
        
        # Редактор зон
        self.zone_editor = ZoneEditorWidget()
        self.zone_editor.zone_updated.connect(self.on_zone_updated)
        self.zone_editor.point_added.connect(self.on_point_added)
        self.zone_editor.point_removed.connect(self.on_point_removed)
        
        layout.addWidget(self.zone_editor, 3)
        
        # Боковая панель с информацией
        side_panel = self._create_side_panel()
        layout.addWidget(side_panel, 1)
        
        tab.setLayout(layout)
        return tab
    
    def _create_side_panel(self) -> QWidget:
        """Создание боковой панели"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Информация о текущей зоне
        current_zone_group = QGroupBox("Текущая зона")
        current_zone_layout = QVBoxLayout()
        
        self.current_zone_info = QLabel("Выберите режим редактирования")
        self.points_count_label = QLabel("Точек: 0")
        
        current_zone_layout.addWidget(self.current_zone_info)
        current_zone_layout.addWidget(self.points_count_label)
        current_zone_group.setLayout(current_zone_layout)
        
        # Список точек
        points_group = QGroupBox("Точки")
        points_layout = QVBoxLayout()
        
        self.points_table = QTableWidget()
        self.points_table.setColumnCount(3)
        self.points_table.setHorizontalHeaderLabels(["№", "X", "Y"])
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        points_layout.addWidget(self.points_table)
        points_group.setLayout(points_layout)
        
        # Справочная информация
        help_group = QGroupBox("Справка")
        help_layout = QVBoxLayout()
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setMaximumHeight(150)
        help_text.setHtml("""
        <b>Управление:</b><br>
        • ЛКМ - добавить точку<br>
        • ПКМ - удалить последнюю точку<br>
        • Ctrl+ЛКМ - выбрать точку для редактирования<br>
        • ПКМ на пустом месте - контекстное меню<br><br>
        
        <b>Минимальные требования:</b><br>
        • ROI, лестницы, поручни: минимум 3 точки<br>
        • Поручни создаются только для существующих лестниц
        """)
        
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        
        layout.addWidget(current_zone_group)
        layout.addWidget(points_group)
        layout.addWidget(help_group)
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def _create_management_tab(self) -> QWidget:
        """Создание вкладки управления зонами"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Список всех зон
        zones_group = QGroupBox("Все зоны")
        zones_layout = QVBoxLayout()
        
        self.zones_list = QListWidget()
        self.zones_list.itemClicked.connect(self.on_zone_selected)
        
        zones_buttons_layout = QHBoxLayout()
        edit_zone_btn = QPushButton("Редактировать")
        edit_zone_btn.clicked.connect(self.edit_selected_zone)
        
        delete_zone_btn = QPushButton("Удалить")
        delete_zone_btn.clicked.connect(self.delete_selected_zone)
        
        zones_buttons_layout.addWidget(edit_zone_btn)
        zones_buttons_layout.addWidget(delete_zone_btn)
        zones_buttons_layout.addStretch()
        
        zones_layout.addWidget(self.zones_list)
        zones_layout.addLayout(zones_buttons_layout)
        zones_group.setLayout(zones_layout)
        
        # Валидация зон
        validation_group = QGroupBox("Валидация")
        validation_layout = QVBoxLayout()
        
        validate_btn = QPushButton("Проверить все зоны")
        validate_btn.clicked.connect(self.validate_all_zones)
        
        self.validation_results = QTextEdit()
        self.validation_results.setReadOnly(True)
        self.validation_results.setMaximumHeight(200)
        
        validation_layout.addWidget(validate_btn)
        validation_layout.addWidget(self.validation_results)
        validation_group.setLayout(validation_layout)
        
        layout.addWidget(zones_group)
        layout.addWidget(validation_group)
        
        tab.setLayout(layout)
        return tab
    
    def _create_settings_tab(self) -> QWidget:
        """Создание вкладки настроек"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Настройки отображения
        display_group = QGroupBox("Настройки отображения")
        display_layout = QFormLayout()
        
        self.show_roi_cb = QCheckBox()
        self.show_roi_cb.setChecked(True)
        self.show_roi_cb.toggled.connect(self.update_zone_editor_settings)
        
        self.show_stairs_cb = QCheckBox()
        self.show_stairs_cb.setChecked(True)
        self.show_stairs_cb.toggled.connect(self.update_zone_editor_settings)
        
        self.show_handrails_cb = QCheckBox()
        self.show_handrails_cb.setChecked(True)
        self.show_handrails_cb.toggled.connect(self.update_zone_editor_settings)
        
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setRange(3, 12)
        self.point_size_slider.setValue(6)
        self.point_size_slider.valueChanged.connect(self.update_zone_editor_settings)
        
        self.line_thickness_slider = QSlider(Qt.Horizontal)
        self.line_thickness_slider.setRange(1, 5)
        self.line_thickness_slider.setValue(2)
        self.line_thickness_slider.valueChanged.connect(self.update_zone_editor_settings)
        
        display_layout.addRow("Показать ROI:", self.show_roi_cb)
        display_layout.addRow("Показать лестницы:", self.show_stairs_cb)
        display_layout.addRow("Показать поручни:", self.show_handrails_cb)
        display_layout.addRow("Размер точек:", self.point_size_slider)
        display_layout.addRow("Толщина линий:", self.line_thickness_slider)
        
        display_group.setLayout(display_layout)
        
        # Настройки видео
        video_group = QGroupBox("Настройки видео")
        video_layout = QVBoxLayout()
        
        video_info_label = QLabel(f"Папка с видео: {self.camera_config.get('video_folder_path', 'Не указана')}")
        video_layout.addWidget(video_info_label)
        
        video_group.setLayout(video_layout)
        
        layout.addWidget(display_group)
        layout.addWidget(video_group)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def _create_bottom_panel(self) -> QWidget:
        """Создание нижней панели"""
        panel = QWidget()
        layout = QHBoxLayout()
        
        # Прогресс-бар для загрузки
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Статус
        self.status_label = QLabel("Готов")
        
        # Кнопки
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Отмена")
        self.apply_btn = QPushButton("Применить")
        
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        self.apply_btn.clicked.connect(self.apply_changes)
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.apply_btn)
        layout.addWidget(self.ok_btn)
        layout.addWidget(self.cancel_btn)
        
        panel.setLayout(layout)
        return panel
    
    def _post_init(self):
        """Пост-инициализация после создания интерфейса"""
        try:
            self.update_stair_ids()
            self.update_zones_list()
            self.update_zone_editor_settings()
            self.on_mode_changed()
        except Exception as e:
            self.logger.error(f"Ошибка пост-инициализации: {e}")
    
    def load_first_frame(self):
        """Загрузка первого кадра из видео"""
        try:
            video_folder = self.camera_config.get("video_folder_path", "")
            if not os.path.exists(video_folder):
                self.status_label.setText("Папка с видео не найдена")
                return
            
            video_files = [f for f in os.listdir(video_folder) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            
            if not video_files:
                self.status_label.setText("Видеофайлы не найдены")
                return
            
            self.video_path = os.path.join(video_folder, video_files[0])
            self.load_frame(0)
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки первого кадра: {e}")
            self.status_label.setText("Ошибка загрузки видео")
    
    def load_frame(self, position: int = 0):
        """Загрузка кадра с использованием отдельного потока"""
        if not self.video_path or not os.path.exists(self.video_path):
            return
        
        self.progress_bar.setVisible(True)
        self.status_label.setText("Загрузка кадра...")
        
        self.frame_loader_thread = FrameLoaderThread(self.video_path, position)
        self.frame_loader_thread.frame_loaded.connect(self.on_frame_loaded)
        self.frame_loader_thread.error_occurred.connect(self.on_frame_load_error)
        self.frame_loader_thread.start()
    
    def on_frame_loaded(self, frame: np.ndarray):
        """Обработка загруженного кадра"""
        try:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.zone_editor.set_image(self.frame)
            self.zone_editor.set_zones(self.roi_zone, self.stair_zones)
            
            self.progress_bar.setVisible(False)
            self.status_label.setText("Кадр загружен")
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки загруженного кадра: {e}")
            self.on_frame_load_error(f"Ошибка обработки кадра: {str(e)}")
    
    def on_frame_load_error(self, error: str):
        """Обработка ошибки загрузки кадра"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ошибка загрузки")
        QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить кадр: {error}")
    
    def load_frame_dialog(self):
        """Диалог загрузки кадра из выбранного видео"""
        try:
            video_folder = self.camera_config.get("video_folder_path", "")
            if not os.path.exists(video_folder):
                QMessageBox.warning(self, "Предупреждение", 
                                  f"Папка с видео не найдена: {video_folder}")
                return
            
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Выбор видеофайла", video_folder, 
                "Видеофайлы (*.mp4 *.avi *.mov *.mkv)"
            )
            
            if file_path:
                self.video_path = file_path
                self.load_frame(0)
                
        except Exception as e:
            self.logger.error(f"Ошибка диалога загрузки кадра: {e}")
    
    def update_stair_ids(self):
        """Обновление списка ID лестниц"""
        try:
            current_id = self.stair_id_combo.currentData()
            
            self.stair_id_combo.blockSignals(True)
            self.stair_id_combo.clear()
            
            for stair in self.stair_zones:
                stair_id = stair.get("id", 0)
                self.stair_id_combo.addItem(f"Лестница {stair_id}", stair_id)
            
            max_id = 0
            for stair in self.stair_zones:
                max_id = max(max_id, stair.get("id", 0))
            
            new_id = max_id + 1
            self.stair_id_combo.addItem(f"Новая лестница ({new_id})", new_id)
            
            index = self.stair_id_combo.findData(current_id)
            if index >= 0:
                self.stair_id_combo.setCurrentIndex(index)
            else:
                self.stair_id_combo.setCurrentIndex(0)
            
            self.stair_id_combo.blockSignals(False)
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления списка лестниц: {e}")
    
    def update_handrail_ids(self):
        """Обновление списка ID поручней"""
        try:
            stair_id = self.stair_id_combo.currentData()
            current_id = self.handrail_id_combo.currentData()
            
            self.handrail_id_combo.blockSignals(True)
            self.handrail_id_combo.clear()
            
            stair_idx = -1
            for idx, stair in enumerate(self.stair_zones):
                if stair.get("id") == stair_id:
                    stair_idx = idx
                    break
            
            if stair_idx >= 0:
                handrails = self.stair_zones[stair_idx].get("handrails", [])
                for handrail in handrails:
                    handrail_id = handrail.get("id", 0)
                    self.handrail_id_combo.addItem(f"Поручень {handrail_id}", handrail_id)
                
                max_id = 0
                for handrail in handrails:
                    max_id = max(max_id, handrail.get("id", 0))
                
                new_id = max_id + 1
                self.handrail_id_combo.addItem(f"Новый поручень ({new_id})", new_id)
            else:
                self.handrail_id_combo.addItem("Поручень 1", 1)
            
            index = self.handrail_id_combo.findData(current_id)
            if index >= 0:
                self.handrail_id_combo.setCurrentIndex(index)
            else:
                self.handrail_id_combo.setCurrentIndex(0)
            
            self.handrail_id_combo.blockSignals(False)
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления списка поручней: {e}")
    
    def on_mode_changed(self):
        """Обработка изменения режима редактирования"""
        try:
            mode = self.mode_combo.currentData()
            
            if mode == "roi":
                self.stair_id_combo.setEnabled(False)
                self.handrail_id_combo.setEnabled(False)
                self.current_zone_info.setText("Редактирование зоны интереса (ROI)")
                
            elif mode == "stair":
                self.stair_id_combo.setEnabled(True)
                self.handrail_id_combo.setEnabled(False)
                stair_id = self.stair_id_combo.currentData()
                self.current_zone_info.setText(f"Редактирование лестницы {stair_id}")
                
            elif mode == "handrail":
                self.stair_id_combo.setEnabled(True)
                self.handrail_id_combo.setEnabled(True)
                self.update_handrail_ids()
                stair_id = self.stair_id_combo.currentData()
                handrail_id = self.handrail_id_combo.currentData()
                self.current_zone_info.setText(f"Редактирование поручня {handrail_id} лестницы {stair_id}")
            
            # Обновление редактора зон
            stair_id = self.stair_id_combo.currentData() or 1
            handrail_id = self.handrail_id_combo.currentData() or 1
            
            self.zone_editor.set_editing_mode(mode, stair_id, handrail_id)
            
        except Exception as e:
            self.logger.error(f"Ошибка изменения режима: {e}")
    
    def on_stair_id_changed(self):
        """Обработка изменения ID лестницы"""
        if self.mode_combo.currentData() == "handrail":
            self.update_handrail_ids()
        self.on_mode_changed()
    
    def on_handrail_id_changed(self):
        """Обработка изменения ID поручня"""
        self.on_mode_changed()
    
    def on_zone_updated(self, zone_type: str, zone_data):
        """Обработка обновления зоны"""
        try:
            if zone_type == "roi":
                self.roi_zone = zone_data
            elif zone_type in ["stair", "handrail"]:
                self.stair_zones = zone_data
                self.update_stair_ids()
                if zone_type == "handrail":
                    self.update_handrail_ids()
            
            self.update_zones_list()
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления зоны: {e}")
    
    def on_point_added(self, x: int, y: int):
        """Обработка добавления точки"""
        self.update_points_table()
        self.status_label.setText(f"Добавлена точка ({x}, {y})")
    
    def on_point_removed(self):
        """Обработка удаления точки"""
        self.update_points_table()
        self.status_label.setText("Точка удалена")
    
    def update_points_table(self):
        """Обновление таблицы точек"""
        try:
            points = self.zone_editor.current_points
            self.points_count_label.setText(f"Точек: {len(points)}")
            
            self.points_table.setRowCount(len(points))
            
            for i, point in enumerate(points):
                self.points_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                self.points_table.setItem(i, 1, QTableWidgetItem(str(int(point[0]))))
                self.points_table.setItem(i, 2, QTableWidgetItem(str(int(point[1]))))
                
        except Exception as e:
            self.logger.error(f"Ошибка обновления таблицы точек: {e}")
    
    def update_zones_list(self):
        """Обновление списка зон"""
        try:
            self.zones_list.clear()
            
            # ROI
            if self.roi_zone:
                roi_item = QListWidgetItem(f"ROI - Зона интереса ({len(self.roi_zone)} точек)")
                roi_item.setData(Qt.UserRole, {"type": "roi", "id": None})
                self.zones_list.addItem(roi_item)
            
            # Лестницы и поручни
            for stair in self.stair_zones:
                stair_id = stair.get("id", 0)
                stair_coords = stair.get("coordinates", [])
                stair_item = QListWidgetItem(f"Лестница {stair_id} ({len(stair_coords)} точек)")
                stair_item.setData(Qt.UserRole, {"type": "stair", "id": stair_id})
                self.zones_list.addItem(stair_item)
                
                if "handrails" in stair:
                    for handrail in stair["handrails"]:
                        handrail_id = handrail.get("id", 0)
                        handrail_coords = handrail.get("coordinates", [])
                        handrail_item = QListWidgetItem(
                            f"    → Поручень {handrail_id} ({len(handrail_coords)} точек)"
                        )
                        handrail_item.setData(Qt.UserRole, {
                            "type": "handrail", 
                            "stair_id": stair_id, 
                            "id": handrail_id
                        })
                        self.zones_list.addItem(handrail_item)
                        
        except Exception as e:
            self.logger.error(f"Ошибка обновления списка зон: {e}")
    
    def update_zone_editor_settings(self):
        """Обновление настроек редактора зон"""
        try:
            if hasattr(self, 'zone_editor'):
                self.zone_editor.show_roi = self.show_roi_cb.isChecked()
                self.zone_editor.show_stairs = self.show_stairs_cb.isChecked()
                self.zone_editor.show_handrails = self.show_handrails_cb.isChecked()
                self.zone_editor.point_radius = self.point_size_slider.value()
                self.zone_editor.line_thickness = self.line_thickness_slider.value()
                self.zone_editor.update_display()
                
        except Exception as e:
            self.logger.error(f"Ошибка обновления настроек редактора: {e}")
    
    def save_current_zone(self):
        """Сохранение текущей зоны"""
        try:
            if self.zone_editor.save_current_zone():
                self.status_label.setText("Зона сохранена")
                QMessageBox.information(self, "Успех", "Зона успешно сохранена!")
            else:
                self.status_label.setText("Ошибка сохранения")
                
        except Exception as e:
            self.logger.error(f"Ошибка сохранения зоны: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения зоны: {str(e)}")
    
    def clear_current_points(self):
        """Очистка текущих точек"""
        self.zone_editor.clear_current_points()
        self.update_points_table()
        self.status_label.setText("Точки очищены")
    
    def on_zone_selected(self, item):
        """Обработка выбора зоны в списке"""
        try:
            zone_data = item.data(Qt.UserRole)
            if not zone_data:
                return
            
            zone_type = zone_data.get("type")
            
            if zone_type == "roi":
                self.mode_combo.setCurrentIndex(self.mode_combo.findData("roi"))
                
            elif zone_type == "stair":
                stair_id = zone_data.get("id")
                self.mode_combo.setCurrentIndex(self.mode_combo.findData("stair"))
                index = self.stair_id_combo.findData(stair_id)
                if index >= 0:
                    self.stair_id_combo.setCurrentIndex(index)
                    
            elif zone_type == "handrail":
                stair_id = zone_data.get("stair_id")
                handrail_id = zone_data.get("id")
                
                self.mode_combo.setCurrentIndex(self.mode_combo.findData("handrail"))
                
                stair_index = self.stair_id_combo.findData(stair_id)
                if stair_index >= 0:
                    self.stair_id_combo.setCurrentIndex(stair_index)
                    self.update_handrail_ids()
                    
                    handrail_index = self.handrail_id_combo.findData(handrail_id)
                    if handrail_index >= 0:
                        self.handrail_id_combo.setCurrentIndex(handrail_index)
            
            self.on_mode_changed()
            
        except Exception as e:
            self.logger.error(f"Ошибка выбора зоны: {e}")
    
    def edit_selected_zone(self):
        """Редактирование выбранной зоны"""
        selected_items = self.zones_list.selectedItems()
        if selected_items:
            self.on_zone_selected(selected_items[0])
    
    def delete_selected_zone(self):
        """Удаление выбранной зоны"""
        try:
            selected_items = self.zones_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "Предупреждение", "Выберите зону для удаления")
                return
            
            item = selected_items[0]
            zone_data = item.data(Qt.UserRole)
            
            if not zone_data:
                return
            
            zone_type = zone_data.get("type")
            
            reply = QMessageBox.question(
                self, "Подтверждение", 
                f"Удалить выбранную зону ({zone_type})?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            if zone_type == "roi":
                self.roi_zone = []
                
            elif zone_type == "stair":
                stair_id = zone_data.get("id")
                self.stair_zones = [s for s in self.stair_zones if s.get("id") != stair_id]
                
            elif zone_type == "handrail":
                stair_id = zone_data.get("stair_id")
                handrail_id = zone_data.get("id")
                
                for stair in self.stair_zones:
                    if stair.get("id") == stair_id and "handrails" in stair:
                        stair["handrails"] = [h for h in stair["handrails"] 
                                            if h.get("id") != handrail_id]
            
            self.zone_editor.set_zones(self.roi_zone, self.stair_zones)
            self.update_zones_list()
            self.update_stair_ids()
            
            self.status_label.setText("Зона удалена")
            
        except Exception as e:
            self.logger.error(f"Ошибка удаления зоны: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка удаления зоны: {str(e)}")
    
    def validate_all_zones(self):
        """Валидация всех зон"""
        try:
            validation_results = []
            
            # Проверка ROI
            if not self.roi_zone:
                validation_results.append("⚠️ ROI зона не настроена")
            elif len(self.roi_zone) < 3:
                validation_results.append("❌ ROI зона содержит менее 3 точек")
            else:
                validation_results.append("✅ ROI зона корректна")
            
            # Проверка лестниц
            if not self.stair_zones:
                validation_results.append("⚠️ Зоны лестниц не настроены")
            else:
                for stair in self.stair_zones:
                    stair_id = stair.get("id", "неизвестно")
                    coords = stair.get("coordinates", [])
                    
                    if len(coords) < 3:
                        validation_results.append(f"❌ Лестница {stair_id}: менее 3 точек")
                    else:
                        validation_results.append(f"✅ Лестница {stair_id}: корректна")
                    
                    # Проверка поручней
                    handrails = stair.get("handrails", [])
                    if not handrails:
                        validation_results.append(f"⚠️ Лестница {stair_id}: нет поручней")
                    else:
                        for handrail in handrails:
                            handrail_id = handrail.get("id", "неизвестно")
                            handrail_coords = handrail.get("coordinates", [])
                            
                            if len(handrail_coords) < 3:
                                validation_results.append(
                                    f"❌ Поручень {handrail_id} лестницы {stair_id}: менее 3 точек"
                                )
                            else:
                                validation_results.append(
                                    f"✅ Поручень {handrail_id} лестницы {stair_id}: корректен"
                                )
            
            # Отображение результатов
            self.validation_results.clear()
            self.validation_results.append("Результаты валидации:\n")
            for result in validation_results:
                self.validation_results.append(result)
            
            # Подсчет ошибок
            errors = len([r for r in validation_results if "❌" in r])
            warnings = len([r for r in validation_results if "⚠️" in r])
            
            if errors == 0 and warnings == 0:
                self.status_label.setText("Валидация пройдена успешно")
            else:
                self.status_label.setText(f"Валидация: {errors} ошибок, {warnings} предупреждений")
                
        except Exception as e:
            self.logger.error(f"Ошибка валидации: {e}")
            self.validation_results.setText(f"Ошибка валидации: {str(e)}")
    
    def apply_changes(self):
        """Применение изменений без закрытия диалога"""
        try:
            # Валидация перед применением
            if not self._validate_before_apply():
                return
            
            self.status_label.setText("Изменения применены")
            QMessageBox.information(self, "Успех", "Изменения успешно применены!")
            
        except Exception as e:
            self.logger.error(f"Ошибка применения изменений: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка применения изменений: {str(e)}")
    
    def _validate_before_apply(self) -> bool:
        """Валидация перед применением изменений"""
        try:
            # Проверяем, что есть хотя бы одна зона
            if not self.roi_zone and not self.stair_zones:
                QMessageBox.warning(self, "Предупреждение", 
                                  "Не настроена ни одна зона. Рекомендуется настроить хотя бы ROI зону.")
                return True  # Не блокируем, но предупреждаем
            
            # Проверяем корректность ROI
            if self.roi_zone and len(self.roi_zone) < 3:
                QMessageBox.critical(self, "Ошибка", 
                                   "ROI зона содержит менее 3 точек. Исправьте или очистите зону.")
                return False
            
            # Проверяем лестницы
            for stair in self.stair_zones:
                coords = stair.get("coordinates", [])
                if len(coords) < 3:
                    stair_id = stair.get("id", "неизвестно")
                    QMessageBox.critical(self, "Ошибка", 
                                       f"Лестница {stair_id} содержит менее 3 точек.")
                    return False
                
                # Проверяем поручни
                handrails = stair.get("handrails", [])
                for handrail in handrails:
                    handrail_coords = handrail.get("coordinates", [])
                    if len(handrail_coords) < 3:
                        handrail_id = handrail.get("id", "неизвестно")
                        stair_id = stair.get("id", "неизвестно")
                        QMessageBox.critical(self, "Ошибка", 
                                           f"Поручень {handrail_id} лестницы {stair_id} содержит менее 3 точек.")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка валидации: {str(e)}")
            return False
    
    def get_zones(self) -> Dict[str, Any]:
        """Получение настроенных зон"""
        return {
            "roi_zone": self.roi_zone,
            "stair_zones": self.stair_zones
        }
    
    def accept(self):
        """Принятие диалога с валидацией"""
        if self._validate_before_apply():
            super().accept()
    
    def closeEvent(self, event):
        """Обработка закрытия диалога"""
        try:
            # Остановка потока загрузки если активен
            if self.frame_loader_thread and self.frame_loader_thread.isRunning():
                self.frame_loader_thread.quit()
                self.frame_loader_thread.wait(1000)
            
            event.accept()
            
        except Exception as e:
            self.logger.error(f"Ошибка закрытия диалога: {e}")
            event.accept()
