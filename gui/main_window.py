import sys
import os
import cv2
import numpy as np
import datetime
import logging
import traceback
from typing import Optional, Dict, Any, List

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QGridLayout, QScrollArea, QFrame,
    QFileDialog, QMessageBox, QLineEdit, QComboBox, QDateEdit, QTableWidget,
    QTableWidgetItem, QProgressBar, QSplitter, QDialog, QFormLayout, QSpinBox,
    QDoubleSpinBox, QCheckBox, QHeaderView, QTextEdit, QGroupBox, QSlider,
    QStatusBar, QMenuBar, QAction, QToolBar
)
from PyQt5.QtCore import Qt, QTimer, QDateTime, pyqtSignal, QThread, QSettings, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon, QColor, QFont, QPalette

from config.camera_config import CameraConfig
from core.video_processor import VideoProcessor
from core.data_manager import DataManager


class VideoProcessingThread(QThread):
    """Улучшенный поток обработки видео с лучшей обработкой ошибок"""
    
    progress_update = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    finished_processing = pyqtSignal()
    
    def __init__(self, video_processor: VideoProcessor, video_path: str):
        super().__init__()
        self.video_processor = video_processor
        self.video_path = video_path
        self.is_processing = False
        self.logger = logging.getLogger(f"{__name__}.VideoProcessingThread")
        self.outcome = "error"
        
    def run(self):
        self.is_processing = True
        try:
            if self.isInterruptionRequested():
                self._progress_callback({'status': 'stopped'})
                return
            self.video_processor.process_video_file(self.video_path, callback=self._progress_callback)
            if self.isInterruptionRequested():
                self.video_processor.stop_processing(wait=False)
            worker = self.video_processor.processing_thread
            if worker is not None:
                worker.join()
        except Exception as exc:
            self.outcome = 'error'
            self.logger.exception("Ошибка обработки видео")
            self.error_occurred.emit(str(exc))
        finally:
            self.is_processing = False
            self.finished_processing.emit()
    
    def _progress_callback(self, data: Dict[str, Any]):
        if not isinstance(data, dict):
            return
        data = dict(data)
        status = data.get('status')
        if status in ('completed', 'stopped', 'error'):
            self.outcome = status
        if self.isInterruptionRequested() and status == 'started':
            self.video_processor.stop_processing(wait=False)
        # Keep the timestamp supplied by the video processor.
        self.progress_update.emit(data)
    
    def stop(self):
        self.requestInterruption()
        self.video_processor.stop_processing(wait=False)


class CameraConfigDialog(QDialog):
    """Улучшенный диалог настройки камеры с валидацией"""
    
    def __init__(self, parent=None, camera_config=None):
        super().__init__(parent)
        self.setWindowTitle("Настройка камеры")
        self.setMinimumWidth(700)
        self.setMinimumHeight(800)
        
        self.camera_config = camera_config or {}
        self.logger = logging.getLogger(f"{__name__}.CameraConfigDialog")
        
        self.init_ui()
        self.load_config()
        
    def init_ui(self):
        """Инициализация улучшенного интерфейса"""
        layout = QVBoxLayout()
        
        # Создаем вкладки для группировки настроек
        tabs = QTabWidget()
        
        # Основные настройки
        basic_tab = self._create_basic_settings_tab()
        tabs.addTab(basic_tab, "Основные")
        
        # Настройки детекции
        detection_tab = self._create_detection_settings_tab()
        tabs.addTab(detection_tab, "Детекция")
        
        # Настройки путей
        paths_tab = self._create_paths_settings_tab()
        tabs.addTab(paths_tab, "Пути")
        
        layout.addWidget(tabs)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        self.validate_btn = QPushButton("Проверить настройки")
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Отмена")
        
        buttons_layout.addWidget(self.validate_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.ok_btn)
        buttons_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
        
        # Подключение сигналов
        self.validate_btn.clicked.connect(self.validate_config)
        self.ok_btn.clicked.connect(self.accept_with_validation)
        self.cancel_btn.clicked.connect(self.reject)
    
    def _create_basic_settings_tab(self) -> QWidget:
        """Создание вкладки основных настроек"""
        widget = QWidget()
        layout = QFormLayout()
        
        # ID камеры
        self.camera_id_edit = QLineEdit()
        self.camera_id_edit.setPlaceholderText("Например: ВК1.9")
        layout.addRow("ID камеры*:", self.camera_id_edit)
        
        # Локация
        self.location_edit = QLineEdit()
        self.location_edit.setPlaceholderText("Например: AUP 2")
        layout.addRow("Локация*:", self.location_edit)
        
        # Описание
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(60)
        self.description_edit.setPlaceholderText("Дополнительное описание камеры")
        layout.addRow("Описание:", self.description_edit)
        
        widget.setLayout(layout)
        return widget
    
    def _create_detection_settings_tab(self) -> QWidget:
        """Создание вкладки настроек детекции"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Группа настроек нарушений
        violation_group = QGroupBox("Настройки нарушений")
        violation_layout = QFormLayout()
        
        self.violation_threshold_spin = QDoubleSpinBox()
        self.violation_threshold_spin.setRange(0.1, 60.0)
        self.violation_threshold_spin.setSingleStep(0.1)
        self.violation_threshold_spin.setSuffix(" сек")
        self.violation_threshold_spin.setToolTip("Время без держания за поручень для фиксации нарушения")
        violation_layout.addRow("Порог нарушения:", self.violation_threshold_spin)
        
        violation_group.setLayout(violation_layout)
        layout.addWidget(violation_group)
        
        # Группа настроек детекции движения
        motion_group = QGroupBox("Детекция движения")
        motion_layout = QFormLayout()
        
        self.motion_threshold_spin = QSpinBox()
        self.motion_threshold_spin.setRange(1, 100)
        self.motion_threshold_spin.setToolTip("Порог чувствительности детекции движения")
        motion_layout.addRow("Порог движения:", self.motion_threshold_spin)
        
        self.motion_min_area_spin = QSpinBox()
        self.motion_min_area_spin.setRange(10, 10000)
        self.motion_min_area_spin.setSuffix(" пикс²")
        self.motion_min_area_spin.setToolTip("Минимальная площадь объекта для детекции движения")
        motion_layout.addRow("Мин. площадь:", self.motion_min_area_spin)
        
        motion_group.setLayout(motion_layout)
        layout.addWidget(motion_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def _create_paths_settings_tab(self) -> QWidget:
        """Создание вкладки настроек путей"""
        widget = QWidget()
        layout = QFormLayout()
        
        # Логотип камеры
        logo_layout = QHBoxLayout()
        self.logo_path_edit = QLineEdit()
        self.logo_browse_btn = QPushButton("Обзор...")
        logo_layout.addWidget(self.logo_path_edit)
        logo_layout.addWidget(self.logo_browse_btn)
        layout.addRow("Логотип:", logo_layout)
        
        # Папка с видео
        video_layout = QHBoxLayout()
        self.video_folder_edit = QLineEdit()
        self.video_folder_browse_btn = QPushButton("Обзор...")
        video_layout.addWidget(self.video_folder_edit)
        video_layout.addWidget(self.video_folder_browse_btn)
        layout.addRow("Папка с видео*:", video_layout)
        
        # Папка для нарушений
        violations_layout = QHBoxLayout()
        self.violations_folder_edit = QLineEdit()
        self.violations_folder_browse_btn = QPushButton("Обзор...")
        violations_layout.addWidget(self.violations_folder_edit)
        violations_layout.addWidget(self.violations_folder_browse_btn)
        layout.addRow("Папка нарушений*:", violations_layout)
        
        # Модель YOLO
        model_layout = QHBoxLayout()
        self.yolo_model_edit = QLineEdit()
        self.model_browse_btn = QPushButton("Обзор...")
        model_layout.addWidget(self.yolo_model_edit)
        model_layout.addWidget(self.model_browse_btn)
        layout.addRow("Модель YOLO*:", model_layout)
        
        # Кнопки настройки зон
        zones_layout = QVBoxLayout()
        self.roi_btn = QPushButton("Настроить зону интереса")
        self.stair_btn = QPushButton("Настроить зоны лестниц и поручней")
        zones_layout.addWidget(self.roi_btn)
        zones_layout.addWidget(self.stair_btn)
        layout.addRow("Зоны:", zones_layout)
        
        widget.setLayout(layout)
        
        # Подключение сигналов
        self.logo_browse_btn.clicked.connect(self.browse_logo)
        self.video_folder_browse_btn.clicked.connect(self.browse_video_folder)
        self.violations_folder_browse_btn.clicked.connect(self.browse_violations_folder)
        self.model_browse_btn.clicked.connect(self.browse_model)
        self.roi_btn.clicked.connect(self.setup_roi)
        self.stair_btn.clicked.connect(self.setup_stair_zones)
        
        return widget
    
    def load_config(self):
        """Загрузка конфигурации в интерфейс"""
        try:
            # Основные настройки
            self.camera_id_edit.setText(self.camera_config.get("camera_id", ""))
            self.location_edit.setText(self.camera_config.get("location", ""))
            self.description_edit.setText(self.camera_config.get("description", ""))
            
            # Настройки детекции
            self.violation_threshold_spin.setValue(
                self.camera_config.get("violation_timer_threshold", 3.0)
            )
            
            motion_config = self.camera_config.get("motion_detection", {})
            self.motion_threshold_spin.setValue(motion_config.get("threshold", 25))
            self.motion_min_area_spin.setValue(motion_config.get("min_area", 500))
            
            # Пути
            self.logo_path_edit.setText(self.camera_config.get("logo_path", ""))
            self.video_folder_edit.setText(self.camera_config.get("video_folder_path", ""))
            self.violations_folder_edit.setText(self.camera_config.get("violations_folder_path", ""))
            self.yolo_model_edit.setText(
                self.camera_config.get("yolo_model_path", "models/yolov8x-pose.pt")
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации: {e}")
            QMessageBox.warning(self, "Предупреждение", 
                              f"Ошибка загрузки конфигурации: {str(e)}")
    
    def validate_config(self) -> bool:
        """Валидация конфигурации с подробными сообщениями"""
        errors = []
        warnings = []
        
        try:
            # Проверка обязательных полей
            if not self.camera_id_edit.text().strip():
                errors.append("ID камеры обязателен для заполнения")
            
            if not self.location_edit.text().strip():
                errors.append("Локация камеры обязательна для заполнения")
            
            if not self.video_folder_edit.text().strip():
                errors.append("Папка с видео обязательна для заполнения")
            
            if not self.violations_folder_edit.text().strip():
                errors.append("Папка для нарушений обязательна для заполнения")
            
            if not self.yolo_model_edit.text().strip():
                errors.append("Путь к модели YOLO обязателен для заполнения")
            
            # Проверка существования путей
            video_folder = self.video_folder_edit.text().strip()
            if video_folder and not os.path.exists(video_folder):
                warnings.append(f"Папка с видео не существует: {video_folder}")
            
            violations_folder = self.violations_folder_edit.text().strip()
            if violations_folder and not os.path.exists(violations_folder):
                warnings.append(f"Папка для нарушений не существует: {violations_folder}")
            
            model_path = self.yolo_model_edit.text().strip()
            if model_path and not os.path.exists(model_path):
                warnings.append(f"Файл модели YOLO не найден: {model_path}")
            
            logo_path = self.logo_path_edit.text().strip()
            if logo_path and not os.path.exists(logo_path):
                warnings.append(f"Файл логотипа не найден: {logo_path}")
            
            # Проверка валидности ID камеры
            camera_id = self.camera_id_edit.text().strip()
            if camera_id and not camera_id.replace('.', '').replace('_', '').replace('-', '').isalnum():
                warnings.append("ID камеры содержит недопустимые символы")
            
            # Проверка значений параметров
            if self.violation_threshold_spin.value() < 0.5:
                warnings.append("Слишком маленький порог нарушения может привести к ложным срабатываниям")
            
            if self.motion_min_area_spin.value() < 100:
                warnings.append("Слишком маленькая минимальная площадь может привести к ложным детекциям")
            
            # Отображение результатов валидации
            if errors:
                QMessageBox.critical(self, "Ошибки конфигурации", 
                                   "Найдены ошибки:\n\n" + "\n".join(f"• {error}" for error in errors))
                return False
            
            if warnings:
                reply = QMessageBox.question(
                    self, "Предупреждения", 
                    "Найдены предупреждения:\n\n" + "\n".join(f"• {warning}" for warning in warnings) + 
                    "\n\nПродолжить?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return False
            
            QMessageBox.information(self, "Валидация", "Конфигурация прошла проверку успешно!")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при валидации: {str(e)}")
            return False
    
    def accept_with_validation(self):
        """Принятие диалога с валидацией"""
        if self.validate_config():
            self.accept()
    
    def browse_logo(self):
        """Выбор файла логотипа"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбор логотипа", "", "Изображения (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.logo_path_edit.setText(path)
    
    def browse_video_folder(self):
        """Выбор папки с видео"""
        path = QFileDialog.getExistingDirectory(self, "Выбор папки с видео")
        if path:
            self.video_folder_edit.setText(path)
    
    def browse_violations_folder(self):
        """Выбор папки для нарушений"""
        path = QFileDialog.getExistingDirectory(self, "Выбор папки для нарушений")
        if path:
            self.violations_folder_edit.setText(path)
    
    def browse_model(self):
        """Выбор файла модели"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбор модели YOLO", "", "Модели (*.pt *.pth *.onnx)"
        )
        if path:
            self.yolo_model_edit.setText(path)
    
    def setup_roi(self):
        """Настройка зоны интереса"""
        QMessageBox.information(
            self, "Информация", 
            "Функционал настройки зоны интереса доступен в редакторе зон"
        )
    
    def setup_stair_zones(self):
        """Настройка зон лестниц и поручней"""
        QMessageBox.information(
            self, "Информация", 
            "Функционал настройки зон лестниц доступен в редакторе зон"
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Получение конфигурации из интерфейса"""
        return {
            "camera_id": self.camera_id_edit.text().strip(),
            "location": self.location_edit.text().strip(),
            "description": self.description_edit.toPlainText().strip(),
            "logo_path": self.logo_path_edit.text().strip(),
            "video_folder_path": self.video_folder_edit.text().strip(),
            "violations_folder_path": self.violations_folder_edit.text().strip(),
            "yolo_model_path": self.yolo_model_edit.text().strip(),
            "violation_timer_threshold": self.violation_threshold_spin.value(),
            "motion_detection": {
                "threshold": self.motion_threshold_spin.value(),
                "min_area": self.motion_min_area_spin.value(),
                "blur_size": 21,
                "dilate_iterations": 2
            },
            "roi_zone": self.camera_config.get("roi_zone", []),
            "stair_zones": self.camera_config.get("stair_zones", [])
        }


class VideoModule(QWidget):
    """Улучшенный модуль обработки видео"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera_config = CameraConfig()
        self.data_manager = DataManager()
        self.logger = logging.getLogger(f"{__name__}.VideoModule")
        self.settings = QSettings()
        
        self.init_ui()
        self.load_cameras()
    
    def init_ui(self):
        """Инициализация улучшенного интерфейса"""
        main_layout = QHBoxLayout()
        
        # Левая панель с улучшенной фильтрацией
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Правая панель с сеткой камер
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 3)
        
        self.setLayout(main_layout)
    
    def _create_left_panel(self) -> QWidget:
        """Создание левой панели с фильтрами"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Заголовок
        header = QLabel("Фильтр камер")
        header.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(header)
        
        # Поиск
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Поиск по ID или локации...")
        self.search_edit.textChanged.connect(self.filter_cameras)
        layout.addWidget(self.search_edit)
        
        # Фильтр по локации
        self.location_combo = QComboBox()
        self.location_combo.addItem("Все локации")
        self.location_combo.currentTextChanged.connect(self.filter_cameras)
        layout.addWidget(self.location_combo)
        
        # Список камер
        self.camera_list = QListWidget()
        self.camera_list.itemClicked.connect(self.select_camera_from_list)
        layout.addWidget(self.camera_list)
        
        # Статистика
        stats_group = QGroupBox("Статистика")
        stats_layout = QVBoxLayout()
        
        self.total_cameras_label = QLabel("Всего камер: 0")
        self.active_cameras_label = QLabel("Активных: 0")
        
        stats_layout.addWidget(self.total_cameras_label)
        stats_layout.addWidget(self.active_cameras_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        panel.setLayout(layout)
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Создание правой панели с камерами"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Заголовок с кнопками управления
        header_layout = QHBoxLayout()
        
        title = QLabel("Камеры видеонаблюдения")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        
        refresh_btn = QPushButton("Обновить")
        refresh_btn.clicked.connect(self.load_cameras)
        
        add_camera_btn = QPushButton("Добавить камеру")
        add_camera_btn.clicked.connect(self.add_camera)
        
        header_layout.addWidget(refresh_btn)
        header_layout.addStretch()
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(add_camera_btn)
        
        layout.addLayout(header_layout)
        
        # Сетка с камерами в прокручиваемой области
        self.camera_grid = QGridLayout()
        
        camera_grid_widget = QWidget()
        camera_grid_widget.setLayout(self.camera_grid)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(camera_grid_widget)
        
        layout.addWidget(scroll_area)
        
        panel.setLayout(layout)
        return panel
    
    def load_cameras(self):
        """Загрузка списка камер с обработкой ошибок"""
        try:
            cameras = self.camera_config.get_all_cameras()
            
            # Очистка интерфейса
            self.camera_list.clear()
            self._clear_camera_grid()
            
            # Обновление комбобокса локаций
            self._update_location_combo(cameras)
            
            # Заполнение списка
            for camera in cameras:
                display_text = f"{camera['camera_id']} - {camera.get('location', 'Не указано')}"
                self.camera_list.addItem(display_text)
            
            # Заполнение сетки
            self._populate_camera_grid(cameras)
            
            # Обновление статистики
            self._update_statistics(cameras)
            
            self.logger.info(f"Загружено {len(cameras)} камер")
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки камер: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки камер: {str(e)}")
    
    def _clear_camera_grid(self):
        """Очистка сетки камер"""
        for i in reversed(range(self.camera_grid.count())):
            widget = self.camera_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
    
    def _update_location_combo(self, cameras: List[Dict[str, Any]]):
        """Обновление комбобокса локаций"""
        current_location = self.location_combo.currentText()
        
        self.location_combo.clear()
        self.location_combo.addItem("Все локации")
        
        locations = set()
        for camera in cameras:
            location = camera.get('location', '').strip()
            if location:
                locations.add(location)
        
        for location in sorted(locations):
            self.location_combo.addItem(location)
        
        # Восстанавливаем выбранную локацию
        index = self.location_combo.findText(current_location)
        if index >= 0:
            self.location_combo.setCurrentIndex(index)
    
    def _populate_camera_grid(self, cameras: List[Dict[str, Any]]):
        """Заполнение сетки камер"""
        row, col = 0, 0
        cols_per_row = 3
        
        for camera in cameras:
            camera_widget = self.create_camera_widget(camera)
            self.camera_grid.addWidget(camera_widget, row, col)
            
            col += 1
            if col >= cols_per_row:
                col = 0
                row += 1
        
        # Добавляем кнопку новой камеры
        add_btn = QPushButton("+ Добавить камеру")
        add_btn.setMinimumHeight(150)
        add_btn.setStyleSheet("""
            QPushButton {
                border: 2px dashed #ccc;
                border-radius: 5px;
                font-size: 14px;
                color: #666;
            }
            QPushButton:hover {
                border-color: #0078d4;
                color: #0078d4;
            }
        """)
        add_btn.clicked.connect(self.add_camera)
        self.camera_grid.addWidget(add_btn, row, col)
    
    def create_camera_widget(self, camera: Dict[str, Any]) -> QWidget:
        """Создание улучшенного виджета камеры"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box | QFrame.Raised)
        widget.setLineWidth(1)
        widget.setMinimumHeight(180)
        widget.setStyleSheet("""
            QFrame {
                border-radius: 5px;
                background-color: #f8f9fa;
            }
            QFrame:hover {
                background-color: #e9ecef;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Логотип или ID камеры
        logo_path = camera.get("logo_path", "")
        if logo_path and os.path.exists(logo_path):
            try:
                logo_label = QLabel()
                pixmap = QPixmap(logo_path)
                logo_label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                logo_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(logo_label)
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки логотипа {logo_path}: {e}")
                self._add_camera_id_label(layout, camera)
        else:
            self._add_camera_id_label(layout, camera)
        
        # Информация о камере
        info_text = f"{camera['camera_id']}\n{camera.get('location', 'Локация не указана')}"
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Кнопки действий
        buttons_layout = QHBoxLayout()
        
        open_btn = QPushButton("Открыть")
        open_btn.clicked.connect(lambda: self.open_camera_processor(camera))
        
        edit_btn = QPushButton("Изменить")
        edit_btn.clicked.connect(lambda: self.edit_camera(camera))
        
        buttons_layout.addWidget(open_btn)
        buttons_layout.addWidget(edit_btn)
        
        layout.addLayout(buttons_layout)
        
        widget.setLayout(layout)
        return widget
    
    def _add_camera_id_label(self, layout: QVBoxLayout, camera: Dict[str, Any]):
        """Добавление метки ID камеры"""
        id_label = QLabel(camera["camera_id"])
        id_label.setAlignment(Qt.AlignCenter)
        id_label.setFont(QFont("Arial", 24, QFont.Bold))
        id_label.setStyleSheet("color: #495057; margin: 10px;")
        layout.addWidget(id_label)
    
    def _update_statistics(self, cameras: List[Dict[str, Any]]):
        """Обновление статистики"""
        total_cameras = len(cameras)
        active_cameras = len([c for c in cameras if self._is_camera_active(c)])
        
        self.total_cameras_label.setText(f"Всего камер: {total_cameras}")
        self.active_cameras_label.setText(f"Активных: {active_cameras}")
    
    def _is_camera_active(self, camera: Dict[str, Any]) -> bool:
        """Проверка активности камеры"""
        video_folder = camera.get("video_folder_path", "")
        return os.path.exists(video_folder)
    
    def filter_cameras(self):
        """Фильтрация списка камер"""
        try:
            search_text = self.search_edit.text().lower()
            location_filter = self.location_combo.currentText()
            
            for i in range(self.camera_list.count()):
                item = self.camera_list.item(i)
                item_text = item.text().lower()
                
                # Фильтр по тексту поиска
                text_match = search_text in item_text
                
                # Фильтр по локации
                location_match = (location_filter == "Все локации" or 
                                location_filter.lower() in item_text)
                
                item.setHidden(not (text_match and location_match))
                
        except Exception as e:
            self.logger.error(f"Ошибка фильтрации камер: {e}")
    
    def select_camera_from_list(self, item):
        """Выбор камеры из списка"""
        try:
            camera_id = item.text().split(" - ")[0]
            camera = self.camera_config.get_camera(camera_id)
            if camera:
                self.open_camera_processor(camera)
        except Exception as e:
            self.logger.error(f"Ошибка выбора камеры: {e}")
            QMessageBox.warning(self, "Ошибка", f"Ошибка выбора камеры: {str(e)}")
    
    def add_camera(self):
        """Добавление новой камеры"""
        try:
            dialog = CameraConfigDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                config = dialog.get_config()
                
                # Проверка уникальности ID
                existing_camera = self.camera_config.get_camera(config["camera_id"])
                if existing_camera:
                    reply = QMessageBox.question(
                        self, "Камера существует",
                        f"Камера с ID '{config['camera_id']}' уже существует. Заменить?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply != QMessageBox.Yes:
                        return
                
                # Сохранение конфигурации
                self.camera_config.add_camera(config)
                
                # Добавление в базу данных
                self.data_manager.add_camera(
                    config["camera_id"],
                    config["location"],
                    config.get("logo_path", "")
                )
                
                # Обновление интерфейса
                self.load_cameras()
                
                QMessageBox.information(self, "Успех", "Камера успешно добавлена!")
                
        except Exception as e:
            self.logger.error(f"Ошибка добавления камеры: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка добавления камеры: {str(e)}")
    
    def edit_camera(self, camera: Dict[str, Any]):
        """Редактирование камеры"""
        try:
            dialog = CameraConfigDialog(self, camera)
            if dialog.exec_() == QDialog.Accepted:
                config = dialog.get_config()
                
                # Сохранение изменений
                self.camera_config.add_camera(config)
                
                # Обновление в базе данных
                self.data_manager.add_camera(
                    config["camera_id"],
                    config["location"],
                    config.get("logo_path", "")
                )
                
                # Обновление интерфейса
                self.load_cameras()
                
                QMessageBox.information(self, "Успех", "Настройки камеры обновлены!")
                
        except Exception as e:
            self.logger.error(f"Ошибка редактирования камеры: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка редактирования камеры: {str(e)}")
    
    def open_camera_processor(self, camera: Dict[str, Any]):
        """Открытие окна обработки камеры"""
        try:
            processor_window = CameraProcessorWindow(self, camera)
            processor_window.setWindowTitle(
                f"Обработка камеры: {camera['camera_id']} - {camera.get('location', '')}"
            )
            processor_window.show()
            
        except Exception as e:
            self.logger.error(f"Ошибка открытия обработчика камеры: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка открытия обработчика: {str(e)}")


class CameraProcessorWindow(QMainWindow):
    """Улучшенное окно обработки видео с камеры"""
    
    def __init__(self, parent=None, camera_config=None):
        super().__init__(parent)
        
        self.camera_config = camera_config
        self.data_manager = DataManager()
        self.video_processor: Optional[VideoProcessor] = None
        self.processing_thread: Optional[VideoProcessingThread] = None
        self.frame_timer = QTimer()
        self.logger = logging.getLogger(f"{__name__}.CameraProcessorWindow")
        
        self.init_ui()
        self.setup_status_bar()
        self.load_videos()
        
        # Подключение таймера обновления кадров с увеличенной частотой
        self.frame_timer.timeout.connect(self.update_frame)
    
    def init_ui(self):
        """Инициализация улучшенного интерфейса"""
        self.setMinimumSize(1400, 900)
        
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Левая панель - список видео
        left_panel = self._create_video_list_panel()
        main_layout.addWidget(left_panel, 2)
        
        # Центральная панель - просмотр видео
        center_panel = self._create_video_view_panel()
        main_layout.addWidget(center_panel, 4)
        
        # Правая панель - настройки и информация
        right_panel = self._create_info_panel()
        main_layout.addWidget(right_panel, 2)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def _create_video_list_panel(self) -> QWidget:
        """Создание панели списка видео"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Заголовок и фильтр
        header = QLabel("Список видео")
        header.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(header)
        
        self.video_filter_edit = QLineEdit()
        self.video_filter_edit.setPlaceholderText("Фильтр по имени файла...")
        self.video_filter_edit.textChanged.connect(self.filter_videos)
        layout.addWidget(self.video_filter_edit)
        
        # Таблица видео
        self.video_table = QTableWidget()
        self.video_table.setColumnCount(4)
        self.video_table.setHorizontalHeaderLabels([
            "Файл", "Дата/Время", "Обработано", "Нарушения"
        ])
        
        header = self.video_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        self.video_table.itemClicked.connect(self.select_video)
        layout.addWidget(self.video_table)
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        
        self.process_btn = QPushButton("Обработать видео")
        self.process_btn.clicked.connect(self.process_selected_video)
        
        self.stop_btn = QPushButton("Остановить")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_processing)
        
        refresh_btn = QPushButton("Обновить")
        refresh_btn.clicked.connect(self.load_videos)
        
        buttons_layout.addWidget(self.process_btn)
        buttons_layout.addWidget(self.stop_btn)
        buttons_layout.addWidget(refresh_btn)
        
        layout.addLayout(buttons_layout)
        
        panel.setLayout(layout)
        return panel
    
    def _create_video_view_panel(self) -> QWidget:
        """Создание панели просмотра видео"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Заголовок
        self.video_view_label = QLabel("Просмотр видео")
        self.video_view_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.video_view_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_view_label)
        
        # Кадр видео
        self.video_frame = QLabel()
        self.video_frame.setMinimumSize(640, 480)
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: #000;
                color: #fff;
            }
        """)
        self.video_frame.setText("Выберите видео для просмотра")
        layout.addWidget(self.video_frame)
        
        # Панель прогресса
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.process_status_label = QLabel()
        self.process_status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.process_status_label)
        
        layout.addLayout(progress_layout)
        
        # Панель режимов обработки
        mode_group = QGroupBox("Режим обработки")
        mode_layout = QVBoxLayout()
        
        self.mode_info_label = QLabel("Режим: не активен")
        self.fps_info_label = QLabel("FPS: -")
        self.persons_info_label = QLabel("Персонал: 0")
        self.violations_info_label = QLabel("Нарушения: 0")
        self.queue_info_label = QLabel("Очередь: 0")  # Добавлено для отладки
        
        mode_layout.addWidget(self.mode_info_label)
        mode_layout.addWidget(self.fps_info_label)
        mode_layout.addWidget(self.persons_info_label)
        mode_layout.addWidget(self.violations_info_label)
        mode_layout.addWidget(self.queue_info_label)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        panel.setLayout(layout)
        return panel
    
    def _create_info_panel(self) -> QWidget:
        """Создание информационной панели"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Информация о камере
        camera_group = QGroupBox("Информация о камере")
        camera_layout = QFormLayout()
        
        self.camera_id_label = QLabel(self.camera_config.get("camera_id", ""))
        self.location_label = QLabel(self.camera_config.get("location", ""))
        self.video_folder_label = QLabel(self.camera_config.get("video_folder_path", ""))
        self.violations_folder_label = QLabel(self.camera_config.get("violations_folder_path", ""))
        self.threshold_label = QLabel(str(self.camera_config.get("violation_timer_threshold", "3.0")))
        
        camera_layout.addRow("ID:", self.camera_id_label)
        camera_layout.addRow("Локация:", self.location_label)
        camera_layout.addRow("Папка видео:", self.video_folder_label)
        camera_layout.addRow("Папка нарушений:", self.violations_folder_label)
        camera_layout.addRow("Порог нарушения:", self.threshold_label)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Кнопки настроек
        settings_group = QGroupBox("Настройки")
        settings_layout = QVBoxLayout()
        
        edit_config_btn = QPushButton("Редактировать настройки")
        edit_config_btn.clicked.connect(self.edit_config)
        
        edit_zones_btn = QPushButton("Редактировать зоны")
        edit_zones_btn.clicked.connect(self.edit_zones)
        
        settings_layout.addWidget(edit_config_btn)
        settings_layout.addWidget(edit_zones_btn)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Логи обработки
        logs_group = QGroupBox("Логи обработки")
        logs_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        self.log_text.setVisible(False)
        
        clear_logs_btn = QPushButton("Очистить логи")
        clear_logs_btn.clicked.connect(lambda: self.log_text.clear())
        
        logs_layout.addWidget(self.log_text)
        logs_layout.addWidget(clear_logs_btn)
        
        logs_group.setLayout(logs_layout)
        layout.addWidget(logs_group)
        
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def setup_status_bar(self):
        """Настройка строки состояния"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Виджеты для строки состояния
        self.status_label = QLabel("Готов")
        self.camera_status_label = QLabel(f"Камера: {self.camera_config.get('camera_id', 'Неизвестно')}")
        
        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.camera_status_label)
    
    def load_videos(self):
        """Загрузка списка видео с улучшенной обработкой ошибок"""
        try:
            video_folder = self.camera_config.get("video_folder_path", "")
            if not video_folder:
                self.status_label.setText("Папка с видео не настроена")
                return
            
            if not os.path.exists(video_folder):
                self.status_label.setText("Папка с видео не найдена")
                QMessageBox.warning(self, "Предупреждение", 
                                  f"Папка с видео не найдена: {video_folder}")
                return
            
            # Получение списка видеофайлов
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
            try:
                all_files = os.listdir(video_folder)
                video_files = [f for f in all_files if f.lower().endswith(video_extensions)]
            except PermissionError:
                QMessageBox.critical(self, "Ошибка", f"Нет доступа к папке: {video_folder}")
                return
            
            # Получение информации об обработанных видео
            try:
                processed_videos = self.data_manager.get_videos_by_camera(self.camera_config["camera_id"])
                processed_filenames = {v["filename"]: v for v in processed_videos}
            except Exception as e:
                self.logger.error(f"Ошибка получения обработанных видео: {e}")
                processed_filenames = {}
            
            # Заполнение таблицы
            self.video_table.setRowCount(len(video_files))
            
            for i, filename in enumerate(sorted(video_files, reverse=True)):
                # Извлечение даты из имени файла
                date_str = self._extract_date_from_filename(filename)
                
                # Информация об обработке
                processed_info = processed_filenames.get(filename)
                if processed_info:
                    processed_status = "Да"
                    violations_count = str(processed_info.get("violation_persons", 0))
                else:
                    processed_status = "Нет"
                    violations_count = "-"
                
                # Заполнение строки таблицы
                self.video_table.setItem(i, 0, QTableWidgetItem(filename))
                self.video_table.setItem(i, 1, QTableWidgetItem(date_str))
                self.video_table.setItem(i, 2, QTableWidgetItem(processed_status))
                self.video_table.setItem(i, 3, QTableWidgetItem(violations_count))
                
                # Цветовое кодирование
                if processed_status == "Да":
                    for col in range(4):
                        item = self.video_table.item(i, col)
                        if item:
                            item.setBackground(QColor(200, 255, 200))  # Светло-зеленый
                
                if violations_count != "-" and violations_count != "0":
                    violations_item = self.video_table.item(i, 3)
                    if violations_item:
                        violations_item.setBackground(QColor(255, 200, 200))  # Светло-красный
            
            self.status_label.setText(f"Загружено {len(video_files)} видеофайлов")
            self.logger.info(f"Загружено {len(video_files)} видеофайлов из {video_folder}")
            
        except Exception as e:
            error_msg = f"Ошибка загрузки видео: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText("Ошибка загрузки видео")
            QMessageBox.critical(self, "Ошибка", error_msg)
    
    def _extract_date_from_filename(self, filename: str) -> str:
        """Извлечение даты из имени файла"""
        try:
            # Формат: 'ВК3.5 2025-04-15 16-44-58_336.avi'
            parts = filename.split(' ')
            if len(parts) >= 3:
                date_part = parts[1]  # 2025-04-15
                time_part = parts[2].split('_')[0].replace('-', ':')  # 16:44:58
                return f"{date_part} {time_part}"
        except Exception as e:
            self.logger.debug(f"Не удалось извлечь дату из {filename}: {e}")
        
        return "Дата неизвестна"
    
    def filter_videos(self):
        """Фильтрация списка видео"""
        filter_text = self.video_filter_edit.text().lower()
        
        for row in range(self.video_table.rowCount()):
            filename_item = self.video_table.item(row, 0)
            date_item = self.video_table.item(row, 1)
            
            if filename_item and date_item:
                show_row = (filter_text in filename_item.text().lower() or 
                           filter_text in date_item.text().lower())
                self.video_table.setRowHidden(row, not show_row)
    
    def select_video(self, item):
        """Выбор видео из списка с предварительным просмотром"""
        try:
            row = item.row()
            filename_item = self.video_table.item(row, 0)
            if not filename_item:
                return
            
            filename = filename_item.text()
            video_path = os.path.join(self.camera_config["video_folder_path"], filename)
            
            if not os.path.exists(video_path):
                QMessageBox.warning(self, "Предупреждение", f"Файл не найден: {video_path}")
                return
            
            self.video_view_label.setText(f"Просмотр: {filename}")
            
            # Захват первого кадра для предварительного просмотра
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                self._display_frame(frame)
                self.status_label.setText(f"Выбрано видео: {filename}")
            else:
                self.video_frame.setText("Ошибка загрузки видео")
                self.status_label.setText("Ошибка загрузки видео")
                
        except Exception as e:
            error_msg = f"Ошибка выбора видео: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText("Ошибка выбора видео")
            QMessageBox.warning(self, "Ошибка", error_msg)
    
    def _display_frame(self, frame: np.ndarray):
        """Отображение кадра в интерфейсе"""
        try:
            # Конвертация в RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            # Создание QImage
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Масштабирование с сохранением пропорций
            scaled_pixmap = pixmap.scaled(
                self.video_frame.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            self.video_frame.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.logger.error(f"Ошибка отображения кадра: {e}")
    
    def process_selected_video(self):
        """Обработка выбранного видео с выбором режима"""
        try:
            # Проверка выбранного видео
            selected_items = self.video_table.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "Предупреждение", "Выберите видео для обработки")
                return
            
            row = selected_items[0].row()
            filename_item = self.video_table.item(row, 0)
            if not filename_item:
                return
            
            filename = filename_item.text()
            video_path = os.path.join(self.camera_config["video_folder_path"], filename)
            
            if not os.path.exists(video_path):
                QMessageBox.warning(self, "Предупреждение", f"Файл не найден: {video_path}")
                return
            
            # Проверка, не идет ли уже обработка
            if self.processing_thread and self.processing_thread.isRunning():
                QMessageBox.warning(self, "Предупреждение", 
                                  "Обработка уже выполняется. Дождитесь завершения или остановите текущую обработку.")
                return
            
            # Диалог выбора режима обработки
            mode_selected = self._show_processing_mode_dialog()
            if mode_selected is None:
                return
            
            # Создание процессора видео
            try:
                if self.video_processor and not self.video_processor.cleanup():
                    raise RuntimeError("Предыдущая обработка еще завершается")
                self.video_processor = VideoProcessor(self.camera_config)
                self.video_processor.set_visualization_mode(mode_selected)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка создания видеопроцессора: {str(e)}")
                return
            
            # Настройка интерфейса для режима без визуализации
            if not mode_selected:
                self._setup_log_display()
            else:
                self.log_text.setVisible(False)
            
            # Запуск обработки
            self._start_video_processing(video_path, filename)
            
        except Exception as e:
            error_msg = f"Ошибка запуска обработки: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Ошибка", error_msg)
    
    def _show_processing_mode_dialog(self) -> Optional[bool]:
        """Показ диалога выбора режима обработки"""
        from PyQt5.QtWidgets import QButtonGroup, QRadioButton
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Выбор режима обработки")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # Описание режимов
        description = QLabel(
            "Выберите режим обработки видео:\n\n"
            "• С визуализацией - отображение обработанного видео с детекциями\n"
            "• Без визуализации - только текстовые логи (быстрее)"
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Радиокнопки
        mode_group = QGroupBox("Режим обработки")
        mode_layout = QVBoxLayout()
        
        button_group = QButtonGroup()
        
        #vis_radio = QRadioButton("С визуализацией")
        #vis_radio.setChecked(True)
        
        logs_radio = QRadioButton("Без визуализации (только логи)")
        
        #button_group.addButton(vis_radio, 1)
        button_group.addButton(logs_radio, 0)
        
        #mode_layout.addWidget(vis_radio)
        mode_layout.addWidget(logs_radio)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Кнопки
        from PyQt5.QtWidgets import QDialogButtonBox
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, dialog
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            return button_group.checkedId() == 1  # True для визуализации, False для логов
        
        return None
    
    def _setup_log_display(self):
        """Настройка отображения логов"""
        self.log_text.setVisible(True)
        self.log_text.clear()
        self.log_text.append("Запуск обработки в режиме без визуализации...\n")
        
        # Настройка логгера для вывода в QTextEdit
        if not hasattr(self, 'log_handler'):
            from PyQt5.QtCore import QObject, pyqtSignal
            
            class LogSignaler(QObject):
                log_message = pyqtSignal(str)
            
            class QTextEditLogHandler(logging.Handler):
                def __init__(self, text_edit):
                    super().__init__()
                    self.text_edit = text_edit
                    self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                    self.signaler = LogSignaler()
                    self.signaler.log_message.connect(self.text_edit.append)
                
                def emit(self, record):
                    msg = self.format(record)
                    self.signaler.log_message.emit(msg)
            
            # Добавляем обработчик к логгеру VideoProcessor
            video_logger = logging.getLogger("VideoProcessor")
            self.log_handler = QTextEditLogHandler(self.log_text)
            self.log_handler.setLevel(logging.DEBUG)
            video_logger.addHandler(self.log_handler)
            video_logger.setLevel(logging.DEBUG)
        
        self.log_text.append("Логгер настроен. Ожидание событий обработки...\n")
    
    def _start_video_processing(self, video_path: str, filename: str):
        """Запуск обработки видео"""
        try:
            # Настройка интерфейса
            self.process_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.process_status_label.setText("Инициализация обработки...")
            
            # Создание и запуск потока обработки
            self.processing_thread = VideoProcessingThread(self.video_processor, video_path)
            self.processing_thread.progress_update.connect(self.update_progress)
            self.processing_thread.error_occurred.connect(self.handle_processing_error)
            self.processing_thread.finished.connect(self.processing_finished)
            self.processing_thread.start()
            
            # Запуск таймера обновления кадров с более высокой частотой
            self.frame_timer.start(33)  # 33 мс = ~30 fps
            
            self.status_label.setText(f"Обработка: {filename}")
            self.logger.info(f"Начата обработка видео: {filename}")
            
        except Exception as e:
            self._reset_processing_ui()
            error_msg = f"Ошибка запуска обработки: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Ошибка", error_msg)
    
    def stop_processing(self):
        """Остановка обработки видео"""
        try:
            if self.processing_thread and self.processing_thread.isRunning():
                self.process_status_label.setText("Остановка обработки...")
                self.processing_thread.stop()
                
                # Останавливаем таймер обновления кадров
                self.frame_timer.stop()
                
                self.logger.info("Обработка остановлена пользователем")
            
        except Exception as e:
            self.logger.error(f"Ошибка остановки обработки: {e}")
    
    def update_frame(self):
        """Обновление кадра при обработке с улучшенной логикой"""
        if not self.video_processor:
            return
        
        try:
            frame_data = self.video_processor.get_next_frame()
            if frame_data:
                visualization, mode, violations = frame_data
                
                # Отображение кадра
                if visualization is not None and visualization.size > 0:
                    self._display_frame(visualization)
                
                # Обновление информации о режиме
                self.mode_info_label.setText(f"Режим: {mode}")
                
                # Обновление информации о нарушениях
                if violations:
                    violation_ids = [str(v["track_id"]) for v in violations]
                    violations_text = f"Нарушения: {', '.join(violation_ids)}"
                    self.violations_info_label.setText(violations_text)
                    self.process_status_label.setText(violations_text)
                else:
                    self.violations_info_label.setText("Нарушения: 0")
                
                # Обновление информации об очереди
                if hasattr(self.video_processor, 'result_queue'):
                    queue_size = self.video_processor.result_queue.qsize()
                    self.queue_info_label.setText(f"Очередь: {queue_size}")
                
        except Exception as e:
            self.logger.error(f"Ошибка обновления кадра: {e}")
    
    def update_progress(self, data: Dict[str, Any]):
        """Обновление прогресса обработки"""
        try:
            status = data.get("status")
            
            if status == "started":
                total_frames = data.get("total_frames", 100)
                self.progress_bar.setMaximum(total_frames)
                self.progress_bar.setValue(0)
                self.process_status_label.setText("Обработка начата...")
                
            elif status == "processing":
                current_frame = data.get("current_frame", 0)
                total_frames = data.get("total_frames", 100)
                progress_percent = (current_frame / total_frames) * 100 if total_frames > 0 else 0
                total_persons = data.get("total_persons", 0)
                violation_persons = data.get("violation_persons", 0)
                avg_fps = data.get("avg_fps", 0)
                mode = data.get("mode", "unknown")
                queue_size = data.get("queue_size", 0)
                
                self.progress_bar.setValue(current_frame)
                
                self.process_status_label.setText(
                    f"Кадр {current_frame}/{total_frames} ({progress_percent:.1f}%)"
                )
                
                self.persons_info_label.setText(f"Персонал: {total_persons}")
                self.violations_info_label.setText(f"Нарушения: {violation_persons}")
                self.fps_info_label.setText(f"FPS: {avg_fps:.1f}")
                self.mode_info_label.setText(f"Режим: {mode}")
                self.queue_info_label.setText(f"Очередь: {queue_size}")
                
            elif status == "completed":
                total_frames = data.get("total_frames", 0)
                total_persons = data.get("total_persons", 0)
                violation_persons = data.get("violation_persons", 0)
                
                self.progress_bar.setValue(self.progress_bar.maximum())
                self.process_status_label.setText(
                    f"Обработка завершена! Персонал: {total_persons}, Нарушений: {violation_persons}"
                )
                
                # Обновление логов в режиме без визуализации
                if self.log_text.isVisible():
                    self.log_text.append("\n======= ИТОГИ ОБРАБОТКИ =======")
                    self.log_text.append(f"Всего кадров обработано: {total_frames}")
                    self.log_text.append(f"Обнаружено персонала: {total_persons}")
                    self.log_text.append(f"Обнаружено нарушителей: {violation_persons}")
                    self.log_text.append("==============================\n")
                
                # Обновление списка видео
                # Список обновляется один раз после завершения рабочего потока.
                
            elif status == "stopped":
                self.process_status_label.setText("Обработка остановлена. Неполный анализ не записан в базу.")

            elif status == "error":
                error_message = data.get("message", "Неизвестная ошибка")
                self.handle_processing_error(error_message)
                
        except Exception as e:
            self.logger.error(f"Ошибка обновления прогресса: {e}")
    
    def handle_processing_error(self, error_message: str):
        """Обработка ошибки во время обработки видео"""
        self.process_status_label.setText(f"Ошибка: {error_message}")
        
        if self.log_text.isVisible():
            self.log_text.append(f"\nОШИБКА: {error_message}\n")
        
        QMessageBox.critical(self, "Ошибка обработки", f"Произошла ошибка:\n{error_message}")
    
    def processing_finished(self):
        self._reset_processing_ui()
        self.load_videos()
        outcome = self.processing_thread.outcome if self.processing_thread else 'error'
        labels = {'completed': 'Обработка завершена', 'stopped': 'Обработка остановлена',
                  'error': 'Обработка завершилась с ошибкой'}
        self.status_label.setText(labels.get(outcome, labels['error']))
        self.logger.info(self.status_label.text())
        if getattr(self, '_close_when_finished', False):
            QTimer.singleShot(0, self.close)
    
    def _reset_processing_ui(self):
        """Сброс интерфейса после обработки"""
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.frame_timer.stop()
        
        # Очистка информации о режиме
        self.mode_info_label.setText("Режим: не активен")
        self.fps_info_label.setText("FPS: -")
        self.queue_info_label.setText("Очередь: 0")
    
    def edit_config(self):
        """Редактирование конфигурации камеры"""
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.information(self, "Обработка идет", "Сначала остановите обработку видео.")
            return
        try:
            dialog = CameraConfigDialog(self, self.camera_config)
            if dialog.exec_() == QDialog.Accepted:
                config = dialog.get_config()
                
                # Сохранение конфигурации
                camera_config_manager = CameraConfig()
                camera_config_manager.add_camera(config)
                
                # Обновление текущей конфигурации
                self.camera_config = config
                
                # Обновление отображения
                self._update_camera_info_display()
                
                # Обновление процессора видео
                if self.video_processor:
                    self.video_processor.cleanup()
                    self.video_processor = None
                
                QMessageBox.information(self, "Успех", "Настройки камеры обновлены!")
                
        except Exception as e:
            error_msg = f"Ошибка редактирования конфигурации: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Ошибка", error_msg)
    
    def _update_camera_info_display(self):
        """Обновление отображения информации о камере"""
        self.camera_id_label.setText(self.camera_config.get("camera_id", ""))
        self.location_label.setText(self.camera_config.get("location", ""))
        self.video_folder_label.setText(self.camera_config.get("video_folder_path", ""))
        self.violations_folder_label.setText(self.camera_config.get("violations_folder_path", ""))
        self.threshold_label.setText(str(self.camera_config.get("violation_timer_threshold", "3.0")))
        
        camera_id = self.camera_config.get("camera_id", "Неизвестно")
        self.camera_status_label.setText(f"Камера: {camera_id}")
    
    def edit_zones(self):
        """Редактирование зон интереса и лестниц"""
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.information(self, "Обработка идет", "Сначала остановите обработку видео.")
            return
        try:
            from gui.zone_editor import ZoneEditorDialog
            
            editor_dialog = ZoneEditorDialog(self, self.camera_config)
            if editor_dialog.exec_() == QDialog.Accepted:
                zones = editor_dialog.get_zones()
                
                # Обновление конфигурации
                self.camera_config["roi_zone"] = zones["roi_zone"]
                self.camera_config["stair_zones"] = zones["stair_zones"]
                
                # Сохранение
                camera_config_manager = CameraConfig()
                camera_config_manager.add_camera(self.camera_config)
                
                # Обновление процессора видео
                if self.video_processor:
                    self.video_processor.cleanup()
                    self.video_processor = None
                
                QMessageBox.information(self, "Успех", "Зоны успешно обновлены!")
                
        except ImportError:
            QMessageBox.warning(self, "Предупреждение", 
                              "Модуль редактора зон не найден. Проверьте установку.")
        except Exception as e:
            error_msg = f"Ошибка редактирования зон: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Ошибка", error_msg)
    
    def closeEvent(self, event):
        if self.processing_thread and self.processing_thread.isRunning():
            if not getattr(self, '_close_when_finished', False):
                reply = QMessageBox.question(self, "Подтверждение",
                                             "Остановить обработку и закрыть окно?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply != QMessageBox.Yes:
                    event.ignore()
                    return
            self._close_when_finished = True
            self.stop_processing()
            event.ignore()
            return
        if self.video_processor:
            self.video_processor.cleanup()
        event.accept()


class AnalyticsModule(QWidget):
    """Улучшенный модуль аналитики с реальными данными"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_manager = DataManager()
        self.logger = logging.getLogger(f"{__name__}.AnalyticsModule")
        
        self.init_ui()
        self.load_data()
    
    def init_ui(self):
        """Инициализация интерфейса аналитики"""
        main_layout = QVBoxLayout()
        
        # Заголовок
        title = QLabel("Аналитика системы безопасности")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Панель фильтров
        filters_panel = self._create_filters_panel()
        main_layout.addWidget(filters_panel)
        
        # Панель ключевых показателей
        dashboard_panel = self._create_dashboard_panel()
        main_layout.addWidget(dashboard_panel)
        
        # Панель статистики по дням
        stats_panel = self._create_daily_stats_panel()
        main_layout.addWidget(stats_panel)
        
        self.setLayout(main_layout)
    
    def _create_filters_panel(self) -> QWidget:
        """Создание панели фильтров"""
        panel = QGroupBox("Фильтры")
        layout = QHBoxLayout()
        
        # Фильтр по дате
        date_layout = QVBoxLayout()
        date_layout.addWidget(QLabel("Период"))
        
        date_range_layout = QHBoxLayout()
        
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDateTime.currentDateTime().addDays(-30).date())
        
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDateTime.currentDateTime().date())
        
        date_range_layout.addWidget(QLabel("С:"))
        date_range_layout.addWidget(self.start_date_edit)
        date_range_layout.addWidget(QLabel("По:"))
        date_range_layout.addWidget(self.end_date_edit)
        
        date_layout.addLayout(date_range_layout)
        layout.addLayout(date_layout)
        
        # Фильтр по локации
        location_layout = QVBoxLayout()
        location_layout.addWidget(QLabel("Локация"))
        
        self.location_combo = QComboBox()
        self.location_combo.addItem("Все локации")
        
        location_layout.addWidget(self.location_combo)
        layout.addLayout(location_layout)
        
        # Фильтр по камере
        camera_layout = QVBoxLayout()
        camera_layout.addWidget(QLabel("Камера"))
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Все камеры")
        
        camera_layout.addWidget(self.camera_combo)
        layout.addLayout(camera_layout)
        
        # Кнопка применения фильтров
        filter_btn_layout = QVBoxLayout()
        filter_btn_layout.addWidget(QLabel(""))  # Выравнивание
        
        apply_btn = QPushButton("Применить фильтры")
        apply_btn.clicked.connect(self.apply_filters)
        
        export_btn = QPushButton("Экспорт данных")
        export_btn.clicked.connect(self.export_data)
        
        filter_btn_layout.addWidget(apply_btn)
        filter_btn_layout.addWidget(export_btn)
        layout.addLayout(filter_btn_layout)
        
        panel.setLayout(layout)
        return panel
    
    def _create_dashboard_panel(self) -> QWidget:
        """Создание панели ключевых показателей"""
        panel = QGroupBox("Ключевые показатели")
        layout = QHBoxLayout()
        
        # Карточки с метриками
        self.total_persons_card = self._create_metric_card("Всего персонала", "0", "#007bff")
        self.violations_card = self._create_metric_card("Нарушений", "0", "#dc3545")
        self.violation_rate_card = self._create_metric_card("Процент нарушений", "0%", "#ffc107")
        self.videos_processed_card = self._create_metric_card("Обработано видео", "0", "#28a745")
        self.processing_hours_card = self._create_metric_card("Часов обработки", "0.0", "#6f42c1")
        
        layout.addWidget(self.total_persons_card)
        layout.addWidget(self.violations_card)
        layout.addWidget(self.violation_rate_card)
        layout.addWidget(self.videos_processed_card)
        layout.addWidget(self.processing_hours_card)
        
        panel.setLayout(layout)
        return panel
    
    def _create_daily_stats_panel(self) -> QWidget:
        """Создание панели статистики по дням"""
        panel = QGroupBox("Статистика по дням")
        layout = QVBoxLayout()
        
        # Скроллируемая область для таблицы
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        # Таблица с данными по дням
        self.daily_stats_table = QTableWidget()
        self.daily_stats_table.setRowCount(4)  # Строки: дата, персонал, нарушения, часы
        
        # Настройка таблицы
        self.daily_stats_table.setVerticalHeaderLabels([
            "Дата",
            "Персонал", 
            "Нарушения",
            "Часы обработки"
        ])
        
        self.daily_stats_table.setAlternatingRowColors(True)
        self.daily_stats_table.verticalHeader().setStretchLastSection(False)
        
        scroll_layout.addWidget(self.daily_stats_table)
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(200)
        
        layout.addWidget(scroll_area)
        panel.setLayout(layout)
        return panel
    
    def _create_metric_card(self, title: str, value: str, color: str) -> QWidget:
        """Создание карточки метрики"""
        card = QFrame()
        card.setFrameStyle(QFrame.Box)
        card.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {color};
                border-radius: 10px;
                background-color: white;
                padding: 10px;
            }}
        """)
        
        layout = QVBoxLayout()
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 10))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"color: {color};")
        title_label.setWordWrap(True)
        
        value_label = QLabel(value)
        value_label.setFont(QFont("Arial", 24, QFont.Bold))
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet(f"color: {color};")
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        
        card.setLayout(layout)
        
        # Сохраняем ссылку на метку значения для обновления
        card.value_label = value_label
        
        return card
    
    def load_data(self):
        """Загрузка данных для аналитики"""
        try:
            # Загрузка локаций
            locations = self.data_manager.get_camera_locations()
            self.location_combo.clear()
            self.location_combo.addItem("Все локации")
            for location in locations:
                self.location_combo.addItem(location)
            
            # Загрузка камер
            cameras = self.data_manager.get_cameras()
            self.camera_combo.clear()
            self.camera_combo.addItem("Все камеры")
            for camera in cameras:
                self.camera_combo.addItem(f"{camera['id']} - {camera['location']}")
            
            # Подключение сигналов
            self.start_date_edit.dateChanged.connect(self.validate_dates)
            self.end_date_edit.dateChanged.connect(self.validate_dates)
            self.location_combo.currentTextChanged.connect(self.update_cameras_by_location)
            
            # Применение фильтров по умолчанию
            self.apply_filters()
            
        except Exception as e:
            error_msg = f"Ошибка загрузки данных аналитики: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Ошибка", error_msg)
    
    def validate_dates(self):
        """Валидация дат"""
        start_date = self.start_date_edit.date()
        end_date = self.end_date_edit.date()
        
        if start_date > end_date:
            self.end_date_edit.setDate(start_date)
    
    def update_cameras_by_location(self):
        """Обновление списка камер при изменении локации"""
        try:
            location = self.location_combo.currentText()
            if location == "Все локации":
                location = None
            
            cameras = self.data_manager.get_cameras()
            
            self.camera_combo.clear()
            self.camera_combo.addItem("Все камеры")
            
            for camera in cameras:
                if location is None or camera['location'] == location:
                    self.camera_combo.addItem(f"{camera['id']} - {camera['location']}")
                    
        except Exception as e:
            self.logger.error(f"Ошибка обновления списка камер: {e}")
    
    def apply_filters(self):
        """Применение фильтров и обновление данных"""
        try:
            # Получение параметров фильтров
            start_date = self.start_date_edit.date().toPyDate()
            end_date = self.end_date_edit.date().toPyDate()
            
            location = self.location_combo.currentText()
            if location == "Все локации":
                location = None
            
            camera_text = self.camera_combo.currentText()
            camera_id = None
            if camera_text != "Все камеры":
                camera_id = camera_text.split(" - ")[0]
            
            # Получение данных с учетом фильтров
            analytics_data = self.data_manager.get_analytics_data(
                start_date=start_date,
                end_date=end_date,
                location=location,
                camera_id=camera_id
            )
            
            # Получение данных по дням из файлов видео
            daily_data = self._get_daily_data_from_videos(start_date, end_date, camera_id, location)
            
            # Обновление таблицы по дням
            self._update_daily_stats_table(daily_data)
            
            # Обновление ключевых показателей
            self._update_metric_cards(analytics_data, daily_data)
            
        except Exception as e:
            error_msg = f"Ошибка применения фильтров: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.warning(self, "Ошибка", error_msg)
    
    def _get_daily_data_from_videos(self, start_date, end_date, camera_id=None, location=None):
        """Получение данных по дням из файлов видео"""
        try:
            # Получаем данные об обработанных видео
            videos_query = """
            SELECT v.filename, v.start_time, v.total_persons, v.violation_persons, 
                   v.duration, v.camera_id, c.location
            FROM processed_videos v
            JOIN cameras c ON v.camera_id = c.id
            WHERE DATE(v.start_time) >= ? AND DATE(v.start_time) <= ?
            """
            
            params = [start_date, end_date]
            
            if camera_id:
                videos_query += " AND v.camera_id = ?"
                params.append(camera_id)
                
            if location:
                videos_query += " AND c.location = ?"
                params.append(location)
                
            videos_query += " ORDER BY v.start_time"
            
            with self.data_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(videos_query, params)
                videos = cursor.fetchall()
            
            # Группируем данные по дням
            daily_data = {}
            
            for video in videos:
                # Извлекаем дату из названия файла или start_time
                video_date = self._extract_date_from_video_record(video)
                if not video_date:
                    continue
                
                date_str = video_date.strftime("%Y-%m-%d")
                
                if date_str not in daily_data:
                    daily_data[date_str] = {
                        'date': video_date,
                        'total_persons': 0,
                        'violation_persons': 0,
                        'processing_hours': 0.0,
                        'videos_count': 0
                    }
                
                daily_data[date_str]['total_persons'] += video['total_persons'] or 0
                daily_data[date_str]['violation_persons'] += video['violation_persons'] or 0
                daily_data[date_str]['processing_hours'] += (video['duration'] or 0) / 3600.0
                daily_data[date_str]['videos_count'] += 1
            
            return daily_data
            
        except Exception as e:
            self.logger.error(f"Ошибка получения данных по дням: {e}")
            return {}
    
    def _extract_date_from_video_record(self, video_record):
        """Извлечение даты из записи видео"""
        try:
            # Сначала пробуем извлечь из имени файла
            filename = video_record['filename']
            # Формат: 'ВК3.5 2025-04-15 16-44-58_336.avi'
            parts = filename.split(' ')
            if len(parts) >= 2:
                date_str = parts[1]  # 2025-04-15
                return datetime.datetime.strptime(date_str, "%Y-%m-%d")
                
            # Если не получилось из имени, используем start_time
            if video_record['start_time']:
                if isinstance(video_record['start_time'], str):
                    return datetime.datetime.fromisoformat(video_record['start_time'].replace('Z', '+00:00'))
                else:
                    return video_record['start_time']
                    
        except Exception as e:
            self.logger.debug(f"Не удалось извлечь дату из {video_record['filename']}: {e}")
        
        return None
    
    def _update_daily_stats_table(self, daily_data):
        """Обновление таблицы статистики по дням"""
        try:
            # Сортируем данные по датам
            sorted_dates = sorted(daily_data.keys())
            
            # Скрываем дни без данных
            visible_dates = [date for date in sorted_dates if daily_data[date]['total_persons'] > 0 or daily_data[date]['violation_persons'] > 0]
            
            if not visible_dates:
                # Если нет данных, показываем пустую таблицу
                self.daily_stats_table.setColumnCount(1)
                self.daily_stats_table.setHorizontalHeaderLabels(["Нет данных"])
                
                for row in range(4):
                    self.daily_stats_table.setItem(row, 0, QTableWidgetItem("-"))
                return
            
            # Устанавливаем количество столбцов равное количеству дней с данными
            self.daily_stats_table.setColumnCount(len(visible_dates))
            
            # Устанавливаем заголовки столбцов (даты)
            date_headers = []
            for date_str in visible_dates:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                formatted_date = date_obj.strftime("%d.%m")
                date_headers.append(formatted_date)
            
            self.daily_stats_table.setHorizontalHeaderLabels(date_headers)
            
            # Заполняем данные по строкам
            for col, date_str in enumerate(visible_dates):
                data = daily_data[date_str]
                
                # Строка 0: Дата (полная)
                full_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%d.%m.%Y")
                date_item = QTableWidgetItem(full_date)
                date_item.setTextAlignment(Qt.AlignCenter)
                self.daily_stats_table.setItem(0, col, date_item)
                
                # Строка 1: Персонал
                persons_item = QTableWidgetItem(str(data['total_persons']))
                persons_item.setTextAlignment(Qt.AlignCenter)
                self.daily_stats_table.setItem(1, col, persons_item)
                
                # Строка 2: Нарушения
                violations_item = QTableWidgetItem(str(data['violation_persons']))
                violations_item.setTextAlignment(Qt.AlignCenter)
                if data['violation_persons'] > 0:
                    violations_item.setBackground(QColor(255, 200, 200))  # Светло-красный
                self.daily_stats_table.setItem(2, col, violations_item)
                
                # Строка 3: Часы обработки
                hours_item = QTableWidgetItem(f"{data['processing_hours']:.1f}")
                hours_item.setTextAlignment(Qt.AlignCenter)
                self.daily_stats_table.setItem(3, col, hours_item)
            
            # Настройка размеров столбцов
            header = self.daily_stats_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeToContents)
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления таблицы по дням: {e}")
    
    def _update_metric_cards(self, analytics_data: Dict[str, Any], daily_data: Dict[str, Any]):
        """Обновление карточек метрик с реальными данными"""
        try:
            total_stats = analytics_data.get("total_stats", {})
            
            # Подсчитываем показатели из daily_data
            total_persons = sum(data['total_persons'] for data in daily_data.values())
            total_violations = sum(data['violation_persons'] for data in daily_data.values())
            total_videos = sum(data['videos_count'] for data in daily_data.values())
            total_hours = sum(data['processing_hours'] for data in daily_data.values())
            
            # Вычисляем процент нарушений
            violation_rate = (total_violations / total_persons * 100) if total_persons > 0 else 0
            
            # Обновление значений карточек
            self.total_persons_card.value_label.setText(str(total_persons))
            self.violations_card.value_label.setText(str(total_violations))
            self.violation_rate_card.value_label.setText(f"{violation_rate:.1f}%")
            self.videos_processed_card.value_label.setText(str(total_videos))
            self.processing_hours_card.value_label.setText(f"{total_hours:.1f}")
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления метрик: {e}")
    
    def export_data(self):
        """Экспорт данных в файл"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Экспорт данных", 
                f"analytics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV файлы (*.csv)"
            )
            
            if not file_path:
                return
            
            # Получение параметров фильтров
            start_date = self.start_date_edit.date().toPyDate()
            end_date = self.end_date_edit.date().toPyDate()
            location = self.location_combo.currentText()
            camera_text = self.camera_combo.currentText()
            
            # Получение данных для экспорта
            camera_id = None
            if camera_text != "Все камеры":
                camera_id = camera_text.split(" - ")[0]
            
            location_filter = None if location == "Все локации" else location
            
            daily_data = self._get_daily_data_from_videos(start_date, end_date, camera_id, location_filter)
            
            # Экспорт в CSV
            import csv
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Заголовок
                writer.writerow(['=== АНАЛИТИКА СИСТЕМЫ БЕЗОПАСНОСТИ ==='])
                writer.writerow([])
                
                # Параметры фильтрации
                writer.writerow(['Параметры фильтрации:'])
                writer.writerow(['Период:', f"{start_date} - {end_date}"])
                writer.writerow(['Локация:', location])
                writer.writerow(['Камера:', camera_text])
                writer.writerow([])
                
                # Ключевые показатели
                writer.writerow(['Ключевые показатели:'])
                total_persons = sum(data['total_persons'] for data in daily_data.values())
                total_violations = sum(data['violation_persons'] for data in daily_data.values())
                total_videos = sum(data['videos_count'] for data in daily_data.values())
                total_hours = sum(data['processing_hours'] for data in daily_data.values())
                violation_rate = (total_violations / total_persons * 100) if total_persons > 0 else 0
                
                writer.writerow(['Всего персонала:', total_persons])
                writer.writerow(['Нарушений:', total_violations])
                writer.writerow(['Процент нарушений:', f"{violation_rate:.1f}%"])
                writer.writerow(['Обработано видео:', total_videos])
                writer.writerow(['Часов обработки:', f"{total_hours:.1f}"])
                writer.writerow([])
                
                # Данные по дням
                writer.writerow(['Статистика по дням:'])
                if daily_data:
                    # Заголовки
                    sorted_dates = sorted(daily_data.keys())
                    visible_dates = [date for date in sorted_dates if daily_data[date]['total_persons'] > 0 or daily_data[date]['violation_persons'] > 0]
                    
                    if visible_dates:
                        headers = ['Показатель'] + [datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%d.%m.%Y") for date in visible_dates]
                        writer.writerow(headers)
                        
                        # Персонал
                        persons_row = ['Персонал'] + [str(daily_data[date]['total_persons']) for date in visible_dates]
                        writer.writerow(persons_row)
                        
                        # Нарушения
                        violations_row = ['Нарушения'] + [str(daily_data[date]['violation_persons']) for date in visible_dates]
                        writer.writerow(violations_row)
                        
                        # Часы обработки
                        hours_row = ['Часы обработки'] + [f"{daily_data[date]['processing_hours']:.1f}" for date in visible_dates]
                        writer.writerow(hours_row)
                    else:
                        writer.writerow(['Нет данных за выбранный период'])
                else:
                    writer.writerow(['Нет данных за выбранный период'])
            
            QMessageBox.information(self, "Экспорт завершен", 
                                  f"Данные успешно экспортированы в файл:\n{file_path}")
            
        except Exception as e:
            error_msg = f"Ошибка экспорта данных: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Ошибка экспорта", error_msg)


class MainWindow(QMainWindow):
    """Улучшенное главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.MainWindow")
        self.settings = QSettings()
        
        self.setWindowTitle("Система видеоаналитики ОКО - Модуль Поручни")
        self.setMinimumSize(1200, 800)
        
        # Восстановление состояния окна
        self._restore_window_state()
        
        self.init_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        
        self.logger.info("Главное окно приложения инициализировано")
    
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        # Создание вкладок
        self.tabs = QTabWidget()
        
        try:
            self.video_module = VideoModule()
            self.analytics_module = AnalyticsModule()
            
            self.tabs.addTab(self.video_module, "📹 Обработка видео")
            #self.tabs.addTab(self.analytics_module, "📊 Аналитика")
            
            self.setCentralWidget(self.tabs)
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации модулей: {e}")
            error_widget = QLabel(f"Ошибка инициализации:\n{str(e)}")
            error_widget.setAlignment(Qt.AlignCenter)
            self.setCentralWidget(error_widget)
    
    def setup_menu_bar(self):
        """Настройка меню"""
        menubar = self.menuBar()
        
        # Меню "Файл"
        file_menu = menubar.addMenu('Файл')
        
        # Действие "Настройки"
        settings_action = QAction('Настройки', self)
        settings_action.setShortcut('Ctrl+,')
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        # Действие "Выход"
        exit_action = QAction('Выход', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Меню "Вид"
        view_menu = menubar.addMenu('Вид')
        
        # Действие "Полноэкранный режим"
        fullscreen_action = QAction('Полноэкранный режим', self)
        fullscreen_action.setShortcut('F11')
        fullscreen_action.setCheckable(True)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Меню "Справка"
        help_menu = menubar.addMenu('Справка')
        
        # Действие "О программе"
        about_action = QAction('О программе', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Настройка строки состояния"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Основная метка состояния
        self.main_status_label = QLabel("Готов к работе")
        self.status_bar.addWidget(self.main_status_label)
        
        # Информация о времени
        self.time_label = QLabel()
        self.status_bar.addPermanentWidget(self.time_label)
        
        # Таймер для обновления времени
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)  # Обновление каждую секунду
        
        self.update_time()
    
    def update_time(self):
        """Обновление времени в строке состояния"""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(current_time)
    
    def show_settings(self):
        """Показ диалога настроек"""
        QMessageBox.information(
            self, "Настройки", 
            "Диалог настроек приложения будет добавлен в следующей версии."
        )
    
    def toggle_fullscreen(self):
        """Переключение полноэкранного режима"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def show_about(self):
        """Показ информации о программе"""
        about_text = """
        <h3>Система видеоаналитики безопасного передвижения по лестницам</h3>
        <p><b>Версия:</b> 1.0.0</p>
        <p><b>Описание:</b> Система для автоматического контроля соблюдения правил безопасности 
        при передвижении персонала по лестницам с использованием компьютерного зрения.</p>
        <p><b>Основные функции:</b></p>
        <ul>
        <li>Детекция движения в зонах лестниц</li>
        <li>Трекинг позы человека</li>
        <li>Автоматическое обнаружение нарушений (не держится за поручень)</li>
        <li>Запись видео нарушений</li>
        <li>Аналитика и отчеты</li>
        </ul>
        <p><b>Технологии:</b> Python, OpenCV, YOLO, PyQt5</p>
        <p><b>Для коммерческих предложений:</b> hazenotes@yandex.ru</p>
        """
        
        QMessageBox.about(self, "О программе", about_text)
    
    def _restore_window_state(self):
        """Восстановление состояния окна"""
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
            
            window_state = self.settings.value("windowState")
            if window_state:
                self.restoreState(window_state)
                
        except Exception as e:
            self.logger.warning(f"Не удалось восстановить состояние окна: {e}")
    
    def _save_window_state(self):
        """Сохранение состояния окна"""
        try:
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("windowState", self.saveState())
        except Exception as e:
            self.logger.warning(f"Не удалось сохранить состояние окна: {e}")
    
    def closeEvent(self, event):
        windows = self.findChildren(CameraProcessorWindow)
        active = [window for window in windows
                  if window.processing_thread and window.processing_thread.isRunning()]
        if active:
            if not getattr(self, '_closing_after_processing', False):
                reply = QMessageBox.question(self, "Подтверждение выхода",
                                             "Остановить активные обработки и выйти?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply != QMessageBox.Yes:
                    event.ignore()
                    return
            self._closing_after_processing = True
            for window in active:
                window.stop_processing()
            QTimer.singleShot(100, self.close)
            event.ignore()
            return
        for window in windows:
            if window.video_processor:
                window.video_processor.cleanup()
        self._save_window_state()
        if hasattr(self, 'time_timer'):
            self.time_timer.stop()
        self.logger.info("Приложение закрывается")
        event.accept()
