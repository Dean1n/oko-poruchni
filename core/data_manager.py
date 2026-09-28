import os
import json
import sqlite3
import datetime
import logging
import threading
from pathlib import Path
from utils.paths import project_path
from typing import Dict, List, Any, Optional, Tuple, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum


class ViolationSeverity(Enum):
    """Уровни серьезности нарушений"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PersonnelRecord:
    """Запись о персонале"""
    id: int
    video_id: int
    global_id: int
    first_seen: datetime.datetime
    last_seen: datetime.datetime
    frames_count: int
    violations_count: int
    first_frame_index: int
    last_frame_index: int
    had_violation: bool = False
    total_violation_time: float = 0.0
    max_violation_time: float = 0.0
    bbox_data: Optional[str] = None  # JSON строка с bbox данными
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return asdict(self)


@dataclass
class ViolationRecord:
    """Запись о нарушении"""
    id: int
    video_id: int
    track_id: int
    global_id: int
    timestamp: datetime.datetime
    duration: float
    severity: ViolationSeverity
    violation_file: Optional[str] = None
    confidence: float = 0.0
    stair_zone_id: Optional[int] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        data = asdict(self)
        data['severity'] = self.severity.value
        return data


@dataclass
class ProcessedVideo:
    """Запись об обработанном видео"""
    id: int
    camera_id: str
    filename: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    duration: float
    processing_time: datetime.datetime
    total_persons: int
    violation_persons: int
    total_frames: int
    processing_fps: float
    file_size: int = 0
    resolution: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return asdict(self)


class DataManager:
    """Улучшенный менеджер данных с расширенным функционалом"""
    
    def __init__(self, database_path: str = "data/analytics.db"):
        """
        Инициализация менеджера данных
        
        Args:
            database_path: Путь к файлу базы данных SQLite
        """
        self.database_path = project_path(database_path)
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Создание схемы базы данных
        self._ensure_database()
        
        # Кэш для частых запросов
        self._cache = {}
        self._cache_timeout = 300  # 5 минут
        self._last_cache_clear = datetime.datetime.now()
        
        self.logger.info(f"DataManager инициализирован с БД: {database_path}")
    
    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для безопасной работы с БД"""
        conn = None
        try:
            with self._lock:
                conn = sqlite3.connect(self.database_path, timeout=30.0)
                conn.row_factory = sqlite3.Row
                # Включение WAL режима для лучшей производительности
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA foreign_keys=ON")
                yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Ошибка работы с БД: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _ensure_database(self):
        """Создание базы данных и необходимых таблиц"""
        try:
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            self._backup_legacy_database()
            
            with self.get_connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                cursor = conn.cursor()
                
                # Таблица камер
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS cameras (
                    id TEXT PRIMARY KEY,
                    location TEXT NOT NULL,
                    logo_path TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
                ''')
                
                # Таблица обработанных видео
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS processed_videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP NOT NULL,
                    duration REAL NOT NULL,
                    processing_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_persons INTEGER DEFAULT 0,
                    violation_persons INTEGER DEFAULT 0,
                    total_frames INTEGER DEFAULT 0,
                    processing_fps REAL DEFAULT 0.0,
                    file_size INTEGER DEFAULT 0,
                    resolution TEXT,
                    FOREIGN KEY (camera_id) REFERENCES cameras (id),
                    UNIQUE(camera_id, filename)
                )
                ''')
                
                # Таблица записей о персонале
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS personnel_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    global_id INTEGER NOT NULL,
                    first_seen TIMESTAMP NOT NULL,
                    last_seen TIMESTAMP NOT NULL,
                    frames_count INTEGER DEFAULT 0,
                    violations_count INTEGER DEFAULT 0,
                    first_frame_index INTEGER DEFAULT 0,
                    last_frame_index INTEGER DEFAULT 0,
                    had_violation BOOLEAN DEFAULT 0,
                    total_violation_time REAL DEFAULT 0.0,
                    max_violation_time REAL DEFAULT 0.0,
                    bbox_data TEXT,
                    FOREIGN KEY (video_id) REFERENCES processed_videos (id),
                    UNIQUE(video_id, global_id)
                )
                ''')
                
                # Таблица нарушений
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    global_id INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    duration REAL NOT NULL,
                    severity TEXT DEFAULT 'medium',
                    violation_file TEXT,
                    confidence REAL DEFAULT 0.0,
                    stair_zone_id INTEGER,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES processed_videos (id)
                )
                ''')
                
                # Таблица статистики по дням
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    date DATE NOT NULL,
                    total_persons INTEGER DEFAULT 0,
                    violation_persons INTEGER DEFAULT 0,
                    total_violations INTEGER DEFAULT 0,
                    processing_hours REAL DEFAULT 0.0,
                    videos_processed INTEGER DEFAULT 0,
                    avg_violation_duration REAL DEFAULT 0.0,
                    FOREIGN KEY (camera_id) REFERENCES cameras (id),
                    UNIQUE(camera_id, date)
                )
                ''')
                
                # Таблица конфигураций
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS configurations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    config_type TEXT NOT NULL,
                    config_data TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (camera_id) REFERENCES cameras (id)
                )
                ''')
                
                # Обновляем старые таблицы до создания индексов.
                self._migrate_columns(cursor)
                self._create_indexes(cursor)
                cursor.execute("PRAGMA user_version=2")
                
                conn.commit()
                self.logger.info("База данных успешно инициализирована")
                
        except Exception as e:
            self.logger.error(f"Ошибка инициализации БД: {e}")
            raise
    
    def _create_indexes(self, cursor):
        """Создание индексов для оптимизации запросов"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_processed_videos_camera_time ON processed_videos(camera_id, start_time)",
            "CREATE INDEX IF NOT EXISTS idx_violations_video_time ON violations(video_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_violations_global_id ON violations(global_id)",
            "CREATE INDEX IF NOT EXISTS idx_personnel_video_global ON personnel_records(video_id, global_id)",
            "CREATE INDEX IF NOT EXISTS idx_daily_stats_camera_date ON daily_stats(camera_id, date)",
            "CREATE INDEX IF NOT EXISTS idx_violations_severity ON violations(severity)",
            "CREATE INDEX IF NOT EXISTS idx_configurations_camera_type ON configurations(camera_id, config_type, is_active)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                self.logger.warning(f"Не удалось создать индекс: {e}")
    
    def _clear_cache_if_needed(self):
        """Очистка кэша при необходимости"""
        now = datetime.datetime.now()
        if (now - self._last_cache_clear).total_seconds() > self._cache_timeout:
            self._cache.clear()
            self._last_cache_clear = now
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Получение данных из кэша"""
        self._clear_cache_if_needed()
        return self._cache.get(key)
    
    def _set_to_cache(self, key: str, value: Any):
        """Сохранение данных в кэш"""
        self._cache[key] = value
    
    def add_camera(self, camera_id: str, location: str, logo_path: str = "", 
                   description: str = "", is_active: bool = True):
        """
        Добавление или обновление информации о камере
        
        Args:
            camera_id: ID камеры
            location: Местоположение камеры
            logo_path: Путь к логотипу камеры
            description: Описание камеры
            is_active: Активность камеры
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Проверяем существование камеры
                cursor.execute("SELECT id FROM cameras WHERE id = ?", (camera_id,))
                exists = cursor.fetchone()
                
                if exists:
                    # Обновляем существующую камеру
                    cursor.execute("""
                    UPDATE cameras 
                    SET location = ?, logo_path = ?, description = ?, 
                        is_active = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """, (location, logo_path, description, is_active, camera_id))
                    self.logger.info(f"Камера {camera_id} обновлена")
                else:
                    # Добавляем новую камеру
                    cursor.execute("""
                    INSERT INTO cameras (id, location, logo_path, description, is_active)
                    VALUES (?, ?, ?, ?, ?)
                    """, (camera_id, location, logo_path, description, is_active))
                    self.logger.info(f"Камера {camera_id} добавлена")
                
                conn.commit()
                
                # Очистка кэша камер
                cache_key = "cameras_all"
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    
        except Exception as e:
            self.logger.error(f"Ошибка добавления камеры {camera_id}: {e}")
            raise
    
    def record_processed_video(self, camera_id: str, filename: str, 
                             start_time: datetime.datetime, end_time: datetime.datetime,
                             total_persons: int, violation_persons: int,
                             total_frames: int = 0, processing_fps: float = 0.0,
                             file_size: int = 0, resolution: str = "") -> int:
        """
        Запись информации об обработанном видео с автоматическим извлечением даты
        
        Args:
            camera_id: ID камеры
            filename: Имя файла
            start_time: Время начала видео
            end_time: Время окончания видео
            total_persons: Общее количество обнаруженных людей
            violation_persons: Количество нарушителей
            total_frames: Общее количество кадров
            processing_fps: Скорость обработки FPS
            file_size: Размер файла в байтах
            resolution: Разрешение видео
            
        Returns:
            ID записи в базе данных
        """
        try:
            duration = (end_time - start_time).total_seconds()
            processing_time = datetime.datetime.now()
            
            # Попытка извлечь дату из имени файла
            video_date = self._extract_date_from_filename(filename)
            if video_date:
                start_time = video_date
                end_time = start_time + datetime.timedelta(seconds=duration)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Проверяем, не обработано ли уже это видео
                cursor.execute("""
                SELECT id FROM processed_videos 
                WHERE camera_id = ? AND filename = ?
                """, (camera_id, filename))
                
                existing = cursor.fetchone()
                if existing:
                    # Обновляем существующую запись
                    video_id = existing['id']
                    cursor.execute("""
                    UPDATE processed_videos 
                    SET start_time = ?, end_time = ?, duration = ?, 
                        processing_time = ?, total_persons = ?, violation_persons = ?,
                        total_frames = ?, processing_fps = ?, file_size = ?, resolution = ?
                    WHERE id = ?
                    """, (start_time, end_time, duration, processing_time, 
                         total_persons, violation_persons, total_frames, 
                         processing_fps, file_size, resolution, video_id))
                    self.logger.info(f"Обновлена запись видео {filename}")
                else:
                    # Создаем новую запись
                    cursor.execute("""
                    INSERT INTO processed_videos 
                    (camera_id, filename, start_time, end_time, duration, processing_time,
                     total_persons, violation_persons, total_frames, processing_fps, 
                     file_size, resolution)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (camera_id, filename, start_time, end_time, duration, 
                         processing_time, total_persons, violation_persons, 
                         total_frames, processing_fps, file_size, resolution))
                    
                    video_id = cursor.lastrowid
                    self.logger.info(f"Записано видео {filename}, ID: {video_id}")
                
                conn.commit()
                
                # Обновляем дневную статистику
                self._update_daily_stats(camera_id, start_time.date(), 
                                       total_persons, violation_persons, 
                                       duration / 3600.0, 1)
                
                return video_id
                
        except Exception as e:
            self.logger.error(f"Ошибка записи видео {filename}: {e}")
            raise
    
    def _extract_date_from_filename(self, filename: str) -> Optional[datetime.datetime]:
        """Извлечение даты из имени файла"""
        try:
            # Формат: 'ВК3.5 2025-04-15 16-44-58_336.avi'
            parts = filename.split(' ')
            if len(parts) >= 3:
                date_str = parts[1]  # 2025-04-15
                time_parts = parts[2].split('_')[0].split('-')  # ['16', '44', '58']
                time_str = ':'.join(time_parts)  # '16:44:58'
                
                timestamp_str = f"{date_str} {time_str}"
                return datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                
        except Exception as e:
            self.logger.debug(f"Не удалось извлечь дату из {filename}: {e}")
        
        return None
    
    def record_personnel_data(self, video_id: int, 
                            personnel_records: Dict[int, Dict[str, Any]]):
        """
        Запись информации о персонале из обработанного видео
        
        Args:
            video_id: ID записи видео
            personnel_records: Словарь с информацией о персонале {global_id: data}
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                for global_id, info in personnel_records.items():
                    # Преобразование bbox данных в JSON
                    bbox_data = None
                    if 'bbox' in info and info['bbox']:
                        bbox_data = json.dumps(info['bbox'])
                    
                    # Подготовка данных
                    first_seen = info.get("first_seen")
                    last_seen = info.get("last_seen")
                    frames_count = info.get("frames_count", 0)
                    violations_count = len(info.get("violations", []))
                    first_frame_index = info.get("frame_index_first", 0)
                    last_frame_index = info.get("frame_index_last", 0)
                    had_violation = info.get("had_violation", False)
                    total_violation_time = sum(v.get("duration", 0) for v in info.get("violations", []))
                    max_violation_time = max((v.get("duration", 0) for v in info.get("violations", [])), default=0)
                    
                    # Проверяем существование записи
                    cursor.execute("""
                    SELECT id FROM personnel_records 
                    WHERE video_id = ? AND global_id = ?
                    """, (video_id, global_id))
                    
                    existing = cursor.fetchone()
                    if existing:
                        # Обновляем существующую запись
                        cursor.execute("""
                        UPDATE personnel_records 
                        SET first_seen = ?, last_seen = ?, frames_count = ?, 
                            violations_count = ?, first_frame_index = ?, 
                            last_frame_index = ?, had_violation = ?,
                            total_violation_time = ?, max_violation_time = ?, bbox_data = ?
                        WHERE id = ?
                        """, (first_seen, last_seen, frames_count, violations_count,
                             first_frame_index, last_frame_index, had_violation,
                             total_violation_time, max_violation_time, bbox_data,
                             existing['id']))
                    else:
                        # Создаем новую запись
                        cursor.execute("""
                        INSERT INTO personnel_records 
                        (video_id, global_id, first_seen, last_seen, frames_count, 
                         violations_count, first_frame_index, last_frame_index, 
                         had_violation, total_violation_time, max_violation_time, bbox_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (video_id, global_id, first_seen, last_seen, frames_count,
                             violations_count, first_frame_index, last_frame_index,
                             had_violation, total_violation_time, max_violation_time, bbox_data))
                
                conn.commit()
                self.logger.info(f"Записаны данные о {len(personnel_records)} персонал")
                
        except Exception as e:
            self.logger.error(f"Ошибка записи данных персонала: {e}")
            raise
    
    def record_violation(self, video_id: int, track_id: int, global_id: int,
                        timestamp: datetime.datetime, duration: float,
                        violation_file: str = "", confidence: float = 0.0,
                        stair_zone_id: Optional[int] = None,
                        description: str = ""):
        """
        Запись информации о нарушении
        
        Args:
            video_id: ID записи видео
            track_id: ID трека нарушителя
            global_id: Глобальный ID персонала
            timestamp: Время нарушения
            duration: Продолжительность нарушения
            violation_file: Имя файла с записью нарушения
            confidence: Уверенность детекции
            stair_zone_id: ID зоны лестницы
            description: Описание нарушения
        """
        try:
            # Определение серьезности нарушения
            severity = self._determine_violation_severity(duration, confidence)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                INSERT INTO violations 
                (video_id, track_id, global_id, timestamp, duration, severity,
                 violation_file, confidence, stair_zone_id, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (video_id, track_id, global_id, timestamp, duration, 
                     severity.value, violation_file, confidence, stair_zone_id, description))
                
                conn.commit()
                self.logger.info(f"Записано нарушение для global_id {global_id}, длительность: {duration:.2f}с")
                
        except Exception as e:
            self.logger.error(f"Ошибка записи нарушения: {e}")
            raise
    
    def _determine_violation_severity(self, duration: float, confidence: float) -> ViolationSeverity:
        """Определение серьезности нарушения"""
        if duration >= 10.0:
            return ViolationSeverity.CRITICAL
        elif duration >= 5.0:
            return ViolationSeverity.HIGH
        elif duration >= 2.0:
            return ViolationSeverity.MEDIUM
        else:
            return ViolationSeverity.LOW
    
    def _update_daily_stats(self, camera_id: str, date: datetime.date,
                          total_persons: int, violation_persons: int,
                          processing_hours: float, videos_count: int):
        """Обновление дневной статистики"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Проверяем существование записи
                cursor.execute("""
                SELECT id, total_persons, violation_persons, total_violations,
                       processing_hours, videos_processed
                FROM daily_stats 
                WHERE camera_id = ? AND date = ?
                """, (camera_id, date))
                
                existing = cursor.fetchone()
                if existing:
                    # Обновляем существующую запись
                    new_total_persons = existing['total_persons'] + total_persons
                    new_violation_persons = existing['violation_persons'] + violation_persons
                    new_processing_hours = existing['processing_hours'] + processing_hours
                    new_videos_count = existing['videos_processed'] + videos_count
                    
                    cursor.execute("""
                    UPDATE daily_stats 
                    SET total_persons = ?, violation_persons = ?, 
                        processing_hours = ?, videos_processed = ?
                    WHERE id = ?
                    """, (new_total_persons, new_violation_persons,
                         new_processing_hours, new_videos_count, existing['id']))
                else:
                    # Создаем новую запись
                    cursor.execute("""
                    INSERT INTO daily_stats 
                    (camera_id, date, total_persons, violation_persons,
                     processing_hours, videos_processed)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (camera_id, date, total_persons, violation_persons,
                         processing_hours, videos_count))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Ошибка обновления дневной статистики: {e}")
    
    def get_cameras(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Получение списка всех камер
        
        Args:
            active_only: Только активные камеры
            
        Returns:
            Список камер с их параметрами
        """
        try:
            cache_key = f"cameras_{'active' if active_only else 'all'}"
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if active_only:
                    cursor.execute("""
                    SELECT id, location, logo_path, description, created_at, updated_at
                    FROM cameras WHERE is_active = 1
                    ORDER BY location, id
                    """)
                else:
                    cursor.execute("""
                    SELECT id, location, logo_path, description, created_at, updated_at, is_active
                    FROM cameras
                    ORDER BY location, id
                    """)
                
                cameras = [dict(row) for row in cursor.fetchall()]
                
                self._set_to_cache(cache_key, cameras)
                return cameras
                
        except Exception as e:
            self.logger.error(f"Ошибка получения списка камер: {e}")
            return []
    
    def get_camera_locations(self) -> List[str]:
        """
        Получение списка всех местоположений камер
        
        Returns:
            Список уникальных местоположений
        """
        try:
            cache_key = "camera_locations"
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                SELECT DISTINCT location FROM cameras 
                WHERE is_active = 1 AND location IS NOT NULL AND location != ''
                ORDER BY location
                """)
                
                locations = [row[0] for row in cursor.fetchall()]
                
                self._set_to_cache(cache_key, locations)
                return locations
                
        except Exception as e:
            self.logger.error(f"Ошибка получения локаций камер: {e}")
            return []
    
    def get_videos_by_camera(self, camera_id: str, 
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Получение списка всех обработанных видео для камеры
        
        Args:
            camera_id: ID камеры
            limit: Ограничение количества записей
            
        Returns:
            Список обработанных видео
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                SELECT id, filename, start_time, end_time, duration, 
                       processing_time, total_persons, violation_persons,
                       total_frames, processing_fps, file_size, resolution
                FROM processed_videos
                WHERE camera_id = ?
                ORDER BY start_time DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, (camera_id,))
                videos = [dict(row) for row in cursor.fetchall()]
                
                return videos
                
        except Exception as e:
            self.logger.error(f"Ошибка получения видео для камеры {camera_id}: {e}")
            return []
    
    def get_violations(self, start_date: Optional[datetime.date] = None,
                      end_date: Optional[datetime.date] = None,
                      camera_id: Optional[str] = None,
                      location: Optional[str] = None,
                      severity: Optional[ViolationSeverity] = None,
                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Получение списка нарушений с фильтрацией
        
        Args:
            start_date: Начальная дата фильтра
            end_date: Конечная дата фильтра
            camera_id: ID камеры для фильтрации
            location: Местоположение для фильтрации
            severity: Серьезность нарушения
            limit: Ограничение количества записей
            
        Returns:
            Список нарушений
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                SELECT v.id, v.track_id, v.global_id, v.timestamp, v.duration, 
                       v.severity, v.violation_file, v.confidence, v.stair_zone_id,
                       v.description, p.camera_id, p.filename, c.location
                FROM violations v
                JOIN processed_videos p ON v.video_id = p.id
                JOIN cameras c ON p.camera_id = c.id
                WHERE 1=1
                """
                
                params = []
                
                if start_date:
                    query += " AND DATE(v.timestamp) >= ?"
                    params.append(start_date.isoformat())
                    
                if end_date:
                    query += " AND DATE(v.timestamp) <= ?"
                    params.append(end_date.isoformat())
                    
                if camera_id:
                    query += " AND p.camera_id = ?"
                    params.append(camera_id)
                    
                if location:
                    query += " AND c.location = ?"
                    params.append(location)
                    
                if severity:
                    query += " AND v.severity = ?"
                    params.append(severity.value)
                
                query += " ORDER BY v.timestamp DESC"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, params)
                violations = [dict(row) for row in cursor.fetchall()]
                
                return violations
                
        except Exception as e:
            self.logger.error(f"Ошибка получения нарушений: {e}")
            return []
    
    def get_analytics_data(self, start_date: Optional[datetime.date] = None,
                          end_date: Optional[datetime.date] = None,
                          camera_id: Optional[str] = None,
                          location: Optional[str] = None) -> Dict[str, Any]:
        """
        Получение агрегированных данных для аналитики
        
        Args:
            start_date: Начальная дата фильтра
            end_date: Конечная дата фильтра
            camera_id: ID камеры для фильтрации
            location: Местоположение для фильтрации
            
        Returns:
            Словарь с агрегированными данными
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Базовые условия фильтрации
                where_conditions = ["1=1"]
                params = []
                
                if start_date:
                    where_conditions.append("DATE(p.start_time) >= ?")
                    params.append(start_date.isoformat())
                    
                if end_date:
                    where_conditions.append("DATE(p.start_time) <= ?")
                    params.append(end_date.isoformat())
                    
                if camera_id:
                    where_conditions.append("p.camera_id = ?")
                    params.append(camera_id)
                    
                if location:
                    where_conditions.append("c.location = ?")
                    params.append(location)
                
                where_clause = " AND ".join(where_conditions)
                
                # Общая статистика
                cursor.execute(f"""
                SELECT 
                    COUNT(DISTINCT p.id) as total_videos,
                    SUM(p.duration) as total_video_hours,
                    SUM(p.total_persons) as total_persons,
                    SUM(p.violation_persons) as total_violations,
                    AVG(p.processing_fps) as avg_processing_fps,
                    COUNT(DISTINCT p.camera_id) as cameras_count
                FROM processed_videos p
                JOIN cameras c ON p.camera_id = c.id
                WHERE {where_clause}
                """, params)
                
                row = cursor.fetchone()
                total_stats = {
                    "total_videos": row['total_videos'] or 0,
                    "total_video_hours": (row['total_video_hours'] or 0) / 3600,
                    "total_persons": row['total_persons'] or 0,
                    "total_violations": row['total_violations'] or 0,
                    "avg_processing_fps": row['avg_processing_fps'] or 0,
                    "cameras_count": row['cameras_count'] or 0
                }
                
                # Статистика по камерам
                cursor.execute(f"""
                SELECT 
                    p.camera_id,
                    c.location,
                    SUM(p.total_persons) as total_persons,
                    SUM(p.violation_persons) as violation_persons,
                    SUM(p.duration) as processed_hours,
                    COUNT(p.id) as videos_count,
                    AVG(p.processing_fps) as avg_fps
                FROM processed_videos p
                JOIN cameras c ON p.camera_id = c.id
                WHERE {where_clause}
                GROUP BY p.camera_id, c.location
                ORDER BY total_persons DESC
                """, params)
                
                camera_stats = []
                for row in cursor.fetchall():
                    camera_stats.append({
                        "camera_id": row['camera_id'],
                        "location": row['location'],
                        "total_persons": row['total_persons'] or 0,
                        "violation_persons": row['violation_persons'] or 0,
                        "processed_hours": (row['processed_hours'] or 0) / 3600,
                        "videos_count": row['videos_count'] or 0,
                        "avg_fps": row['avg_fps'] or 0
                    })
                
                # Статистика нарушений по серьезности
                cursor.execute(f"""
                SELECT v.severity, COUNT(*) as count, AVG(v.duration) as avg_duration
                FROM violations v
                JOIN processed_videos p ON v.video_id = p.id
                JOIN cameras c ON p.camera_id = c.id
                WHERE {where_clause}
                GROUP BY v.severity
                """, params)
                
                severity_stats = {}
                for row in cursor.fetchall():
                    severity_stats[row['severity']] = {
                        "count": row['count'],
                        "avg_duration": row['avg_duration'] or 0
                    }
                
                return {
                    "total_stats": total_stats,
                    "camera_stats": camera_stats,
                    "severity_stats": severity_stats
                }
                
        except Exception as e:
            self.logger.error(f"Ошибка получения аналитических данных: {e}")
            return {
                "total_stats": {},
                "camera_stats": [],
                "severity_stats": {}
            }
    
    def get_daily_statistics(self, camera_id: Optional[str] = None,
                           start_date: Optional[datetime.date] = None,
                           end_date: Optional[datetime.date] = None) -> List[Dict[str, Any]]:
        """
        Получение дневной статистики
        
        Args:
            camera_id: ID камеры (None для всех камер)
            start_date: Начальная дата
            end_date: Конечная дата
            
        Returns:
            Список дневной статистики
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                SELECT ds.camera_id, c.location, ds.date, ds.total_persons,
                       ds.violation_persons, ds.total_violations, ds.processing_hours,
                       ds.videos_processed, ds.avg_violation_duration
                FROM daily_stats ds
                JOIN cameras c ON ds.camera_id = c.id
                WHERE 1=1
                """
                
                params = []
                
                if camera_id:
                    query += " AND ds.camera_id = ?"
                    params.append(camera_id)
                
                if start_date:
                    query += " AND ds.date >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND ds.date <= ?"
                    params.append(end_date)
                
                query += " ORDER BY ds.date DESC, ds.camera_id"
                
                cursor.execute(query, params)
                stats = [dict(row) for row in cursor.fetchall()]
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Ошибка получения дневной статистики: {e}")
            return []
    
    def save_camera_configuration(self, camera_id: str, config_type: str, 
                                config_data: Dict[str, Any]) -> bool:
        """
        Сохранение конфигурации камеры
        
        Args:
            camera_id: ID камеры
            config_type: Тип конфигурации
            config_data: Данные конфигурации
            
        Returns:
            True если сохранение успешно
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Деактивируем предыдущие версии
                cursor.execute("""
                UPDATE configurations 
                SET is_active = 0 
                WHERE camera_id = ? AND config_type = ?
                """, (camera_id, config_type))
                
                # Создаем новую запись
                cursor.execute("""
                INSERT INTO configurations (camera_id, config_type, config_data)
                VALUES (?, ?, ?)
                """, (camera_id, config_type, json.dumps(config_data)))
                
                conn.commit()
                self.logger.info(f"Сохранена конфигурация {config_type} для камеры {camera_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Ошибка сохранения конфигурации: {e}")
            return False
    
    def load_camera_configuration(self, camera_id: str, 
                                config_type: str) -> Optional[Dict[str, Any]]:
        """
        Загрузка конфигурации камеры
        
        Args:
            camera_id: ID камеры
            config_type: Тип конфигурации
            
        Returns:
            Данные конфигурации или None
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                SELECT config_data FROM configurations
                WHERE camera_id = ? AND config_type = ? AND is_active = 1
                ORDER BY created_at DESC LIMIT 1
                """, (camera_id, config_type))
                
                row = cursor.fetchone()
                if row:
                    return json.loads(row['config_data'])
                
                return None
                
        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации: {e}")
            return None
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """
        Очистка старых данных
        
        Args:
            days_to_keep: Количество дней для хранения данных
            
        Returns:
            Словарь с количеством удаленных записей
        """
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
            deleted_counts = {}
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Удаление старых видео и связанных данных
                cursor.execute("""
                DELETE FROM violations 
                WHERE video_id IN (
                    SELECT id FROM processed_videos 
                    WHERE start_time < ?
                )
                """, (cutoff_date,))
                deleted_counts['violations'] = cursor.rowcount
                
                cursor.execute("""
                DELETE FROM personnel_records 
                WHERE video_id IN (
                    SELECT id FROM processed_videos 
                    WHERE start_time < ?
                )
                """, (cutoff_date,))
                deleted_counts['personnel_records'] = cursor.rowcount
                
                cursor.execute("""
                DELETE FROM processed_videos 
                WHERE start_time < ?
                """, (cutoff_date,))
                deleted_counts['processed_videos'] = cursor.rowcount
                
                # Удаление старой дневной статистики
                cutoff_date_only = cutoff_date.date()
                cursor.execute("DELETE FROM daily_stats WHERE date < ?", (cutoff_date_only,))
                deleted_counts['daily_stats'] = cursor.rowcount
                
                # Удаление старых конфигураций
                cursor.execute("""
                DELETE FROM configurations 
                WHERE is_active = 0 AND created_at < ?
                """, (cutoff_date,))
                deleted_counts['configurations'] = cursor.rowcount
                
                conn.commit()
                
                # Оптимизация базы данных
                cursor.execute("VACUUM")
                
                self.logger.info(f"Очистка данных завершена: {deleted_counts}")
                return deleted_counts
                
        except Exception as e:
            self.logger.error(f"Ошибка очистки данных: {e}")
            return {}
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Получение информации о базе данных
        
        Returns:
            Словарь с информацией о БД
        """
        try:
            info = {
                "database_path": self.database_path,
                "file_size": 0,
                "tables": {},
                "indexes": []
            }
            
            # Размер файла
            if os.path.exists(self.database_path):
                info["file_size"] = os.path.getsize(self.database_path)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Информация о таблицах
                cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                
                for table_row in cursor.fetchall():
                    table_name = table_row[0]
                    
                    # Количество записей
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    
                    info["tables"][table_name] = {"count": count}
                
                # Информация об индексах
                cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
                """)
                
                info["indexes"] = [row[0] for row in cursor.fetchall()]
            
            return info
            
        except Exception as e:
            self.logger.error(f"Ошибка получения информации о БД: {e}")
            return {}
    
    def export_data(self, output_path: str, data_type: str = "all",
                   start_date: Optional[datetime.date] = None,
                   end_date: Optional[datetime.date] = None) -> bool:
        """
        Экспорт данных в файл
        
        Args:
            output_path: Путь для сохранения файла
            data_type: Тип данных для экспорта
            start_date: Начальная дата
            end_date: Конечная дата
            
        Returns:
            True если экспорт успешен
        """
        try:
            import csv
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                if data_type == "violations" or data_type == "all":
                    violations = self.get_violations(start_date, end_date)
                    
                    if violations:
                        writer.writerow(["=== НАРУШЕНИЯ ==="])
                        headers = list(violations[0].keys())
                        writer.writerow(headers)
                        
                        for violation in violations:
                            writer.writerow([violation.get(h, "") for h in headers])
                        
                        writer.writerow([])
                
                if data_type == "analytics" or data_type == "all":
                    analytics = self.get_analytics_data(start_date, end_date)
                    
                    writer.writerow(["=== АНАЛИТИКА ==="])
                    
                    # Общая статистика
                    writer.writerow(["Общая статистика"])
                    for key, value in analytics["total_stats"].items():
                        writer.writerow([key, value])
                    
                    writer.writerow([])
                    
                    # Статистика по камерам
                    if analytics["camera_stats"]:
                        writer.writerow(["Статистика по камерам"])
                        headers = list(analytics["camera_stats"][0].keys())
                        writer.writerow(headers)
                        
                        for camera_stat in analytics["camera_stats"]:
                            writer.writerow([camera_stat.get(h, "") for h in headers])
            
            self.logger.info(f"Данные экспортированы в {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка экспорта данных: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Получение статистики производительности
        
        Returns:
            Словарь со статистикой производительности
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Среднее время обработки
                cursor.execute("""
                SELECT AVG(processing_fps) as avg_fps,
                       AVG(duration / total_frames * processing_fps) as avg_processing_ratio
                FROM processed_videos 
                WHERE total_frames > 0 AND processing_fps > 0
                """)
                
                perf_row = cursor.fetchone()
                
                # Статистика по серьезности нарушений
                cursor.execute("""
                SELECT severity, COUNT(*) as count, AVG(duration) as avg_duration
                FROM violations
                GROUP BY severity
                """)
                
                severity_stats = {}
                for row in cursor.fetchall():
                    severity_stats[row['severity']] = {
                        "count": row['count'],
                        "avg_duration": row['avg_duration']
                    }
                
                # Загруженность по дням недели
                cursor.execute("""
                SELECT strftime('%w', start_time) as day_of_week, 
                       COUNT(*) as videos_count,
                       SUM(total_persons) as total_persons
                FROM processed_videos
                GROUP BY strftime('%w', start_time)
                ORDER BY day_of_week
                """)
                
                weekly_stats = {}
                day_names = ['Воскресенье', 'Понедельник', 'Вторник', 'Среда', 
                           'Четверг', 'Пятница', 'Суббота']
                
                for row in cursor.fetchall():
                    day_idx = int(row['day_of_week'])
                    weekly_stats[day_names[day_idx]] = {
                        "videos_count": row['videos_count'],
                        "total_persons": row['total_persons']
                    }
                
                return {
                    "avg_processing_fps": perf_row['avg_fps'] or 0,
                    "avg_processing_ratio": perf_row['avg_processing_ratio'] or 0,
                    "severity_distribution": severity_stats,
                    "weekly_distribution": weekly_stats,
                    "cache_size": len(self._cache)
                }
                
        except Exception as e:
            self.logger.error(f"Ошибка получения статистики производительности: {e}")
            return {}
    
    def close(self):
        """Закрытие соединения и очистка ресурсов"""
        try:
            self._cache.clear()
            self.logger.info("DataManager закрыт")
        except Exception as e:
            self.logger.error(f"Ошибка закрытия DataManager: {e}")

    def _backup_legacy_database(self):
        path = Path(self.database_path)
        if not path.exists() or path.stat().st_size == 0:
            return
        with sqlite3.connect(self.database_path) as source:
            if source.execute("PRAGMA user_version").fetchone()[0] >= 2:
                return
            backup = path.with_name(path.name + '.before_schema_v2.bak')
            if backup.exists():
                backup = path.with_name(path.name + '.before_schema_v2_' +
                                        datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.bak')
            with sqlite3.connect(str(backup)) as target:
                source.backup(target)
            self.logger.info("Резервная копия базы до обновления: %s", backup)

    def _migrate_columns(self, cursor):
        additions = {
            'cameras': {'description': 'TEXT', 'created_at': 'TIMESTAMP',
                        'updated_at': 'TIMESTAMP', 'is_active': 'BOOLEAN DEFAULT 1'},
            'processed_videos': {'total_frames': 'INTEGER DEFAULT 0',
                                 'processing_fps': 'REAL DEFAULT 0.0',
                                 'file_size': 'INTEGER DEFAULT 0', 'resolution': 'TEXT'},
            'violations': {'global_id': 'INTEGER', 'severity': "TEXT DEFAULT 'medium'",
                           'confidence': 'REAL DEFAULT 0.0', 'stair_zone_id': 'INTEGER',
                           'description': 'TEXT', 'created_at': 'TIMESTAMP'},
        }
        for table, columns in additions.items():
            existing = {row[1] for row in cursor.execute(f'PRAGMA table_info({table})')}
            for name, declaration in columns.items():
                if name not in existing:
                    cursor.execute(f'ALTER TABLE {table} ADD COLUMN {name} {declaration}')
        cursor.execute('UPDATE violations SET global_id=track_id WHERE global_id IS NULL')

    def save_analysis(self, camera, filename, start_time, end_time, stats,
                      personnel, episodes, file_size=0, resolution=''):
        """Replace one complete analysis atomically; repeated runs do not inflate totals."""
        camera_id = camera['camera_id']
        total_persons = len(personnel)
        violation_persons = sum(bool(p['had_violation']) for p in personnel.values())
        with self.get_connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute("""INSERT INTO cameras (id, location, logo_path, description)
                            VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET
                            location=excluded.location, logo_path=excluded.logo_path,
                            description=excluded.description, updated_at=CURRENT_TIMESTAMP""",
                         (camera_id, camera.get('location', ''), camera.get('logo_path', ''),
                          camera.get('description', '')))
            old = conn.execute('SELECT id, start_time FROM processed_videos WHERE camera_id=? AND filename=?',
                               (camera_id, filename)).fetchone()
            values = (start_time, end_time, (end_time-start_time).total_seconds(),
                      datetime.datetime.now(), total_persons, violation_persons,
                      stats.frames_processed, stats.avg_fps, file_size, resolution)
            if old:
                video_id = old['id']
                conn.execute("""UPDATE processed_videos SET start_time=?, end_time=?, duration=?,
                             processing_time=?, total_persons=?, violation_persons=?, total_frames=?,
                             processing_fps=?, file_size=?, resolution=? WHERE id=?""", values+(video_id,))
                conn.execute('DELETE FROM violations WHERE video_id=?', (video_id,))
                conn.execute('DELETE FROM personnel_records WHERE video_id=?', (video_id,))
            else:
                cur = conn.execute("""INSERT INTO processed_videos (camera_id, filename, start_time,
                              end_time, duration, processing_time, total_persons, violation_persons,
                              total_frames, processing_fps, file_size, resolution)
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (camera_id, filename)+values)
                video_id = cur.lastrowid
            for global_id, person in personnel.items():
                durations = [e['duration'] for e in episodes if e['global_id'] == global_id]
                conn.execute("""INSERT INTO personnel_records (video_id, global_id, first_seen, last_seen,
                             frames_count, violations_count, first_frame_index, last_frame_index,
                             had_violation, total_violation_time, max_violation_time, bbox_data)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                             (video_id, global_id, person['first_seen'], person['last_seen'],
                              person['frames_count'], len(durations), person['first_frame_index'],
                              person['last_frame_index'], person['had_violation'], sum(durations),
                              max(durations, default=0), json.dumps(person.get('bbox'))))
            for episode in episodes:
                conn.execute("""INSERT INTO violations (video_id, track_id, global_id, timestamp, duration,
                             severity, violation_file, confidence, stair_zone_id, description)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                             (video_id, episode['track_id'], episode['global_id'], episode['timestamp'],
                              episode['duration'], self._determine_violation_severity(
                                  episode['duration'], episode.get('confidence', 0)).value,
                              episode.get('violation_file', ''), episode.get('confidence', 0),
                              episode.get('stair_zone_id'), 'Нет подтвержденного контакта с поручнем'))
            dates = {start_time.date().isoformat()}
            if old and old['start_time']:
                dates.add(str(old['start_time'])[:10])
            for day in dates:
                conn.execute('DELETE FROM daily_stats WHERE camera_id=? AND date=?', (camera_id, day))
                totals = conn.execute("""SELECT COUNT(*) n, COALESCE(SUM(total_persons),0) people,
                                      COALESCE(SUM(violation_persons),0) offenders,
                                      COALESCE(SUM(duration),0)/3600.0 hours
                                      FROM processed_videos WHERE camera_id=? AND DATE(start_time)=?""",
                                      (camera_id, day)).fetchone()
                if totals['n']:
                    violations = conn.execute("""SELECT COUNT(*) n, COALESCE(AVG(v.duration),0) duration
                                               FROM violations v JOIN processed_videos p ON p.id=v.video_id
                                               WHERE p.camera_id=? AND DATE(p.start_time)=?""",
                                               (camera_id, day)).fetchone()
                    conn.execute("""INSERT INTO daily_stats (camera_id, date, total_persons, violation_persons,
                                 total_violations, processing_hours, videos_processed, avg_violation_duration)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                                 (camera_id, day, totals['people'], totals['offenders'], violations['n'],
                                  totals['hours'], totals['n'], violations['duration']))
            conn.commit()
        self._cache.clear()
        return video_id
