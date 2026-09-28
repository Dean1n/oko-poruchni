import cv2
import os
import time
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable
import threading
import queue
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from enum import Enum

from core.motion_detector import MotionDetector
from core.pose_tracker import PoseTracker
from core.data_manager import DataManager
from utils.paths import project_path


class ProcessingMode(Enum):
    """Режимы обработки видео"""
    MOTION_DETECTION = "motion_detection"
    POSE_TRACKING = "pose_tracking"
    IDLE = "idle"


@dataclass
class ProcessingStats:
    """Статистика обработки"""
    frames_processed: int = 0
    motion_frames: int = 0
    tracking_frames: int = 0
    total_persons: int = 0
    total_violations: int = 0
    processing_time: float = 0.0
    avg_fps: float = 0.0


@dataclass
class PersonRecord:
    """Запись о персонале"""
    global_id: int
    track_ids: List[int]
    first_seen: datetime.datetime
    last_seen: datetime.datetime
    frames_count: int = 0
    violations_count: int = 0
    first_frame_index: int = 0
    last_frame_index: int = 0
    had_violation: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None
    in_frame: bool = False


class VideoRecorder:
    """Управление записью видео нарушений"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.active_recordings: Dict[int, Dict[str, Any]] = {}
        self.frame_buffers: Dict[int, List[np.ndarray]] = {}
        self.recording_session = uuid.uuid4().hex[:8]
        self.max_buffer_size = 150  # ~5 секунд при 30 FPS
        self.lock = threading.RLock()
        
        os.makedirs(output_dir, exist_ok=True)
    
    def start_recording(self, global_id: int, timestamp: datetime.datetime, 
                       width: int, height: int, fps: float, camera_id: str):
        """Начало записи для персонала"""
        with self.lock:
            if global_id in self.active_recordings:
                return  # Уже записываем
            
            timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{camera_id}_person_{global_id}_{timestamp_str}_{self.recording_session}.avi"
            filepath = os.path.join(self.output_dir, filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(f"Не удалось создать видеозапись: {filepath}")
            
            # Записываем буферные кадры если есть
            buffer_frames = 0
            if global_id in self.frame_buffers:
                for frame in self.frame_buffers[global_id]:
                    writer.write(frame)
                    buffer_frames += 1
            
            self.active_recordings[global_id] = {
                "writer": writer,
                "filename": filename,
                "filepath": filepath,
                "start_time": timestamp,
                "violation_detected": False,
                "frames_written": buffer_frames
            }
            
            logging.info(f"Начата запись для ID {global_id}, буферных кадров: {buffer_frames}")
    
    def add_frame_to_buffer(self, global_id: int, frame: np.ndarray):
        """Добавление кадра в буфер"""
        with self.lock:
            if global_id not in self.frame_buffers:
                self.frame_buffers[global_id] = []
            
            buffer = self.frame_buffers[global_id]
            buffer.append(frame.copy())
            
            # Ограничиваем размер буфера
            if len(buffer) > self.max_buffer_size:
                buffer.pop(0)
    
    def write_frame(self, global_id: int, frame: np.ndarray, is_violation: bool = False):
        """Запись кадра в активную запись"""
        with self.lock:
            if global_id not in self.active_recordings:
                return
            
            recording = self.active_recordings[global_id]
            writer = recording["writer"]
            
            writer.write(frame)
            recording["frames_written"] += 1
            
            if is_violation:
                recording["violation_detected"] = True
    
    def finalize_recording(self, global_id: int) -> Optional[str]:
        """Завершение записи"""
        with self.lock:
            if global_id not in self.active_recordings:
                return None
            
            recording = self.active_recordings[global_id]
            writer = recording["writer"]
            filepath = recording["filepath"]
            filename = recording["filename"]
            violation_detected = recording["violation_detected"]
            
            writer.release()
            
            # Удаляем файл если не было нарушений
            if not violation_detected:
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    logging.info(f"Удален файл без нарушений: {filename}")
                    filename = None
                except Exception as e:
                    logging.error(f"Ошибка при удалении файла {filepath}: {e}")
            else:
                logging.info(f"Сохранена запись нарушения: {filename}")
            
            del self.active_recordings[global_id]
            
            # Очищаем буфер
            if global_id in self.frame_buffers:
                del self.frame_buffers[global_id]
            
            return filename if violation_detected else None
    
    def cleanup(self):
        errors = []
        with self.lock:
            for global_id in list(self.active_recordings):
                try:
                    self.finalize_recording(global_id)
                except Exception as exc:
                    errors.append(exc)
            self.frame_buffers.clear()
        if errors:
            raise RuntimeError(f"Не удалось закрыть видеозапись: {errors[0]}")


class VideoProcessor:
    """Улучшенный видеопроцессор с оптимизированной архитектурой"""
    
    def __init__(self, camera_config: Dict[str, Any]):
        camera_config = camera_config.copy()
        for key in ('yolo_model_path', 'video_folder_path', 'violations_folder_path'):
            if camera_config.get(key):
                camera_config[key] = project_path(camera_config[key])
        self.camera_config = camera_config
        self.camera_id = camera_config["camera_id"]
        self.logger = logging.getLogger(f"VideoProcessor.{self.camera_id}")
        
        # Инициализация компонентов
        self._init_components()
        
        # Состояние обработки
        self.current_mode = ProcessingMode.MOTION_DETECTION
        self.stats = ProcessingStats()
        
        # Управление персоналом
        self.next_global_id = 1
        self.track_id_mapping: Dict[int, int] = {}
        self.personnel_records: Dict[int, PersonRecord] = {}
        self.episodes = []
        self.active_episodes = {}
        self.recording_files = {}
        self.data_manager = DataManager()
        
        # Управление записью
        violations_folder = camera_config["violations_folder_path"]
        self.video_recorder = VideoRecorder(violations_folder)
        
        # Потоки и синхронизация
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=10)
        self.lock = threading.RLock()
        
        # Состояние обработки
        self.is_processing = False
        self.video_cap = None
        
        # Настройки режимов
        self.mode_config = {
            "motion_detection_threshold": 10,  # кадров для переключения на трекинг
            "tracking_timeout": 15,  # кадров без детекций для возврата к motion
            "cooldown_frames": 20,   # кадров охлаждения между переключениями
        }
        
        self.mode_counters = {
            "consecutive_motion": 0,
            "consecutive_no_tracks": 0,
            "cooldown": 0
        }
        
        # Настройки визуализации
        self.enable_visualization = True
        
    def _init_components(self):
        """Инициализация компонентов обработки"""
        try:
            # Детектор движения
            motion_config = self.camera_config.get("motion_detection", {})
            self.motion_detector = MotionDetector(motion_config)
            roi_zone = self.camera_config.get("roi_zone", [])
            if roi_zone:
                self.motion_detector.set_roi(roi_zone)
            
            # Трекер позы с улучшенной инициализацией
            model_path = project_path(self.camera_config.get("yolo_model_path", "models/model.pt"))
            
            # Проверяем существование файла модели
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Файл модели не найден: {model_path}")
                
            self.pose_tracker = PoseTracker(model_path)
            
            # Установка зон лестниц
            stair_zones = self.camera_config.get("stair_zones", [])
            if stair_zones:
                self.pose_tracker.set_zones(stair_zones)
            
            # Параметры нарушений
            self.violation_threshold = self.camera_config.get("violation_timer_threshold", 3.0)
            
            self.logger.info("Компоненты видеопроцессора инициализированы")
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации компонентов: {e}")
            raise
    
    def set_visualization_mode(self, enable: bool):
        """Установка режима визуализации"""
        self.enable_visualization = enable
        self.logger.info(f"Режим визуализации: {'включен' if enable else 'выключен'}")
    
    def process_video_file(self, video_path: str, callback: Optional[Callable] = None):
        with self.lock:
            if self.processing_thread and self.processing_thread.is_alive():
                raise RuntimeError("Обработка уже выполняется")
            self.stop_event.clear()
            self._reset_state()
            self.is_processing = True
            self.processing_thread = threading.Thread(
                target=self._process_video_thread, args=(video_path, callback), daemon=True)
            try:
                self.processing_thread.start()
            except Exception:
                self.is_processing = False
                raise
        self.logger.info("Запущена обработка видео: %s", video_path)
    
    def stop_processing(self, wait: bool = True, timeout: float = 5.0):
        self.stop_event.set()
        worker = self.processing_thread
        if wait and worker and worker is not threading.current_thread():
            worker.join(timeout=timeout)
        # Resource release belongs to the worker, never to a timed-out caller.
        return not (worker and worker.is_alive())
    
    def get_next_frame(self) -> Optional[Tuple[np.ndarray, str, List[Dict[str, Any]]]]:
        """Получение следующего обработанного кадра"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _reset_state(self):
        with self.lock:
            self.current_mode = ProcessingMode.MOTION_DETECTION
            self.stats = ProcessingStats()
            self.next_global_id = 1
            self.track_id_mapping.clear()
            self.personnel_records.clear()
            self.episodes.clear()
            self.active_episodes.clear()
            self.recording_files.clear()
            self.pose_tracker.reset_tracking()
            self.video_recorder.recording_session = uuid.uuid4().hex[:8]
            self.motion_detector.reset_background()
            self.mode_counters = {key: 0 for key in self.mode_counters}
            self._clear_queues()
    
    def _clear_queues(self):
        """Очистка очередей"""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
    
    def _process_video_thread(self, video_path: str, callback: Optional[Callable]):
        outcome = {'status': 'error', 'message': 'Обработка не завершена'}
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
            self.video_cap = cv2.VideoCapture(video_path)
            if not self.video_cap.isOpened():
                raise ValueError(f"Не удалось открыть видео: {video_path}")
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not np.isfinite(fps) or fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
                raise ValueError("Некорректные параметры видеофайла")
            video_timestamp = self._extract_timestamp_from_filename(os.path.basename(video_path))
            self.logger.info("Видео: %sx%s, %s FPS, %s кадров", width, height, fps, frame_count)
            if callback:
                callback({'status': 'started', 'total_frames': frame_count})
            self._process_frames(fps, width, height, video_timestamp, callback, frame_count)
            if self.stop_event.is_set():
                outcome = {'status': 'stopped', 'current_frame': self.stats.frames_processed}
            else:
                for episode in self.episodes:
                    episode.setdefault('violation_file', self.recording_files.get(episode['global_id'], ''))
                self.data_manager.save_analysis(
                    self.camera_config, os.path.basename(video_path), video_timestamp,
                    video_timestamp + datetime.timedelta(seconds=self.stats.frames_processed / fps),
                    self.stats, {gid: asdict(record) for gid, record in self.personnel_records.items()},
                    self.episodes, os.path.getsize(video_path), f'{width}x{height}')
                outcome = {'status': 'completed', 'total_frames': self.stats.frames_processed,
                           'total_persons': self.stats.total_persons,
                           'violation_persons': self.stats.total_violations}
        except Exception as exc:
            self.logger.exception("Ошибка при обработке видео")
            outcome = {'status': 'error', 'message': str(exc)}
        finally:
            try:
                if self.video_cap is not None:
                    self.video_cap.release()
                    self.video_cap = None
                self.video_recorder.cleanup()
            except Exception as exc:
                self.logger.exception("Ошибка закрытия ресурсов")
                outcome = {'status': 'error', 'message': str(exc)}
            finally:
                self.is_processing = False
            if callback:
                callback(outcome)
    
    def _process_frames(self, fps, width, height, start_timestamp, callback, total_frames):
        """Обработка кадров видео"""
        frame_idx = 0
        last_callback_time = time.time()
        processing_start = time.time()
        
        while not self.stop_event.is_set() and self.video_cap and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if not ret:
                break
            
            frame_idx += 1
            current_timestamp = start_timestamp + datetime.timedelta(seconds=(frame_idx - 1) / fps)
            
            try:
                # Обработка кадра
                violations, visualization = self._process_single_frame(
                    frame, current_timestamp, frame_idx, width, height, fps
                )
                
                # Добавление результата в очередь для визуализации
                if self.enable_visualization and visualization is not None:
                    self._add_result_to_queue(visualization, violations, current_timestamp)
                
                # Обновление статистики
                self.stats.frames_processed += 1
                elapsed = time.time() - processing_start
                self.stats.avg_fps = self.stats.frames_processed / elapsed if elapsed > 0 else 0
                
                # Обновление прогресса
                current_time = time.time()
                if callback and (current_time - last_callback_time) > 0.1:  # Каждые 0.1 сек
                    progress_data = {
                        "status": "processing",
                        "current_frame": frame_idx,
                        "total_frames": total_frames,
                        "progress": frame_idx / total_frames if total_frames > 0 else 0,
                        "timestamp": current_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "mode": self.current_mode.value,
                        "total_persons": self.stats.total_persons,
                        "violation_persons": self.stats.total_violations,
                        "avg_fps": self.stats.avg_fps
                    }
                    callback(progress_data)
                    last_callback_time = current_time
                
                
            except Exception as e:
                self.logger.exception(f"Ошибка при обработке кадра {frame_idx}")
                raise
        
        # Финализация записей
        self._finalize_all_recordings()
        
        # Статистика
        total_time = time.time() - processing_start
        self.stats.processing_time = total_time
        self.stats.avg_fps = self.stats.frames_processed / total_time if total_time > 0 else 0
        
        self.logger.info(f"Обработано {self.stats.frames_processed} кадров за {total_time:.2f}с")
    
    def _process_single_frame(self, frame, timestamp, frame_idx, width, height, fps):
        """Обработка одного кадра"""
        violations = []
        visualization = frame.copy() if self.enable_visualization else None
        
        # Обновление счетчиков режимов
        self._update_mode_counters()
        
        # Обработка в зависимости от текущего режима
        if self.current_mode == ProcessingMode.MOTION_DETECTION:
            violations, visualization = self._process_motion_detection(frame, visualization)
        elif self.current_mode == ProcessingMode.POSE_TRACKING:
            violations, visualization = self._process_pose_tracking(
                frame, visualization, timestamp, frame_idx, width, height, fps
            )
        
        # Write every source frame of an active clip, including short detection gaps.
        self._manage_video_recording(frame, timestamp, width, height, fps, violations)
        # Переключение режимов
        self._update_processing_mode()
        
        return violations, visualization
    
    def _process_motion_detection(self, frame, visualization):
        """Обработка в режиме детекции движения"""
        try:
            motion_detected, motion_viz = self.motion_detector.detect(frame)
            
            if motion_detected:
                self.mode_counters["consecutive_motion"] += 1
                self.stats.motion_frames += 1
            else:
                self.mode_counters["consecutive_motion"] = 0
            
            if self.enable_visualization and motion_viz is not None:
                visualization = motion_viz
            
            return [], visualization
            
        except Exception as e:
            self.logger.error(f"Ошибка детекции движения: {e}")
            return [], visualization
    
    def _process_pose_tracking(self, frame, visualization, timestamp, frame_idx, width, height, fps):
        """Обработка в режиме трекинга поз"""
        try:
            # Трекинг и детекция нарушений
            yolo_violations, pose_viz = self.pose_tracker.process(frame, self.violation_threshold, (frame_idx - 1) / fps)
            
            # Обработка треков
            active_tracks = set()
            if hasattr(self.pose_tracker, 'tracked_persons'):
                active_tracks = set(self.pose_tracker.current_track_ids)
            
            if active_tracks:
                self.mode_counters["consecutive_no_tracks"] = 0
                self.stats.tracking_frames += 1
                
            else:
                self.mode_counters["consecutive_no_tracks"] += 1

            self._update_personnel_records(active_tracks, yolo_violations, timestamp, frame_idx)
            
            # Преобразование в глобальные ID
            global_violations = self._convert_to_global_violations(yolo_violations)
            
            if self.enable_visualization and pose_viz is not None:
                visualization = self._enhance_pose_visualization(pose_viz, global_violations)
            
            return global_violations, visualization
            
        except Exception as e:
            self.logger.exception("Ошибка трекинга поз")
            raise
    
    def _update_personnel_records(self, active_tracks, violations, timestamp, frame_idx):
        """Обновление записей о персонале"""
        with self.lock:
            current_global_ids = set()
            
            # Обработка активных треков
            for track_id in active_tracks:
                if track_id not in self.track_id_mapping:
                    # Новый персонал
                    global_id = self.next_global_id
                    self.track_id_mapping[track_id] = global_id
                    self.next_global_id += 1
                    
                    self.personnel_records[global_id] = PersonRecord(
                        global_id=global_id,
                        track_ids=[track_id],
                        first_seen=timestamp,
                        last_seen=timestamp,
                        frames_count=1,
                        first_frame_index=frame_idx,
                        last_frame_index=frame_idx,
                        in_frame=True
                    )
                    
                    self.logger.debug(f"Новый персонал: track_id {track_id} -> global_id {global_id}")
                    
                else:
                    # Существующий персонал
                    global_id = self.track_id_mapping[track_id]
                    if global_id in self.personnel_records:
                        record = self.personnel_records[global_id]
                        record.last_seen = timestamp
                        record.frames_count += 1
                        record.last_frame_index = frame_idx
                        record.in_frame = True
                        
                        if track_id not in record.track_ids:
                            record.track_ids.append(track_id)
                
                global_id = self.track_id_mapping[track_id]
                self.personnel_records[global_id].bbox = self.pose_tracker.tracked_persons[track_id].get('bbox')
                current_global_ids.add(global_id)
            
            # Обновление статуса "не в кадре"
            for global_id, record in self.personnel_records.items():
                record.in_frame = global_id in current_global_ids
            
            # One database event per uninterrupted confirmed violation episode.
            current_violation_ids = {v['track_id'] for v in violations}
            for track_id in list(self.active_episodes):
                if track_id not in current_violation_ids:
                    del self.active_episodes[track_id]
            for violation in violations:
                track_id = violation['track_id']
                if track_id in self.track_id_mapping:
                    global_id = self.track_id_mapping[track_id]
                    if global_id in self.personnel_records:
                        self.personnel_records[global_id].had_violation = True
                        if track_id not in self.active_episodes:
                            episode = {'track_id': track_id, 'global_id': global_id,
                                       'timestamp': timestamp - datetime.timedelta(seconds=violation['violation_time']),
                                       'duration': violation['violation_time'],
                                       'stair_zone_id': violation.get('stair_zone_id'),
                                       'confidence': violation.get('confidence', 0.0)}
                            self.episodes.append(episode)
                            self.active_episodes[track_id] = episode
                            self.personnel_records[global_id].violations_count += 1
                        else:
                            self.active_episodes[track_id]['duration'] = violation['violation_time']
            
            # Обновление общей статистики
            self.stats.total_persons = len(self.personnel_records)
            self.stats.total_violations = sum(1 for p in self.personnel_records.values() if p.had_violation)
    
    def _manage_video_recording(self, frame, timestamp, width, height, fps, violations):
        violation_track_ids = {v['track_id'] for v in violations}
        for track_id in self.pose_tracker.current_track_ids:
            global_id = self.track_id_mapping.get(track_id)
            if global_id is not None and global_id not in self.video_recorder.active_recordings:
                self.video_recorder.start_recording(global_id, timestamp, width, height, fps, self.camera_id)
        self._finalize_inactive_recordings(timestamp)
        # Use one frame per source frame so exported clip speed remains correct.
        visible = {self.track_id_mapping[tid]: tid for tid in self.pose_tracker.current_track_ids
                   if tid in self.track_id_mapping}
        for global_id in list(self.video_recorder.active_recordings):
            track_id = visible.get(global_id)
            is_violation = track_id in violation_track_ids
            frame_copy = frame.copy()
            if track_id is not None:
                self._draw_person_on_frame(frame_copy, global_id,
                                           self.pose_tracker.tracked_persons[track_id], is_violation)
            self.video_recorder.write_frame(global_id, frame_copy, is_violation)
    
    def _draw_person_on_frame(self, frame, global_id, person_info, is_violation):
        """Отрисовка персонала на кадре для записи"""
        bbox = person_info.get('bbox')
        if not bbox:
            return
        
        x1, y1, x2, y2 = bbox
        color = ((0, 0, 255) if is_violation else
                 (0, 180, 255) if person_info.get('contact_state') == 'unknown' else (0, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
        #status = "VIOLATION" if is_violation else "OK"
        #label = f"ID: {global_id} - {status}"
        
        #cv2.putText(frame, label, (x1, y1 - 10), 
                   #cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _finalize_inactive_recordings(self, current_timestamp):
        """Завершение записей для неактивного персонала"""
        to_finalize = []
        
        for global_id in self.video_recorder.active_recordings:
            if global_id in self.personnel_records:
                record = self.personnel_records[global_id]
                if not record.in_frame:
                    time_since_last_seen = (current_timestamp - record.last_seen).total_seconds()
                    if time_since_last_seen > 2.0:  # 2 секунды
                        to_finalize.append(global_id)
        
        for global_id in to_finalize:
            filename = self.video_recorder.finalize_recording(global_id)
            if filename:
                self._remember_recording(global_id, filename)
                self.logger.info(f"Сохранена запись нарушения для ID {global_id}: {filename}")
    
    def _finalize_all_recordings(self):
        for global_id in list(self.video_recorder.active_recordings):
            filename = self.video_recorder.finalize_recording(global_id)
            if filename:
                self._remember_recording(global_id, filename)
        self.active_episodes.clear()
    
    def _convert_to_global_violations(self, yolo_violations):
        """Преобразование track_id в global_id для нарушений"""
        global_violations = []
        
        for violation in yolo_violations:
            track_id = violation['track_id']
            if track_id in self.track_id_mapping:
                global_violation = violation.copy()
                global_violation['global_id'] = self.track_id_mapping[track_id]
                global_violations.append(global_violation)
        
        return global_violations
    
    def _enhance_pose_visualization(self, pose_viz, violations):
        """Улучшение визуализации поз с глобальными ID"""
        try:
            enhanced = pose_viz.copy()
            
            # Добавляем глобальные ID вместо track_id
            for track_id, global_id in self.track_id_mapping.items():
                if track_id in self.pose_tracker.tracked_persons:
                    person_info = self.pose_tracker.tracked_persons[track_id]
                    bbox = person_info.get('bbox')
                    
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        
                        # Определяем есть ли нарушение
                        has_violation = any(v['global_id'] == global_id for v in violations)
                        color = (0, 0, 255) if has_violation else (0, 180, 255) if person_info.get('contact_state') == 'unknown' else (0, 255, 0)
                        
                        # Рисуем рамку и ID
                        cv2.rectangle(enhanced, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(enhanced, f"ID: {global_id}", (x1, y1 - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Добавляем информацию о режиме
            mode_text = f"Mode: {self.current_mode.value}"
            cv2.putText(enhanced, mode_text, (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Ошибка улучшения визуализации: {e}")
            return pose_viz
    
    def _update_mode_counters(self):
        """Обновление счетчиков режимов"""
        if self.mode_counters["cooldown"] > 0:
            self.mode_counters["cooldown"] -= 1
    
    def _update_processing_mode(self):
        """Обновление режима обработки"""
        if self.mode_counters["cooldown"] > 0:
            return  # Период охлаждения
        
        current = self.current_mode
        
        if current == ProcessingMode.MOTION_DETECTION:
            # Переход к трекингу при стабильном движении
            if self.mode_counters["consecutive_motion"] >= self.mode_config["motion_detection_threshold"]:
                self.current_mode = ProcessingMode.POSE_TRACKING
                self.mode_counters["cooldown"] = self.mode_config["cooldown_frames"]
                self.mode_counters["consecutive_motion"] = 0
                self.logger.debug("Переключение: детекция движения -> трекинг поз")
        
        elif current == ProcessingMode.POSE_TRACKING:
            # Возврат к детекции движения при отсутствии треков
            if self.mode_counters["consecutive_no_tracks"] >= self.mode_config["tracking_timeout"]:
                self.current_mode = ProcessingMode.MOTION_DETECTION
                self.mode_counters["cooldown"] = self.mode_config["cooldown_frames"]
                self.mode_counters["consecutive_no_tracks"] = 0
                self.logger.debug("Переключение: трекинг поз -> детекция движения")
    
    def _add_result_to_queue(self, visualization, violations, timestamp):
        """Добавление результата в очередь"""
        try:
            violation_info = [
                {
                    "track_id": v['global_id'],
                    "bbox": v['bbox'],
                    "time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "confidence": v.get('confidence', 0.0)
                }
                for v in violations
            ]
            
            # Добавляем результат с блокировкой если очередь заполнена
            try:
                self.result_queue.put((visualization, self.current_mode.value, violation_info), block=False)
            except queue.Full:
                # Удаляем старый результат и добавляем новый
                try:
                    self.result_queue.get_nowait()
                    self.result_queue.put((visualization, self.current_mode.value, violation_info), block=False)
                except queue.Empty:
                    pass
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления результата в очередь: {e}")
    
    def _extract_timestamp_from_filename(self, filename: str) -> datetime.datetime:
        """Извлечение временной метки из имени файла"""
        try:
            # Формат: 'ВК3.5 2025-04-15 16-44-58_336.avi'
            parts = filename.split(' ')
            if len(parts) >= 3:
                date_str = parts[1]
                time_parts = parts[2].split('_')[0].split('-')
                time_str = ':'.join(time_parts)
                
                timestamp_str = f"{date_str} {time_str}"
                return datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                
        except Exception as e:
            self.logger.debug(f"Не удалось извлечь временную метку из {filename}: {e}")
        
        return datetime.datetime.now()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики обработки"""
        return {
            "processing_stats": {
                "frames_processed": self.stats.frames_processed,
                "motion_frames": self.stats.motion_frames,
                "tracking_frames": self.stats.tracking_frames,
                "avg_fps": self.stats.avg_fps,
                "processing_time": self.stats.processing_time
            },
            "personnel_stats": {
                "total_persons": self.stats.total_persons,
                "violation_persons": self.stats.total_violations,
                "active_records": len([p for p in self.personnel_records.values() if p.in_frame])
            },
            "current_mode": self.current_mode.value,
            "pose_tracker_stats": self.pose_tracker.get_statistics() if self.pose_tracker else {}
        }
    
    def cleanup(self):
        if not self.stop_processing():
            self.logger.warning("Поток еще останавливается; ресурсы остаются у него")
            return False
        if self.pose_tracker:
            self.pose_tracker.cleanup()
        self.video_recorder.cleanup()
        self._clear_queues()
        return True

    def _remember_recording(self, global_id, filename):
        self.recording_files[global_id] = filename
        for episode in self.episodes:
            if episode['global_id'] == global_id and 'violation_file' not in episode:
                episode['violation_file'] = filename
