import cv2
import os
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any

def get_frame_at_position(video_path: str, position=0, position_type="frame") -> Optional[np.ndarray]:
    """
    Получение кадра из видео на указанной позиции
    
    Args:
        video_path: Путь к видеофайлу
        position: Позиция (номер кадра или время в секундах)
        position_type: Тип позиции ("frame" или "time")
        
    Returns:
        Кадр в виде numpy массива или None в случае ошибки
    """
    if not os.path.exists(video_path):
        return None
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        if position_type == "time":
            # Переводим время в секундах в номер кадра
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                return None
            position = int(position * fps)
        
        # Устанавливаем позицию
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        
        # Считываем кадр
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        
    except Exception as e:
        print(f"Ошибка при получении кадра: {e}")
    
    return None

def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Получение информации о видеофайле
    
    Args:
        video_path: Путь к видеофайлу
        
    Returns:
        Словарь с информацией о видео
    """
    info = {
        "exists": os.path.exists(video_path),
        "filename": os.path.basename(video_path),
        "size_bytes": 0,
        "width": 0,
        "height": 0,
        "fps": 0,
        "frame_count": 0,
        "duration_seconds": 0,
        "codec": ""
    }
    
    if not info["exists"]:
        return info
    
    # Получаем размер файла
    info["size_bytes"] = os.path.getsize(video_path)
    
    try:
        # Открываем видео для получения технической информации
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return info
        
        # Получаем параметры видео
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info["fps"] = cap.get(cv2.CAP_PROP_FPS)
        info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Вычисляем длительность
        if info["fps"] > 0:
            info["duration_seconds"] = info["frame_count"] / info["fps"]
        
        # Получаем кодек
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        info["codec"] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        # Освобождаем ресурсы
        cap.release()
        
    except Exception as e:
        print(f"Ошибка при получении информации о видео: {e}")
    
    return info

def convert_video(input_path: str, output_path: str, 
                 codec: str = 'XVID', 
                 fps: Optional[float] = None,
                 resolution: Optional[Tuple[int, int]] = None) -> bool:
    """
    Конвертация видео в другой формат или с другими параметрами
    
    Args:
        input_path: Путь к исходному видео
        output_path: Путь для сохранения результата
        codec: Кодек в виде четырех символов
        fps: Частота кадров (если None, используется исходная)
        resolution: Разрешение (ширина, высота) (если None, используется исходное)
        
    Returns:
        True, если конвертация успешна, иначе False
    """
    if not os.path.exists(input_path):
        return False
    
    try:
        # Создаем директорию для выходного файла, если её нет
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Открываем исходное видео
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        
        # Получаем параметры исходного видео
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Устанавливаем параметры выходного видео
        out_width, out_height = resolution if resolution else (orig_width, orig_height)
        out_fps = fps if fps else orig_fps
        
        # Создаем объект для записи
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, out_fps, (out_width, out_height))
        
        if not out.isOpened():
            cap.release()
            return False
        
        # Конвертируем видео
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Изменяем разрешение, если нужно
            if resolution and (orig_width != out_width or orig_height != out_height):
                frame = cv2.resize(frame, (out_width, out_height))
            
            # Записываем кадр
            out.write(frame)
        
        # Освобождаем ресурсы
        cap.release()
        out.release()
        
        return os.path.exists(output_path)
        
    except Exception as e:
        print(f"Ошибка при конвертации видео: {e}")
        return False

def extract_frames(video_path: str, 
                  output_dir: str,
                  interval: Union[int, float] = 1.0,
                  interval_type: str = "seconds",
                  max_frames: Optional[int] = None,
                  filename_template: str = "frame_{:04d}.jpg",
                  image_format: str = "jpg",
                  image_quality: int = 95) -> List[str]:
    """
    Извлечение кадров из видео
    
    Args:
        video_path: Путь к видеофайлу
        output_dir: Директория для сохранения кадров
        interval: Интервал между кадрами (в секундах или кадрах)
        interval_type: Тип интервала ("seconds" или "frames")
        max_frames: Максимальное количество кадров для извлечения
        filename_template: Шаблон имени файла
        image_format: Формат изображения (jpg, png)
        image_quality: Качество изображения (для JPEG)
        
    Returns:
        Список путей к сохраненным кадрам
    """
    if not os.path.exists(video_path):
        return []
    
    # Создаем директорию для выходных файлов, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    saved_frames = []
    
    try:
        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        # Получаем параметры видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Вычисляем интервал в кадрах
        if interval_type == "seconds":
            frame_interval = int(fps * interval)
        else:
            frame_interval = int(interval)
        
        # Ограничиваем количество кадров, если указано
        if max_frames:
            frames_to_extract = min(max_frames, total_frames // frame_interval + 1)
        else:
            frames_to_extract = total_frames // frame_interval + 1
        
        # Извлекаем кадры
        frame_count = 0
        saved_count = 0
        
        while saved_count < frames_to_extract:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Формируем имя файла
                output_path = os.path.join(output_dir, filename_template.format(saved_count))
                
                # Сохраняем кадр
                if image_format.lower() == "jpg" or image_format.lower() == "jpeg":
                    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, image_quality])
                elif image_format.lower() == "png":
                    cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, min(9, (100 - image_quality) // 10)])
                else:
                    cv2.imwrite(output_path, frame)
                
                saved_frames.append(output_path)
                saved_count += 1
            
            frame_count += 1
        
        # Освобождаем ресурсы
        cap.release()
        
    except Exception as e:
        print(f"Ошибка при извлечении кадров: {e}")
    
    return saved_frames

def create_video_from_images(images: List[str], 
                           output_path: str,
                           fps: float = 30.0,
                           codec: str = 'XVID',
                           resolution: Optional[Tuple[int, int]] = None) -> bool:
    """
    Создание видео из набора изображений
    
    Args:
        images: Список путей к изображениям
        output_path: Путь для сохранения видео
        fps: Частота кадров
        codec: Кодек в виде четырех символов
        resolution: Разрешение (ширина, высота) (если None, используется разрешение первого изображения)
        
    Returns:
        True, если создание успешно, иначе False
    """
    if not images:
        return False
    
    try:
        # Создаем директорию для выходного файла, если её нет
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Получаем разрешение первого изображения, если не указано
        if not resolution:
            img = cv2.imread(images[0])
            if img is None:
                return False
            height, width = img.shape[:2]
            resolution = (width, height)
        
        # Создаем объект для записи
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        
        if not out.isOpened():
            return False
        
        # Добавляем каждое изображение в видео
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Изменяем разрешение, если нужно
            if img.shape[:2] != (resolution[1], resolution[0]):
                img = cv2.resize(img, resolution)
            
            # Записываем кадр
            out.write(img)
        
        # Освобождаем ресурсы
        out.release()
        
        return os.path.exists(output_path)
        
    except Exception as e:
        print(f"Ошибка при создании видео из изображений: {e}")
        return False
