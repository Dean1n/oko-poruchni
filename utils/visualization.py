import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

def draw_zone(frame: np.ndarray, 
             points: List[Tuple[int, int]], 
             color: Tuple[int, int, int], 
             thickness: int = 2,
             alpha: float = 0.3,
             label: Optional[str] = None,
             label_position: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Отрисовка зоны на кадре
    
    Args:
        frame: Кадр для отрисовки
        points: Список координат вершин зоны
        color: Цвет (BGR)
        thickness: Толщина линии
        alpha: Прозрачность заливки
        label: Текстовая метка для зоны
        label_position: Позиция для метки (x, y)
        
    Returns:
        Кадр с отрисованной зоной
    """
    if len(points) < 3:
        return frame
    
    result = frame.copy()
    overlay = result.copy()
    
    # Преобразуем точки в формат numpy
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Рисуем заполненный полигон на оверлее
    cv2.fillPoly(overlay, [pts], color)
    
    # Объединяем с исходным кадром с учетом прозрачности
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    
    # Рисуем контур
    cv2.polylines(result, [pts], True, color, thickness)
    
    # Добавляем метку, если указана
    if label:
        if label_position:
            label_x, label_y = label_position
        else:
            # Вычисляем центр полигона
            moments = cv2.moments(pts)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                # Если не удалось вычислить центр, берем первую точку
                cx, cy = points[0]
            label_x, label_y = cx, cy
        
        # Рисуем метку
        cv2.putText(result, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result

def draw_zones(frame: np.ndarray, 
              zones_config: Dict[str, Any], 
              show_roi: bool = True,
              show_stairs: bool = True,
              show_handrails: bool = True) -> np.ndarray:
    """
    Отрисовка всех зон из конфигурации
    
    Args:
        frame: Кадр для отрисовки
        zones_config: Конфигурация зон
        show_roi: Отображать зону интереса
        show_stairs: Отображать зоны лестниц
        show_handrails: Отображать зоны поручней
        
    Returns:
        Кадр с отрисованными зонами
    """
    result = frame.copy()
    
    # Настройки визуализации
    viz_config = zones_config.get("visualization", {})
    
    # Зона интереса (ROI)
    if show_roi and "roi_zone" in zones_config and zones_config["roi_zone"]:
        roi_color = viz_config.get("roi_color", (255, 0, 0))  # Красный по умолчанию
        roi_alpha = viz_config.get("roi_alpha", 0.3)
        
        result = draw_zone(
            result, 
            zones_config["roi_zone"], 
            roi_color, 
            2, 
            roi_alpha,
            "ROI"
        )
    
    # Лестницы и поручни
    if (show_stairs or show_handrails) and "stair_zones" in zones_config:
        stair_color = viz_config.get("stair_color", (128, 128, 128))  # Серый по умолчанию
        stair_alpha = viz_config.get("stair_alpha", 0.2)
        
        handrail_color = viz_config.get("handrail_color", (0, 255, 0))  # Зеленый по умолчанию
        handrail_alpha = viz_config.get("handrail_alpha", 0.4)
        
        for stair in zones_config["stair_zones"]:
            # Отрисовка лестницы
            if show_stairs and "coordinates" in stair:
                result = draw_zone(
                    result, 
                    stair["coordinates"], 
                    stair_color, 
                    2, 
                    stair_alpha,
                    f"Stair {stair.get('id', '')}"
                )
            
            # Отрисовка поручней
            if show_handrails and "handrails" in stair:
                for i, handrail in enumerate(stair["handrails"]):
                    if "coordinates" in handrail:
                        result = draw_zone(
                            result, 
                            handrail["coordinates"], 
                            handrail_color, 
                            2, 
                            handrail_alpha,
                            f"Handrail {handrail.get('id', '')}"
                        )
    
    return result

def draw_detection_box(frame: np.ndarray, 
                     bbox: Tuple[int, int, int, int], 
                     color: Tuple[int, int, int],
                     label: Optional[str] = None,
                     thickness: int = 2) -> np.ndarray:
    """
    Отрисовка прямоугольника детекции
    
    Args:
        frame: Кадр для отрисовки
        bbox: Координаты прямоугольника (x1, y1, x2, y2)
        color: Цвет (BGR)
        label: Текстовая метка
        thickness: Толщина линии
        
    Returns:
        Кадр с отрисованным прямоугольником
    """
    x1, y1, x2, y2 = bbox
    
    # Рисуем прямоугольник
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Добавляем метку, если указана
    if label:
        # Фон для текста
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        text_w, text_h = text_size
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 8, y1), color, -1)
        
        # Текст
        cv2.putText(frame, label, (x1 + 4, y1 - 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame

def draw_skeleton(frame: np.ndarray, 
                 keypoints: np.ndarray,
                 color_scheme: Dict[str, Tuple[int, int, int]] = None,
                 thickness: int = 2,
                 radius: int = 5,
                 min_confidence: float = 0.5) -> np.ndarray:
    """
    Отрисовка скелета позы человека
    
    Args:
        frame: Кадр для отрисовки
        keypoints: Массив ключевых точек [x, y, conf]
        color_scheme: Схема цветов для разных частей тела
        thickness: Толщина линий
        radius: Радиус точек
        min_confidence: Минимальная уверенность для отрисовки точки
        
    Returns:
        Кадр с отрисованным скелетом
    """
    if color_scheme is None:
        color_scheme = {
            'nose': (255, 0, 0),        # Синий
            'left_shoulder': (0, 255, 0),    # Зеленый
            'right_shoulder': (0, 0, 255),   # Красный
            'left_elbow': (0, 255, 0),       # Зеленый
            'right_elbow': (0, 0, 255),      # Красный
            'left_wrist': (0, 255, 0),       # Зеленый
            'right_wrist': (0, 0, 255),      # Красный
            'left_hip': (255, 255, 0),       # Желтый
            'right_hip': (255, 0, 255),      # Фиолетовый
            'left_knee': (255, 255, 0),      # Желтый
            'right_knee': (255, 0, 255),     # Фиолетовый
            'left_ankle': (255, 255, 0),     # Желтый
            'right_ankle': (255, 0, 255),    # Фиолетовый
        }
    
    # Индексы ключевых точек YOLO Pose
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear, 
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow, 
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip, 
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    
    # Связи между ключевыми точками для отрисовки скелета
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), # Голова
        (5, 6), # Плечи
        (5, 7), (7, 9), # Левая рука
        (6, 8), (8, 10), # Правая рука
        (5, 11), (6, 12), # Туловище
        (11, 12), # Бедра
        (11, 13), (13, 15), # Левая нога
        (12, 14), (14, 16) # Правая нога
    ]
    
    # Обозначения точек
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Получаем цвета для каждой точки
    colors = {}
    for i, name in enumerate(keypoint_names):
        # Если имя точки есть в схеме цветов, используем его
        # Иначе используем белый цвет
        colors[i] = color_scheme.get(name, (255, 255, 255))
    
    # Рисуем точки
    for i, kp in enumerate(keypoints):
        x, y, conf = kp
        if conf >= min_confidence:
            color = colors.get(i, (255, 255, 255))
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)
    
    # Рисуем связи (скелет)
    for pair in skeleton:
        idx1, idx2 = pair
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            pt1 = keypoints[idx1]
            pt2 = keypoints[idx2]
            
            if pt1[2] >= min_confidence and pt2[2] >= min_confidence:
                # Цвет линии - средний между цветами точек
                color1 = colors.get(idx1, (255, 255, 255))
                color2 = colors.get(idx2, (255, 255, 255))
                line_color = tuple(int((c1 + c2) / 2) for c1, c2 in zip(color1, color2))
                
                # Рисуем линию
                cv2.line(frame, 
                        (int(pt1[0]), int(pt1[1])), 
                        (int(pt2[0]), int(pt2[1])), 
                        line_color, thickness)
    
    return frame

def draw_status_text(frame: np.ndarray, 
                    text: str, 
                    position: Tuple[int, int] = (10, 30),
                    font_scale: float = 0.7,
                    color: Tuple[int, int, int] = (255, 255, 255),
                    thickness: int = 2,
                    with_background: bool = True) -> np.ndarray:
    """
    Отрисовка статусного текста
    
    Args:
        frame: Кадр для отрисовки
        text: Текст для отображения
        position: Позиция текста (x, y)
        font_scale: Масштаб шрифта
        color: Цвет текста
        thickness: Толщина линий текста
        with_background: Добавлять ли фон под текст
        
    Returns:
        Кадр с отрисованным текстом
    """
    # Вычисляем размер текста
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    x, y = position
    
    # Рисуем фон, если требуется
    if with_background:
        # Добавляем отступы
        padding = 5
        bg_rect = (
            x - padding, 
            y - text_height - padding - baseline, 
            text_width + 2 * padding, 
            text_height + 2 * padding + baseline
        )
        
        # Полупрозрачный фон
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (bg_rect[0], bg_rect[1]), 
            (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), 
            (0, 0, 0), 
            -1
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Рисуем текст
    cv2.putText(
        frame, text, position, 
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
    )
    
    return frame

def draw_violation_info(frame: np.ndarray, 
                       violations: List[Dict[str, Any]], 
                       position: Tuple[int, int] = (10, 60),
                       font_scale: float = 0.6) -> np.ndarray:
    """
    Отрисовка информации о нарушениях
    
    Args:
        frame: Кадр для отрисовки
        violations: Список нарушений
        position: Начальная позиция для отрисовки (x, y)
        font_scale: Масштаб шрифта
        
    Returns:
        Кадр с отрисованной информацией
    """
    if not violations:
        return frame
    
    x, y = position
    line_height = int(30 * font_scale)
    
    # Заголовок
    draw_status_text(
        frame, 
        f"Violations: {len(violations)}", 
        (x, y), 
        font_scale, 
        (0, 0, 255), 
        2, 
        True
    )
    
    y += line_height
    
    # Информация о каждом нарушении
    for v in violations:
        track_id = v.get("track_id", "unknown")
        violation_time = v.get("violation_time", 0.0)
        stair_id = v.get("stair_zone_id", "unknown")
        
        text = f"ID: {track_id}, Time: {violation_time:.1f}s, Stair: {stair_id}"
        draw_status_text(
            frame, 
            text, 
            (x, y), 
            font_scale, 
            (0, 165, 255), 
            1, 
            True
        )
        
        y += line_height
    
    return frame

def create_violation_thumbnail(frame: np.ndarray, 
                             violation: Dict[str, Any], 
                             size: Tuple[int, int] = (320, 240)) -> np.ndarray:
    """
    Создание миниатюры нарушения
    
    Args:
        frame: Кадр с нарушением
        violation: Информация о нарушении
        size: Размер миниатюры (ширина, высота)
        
    Returns:
        Миниатюра нарушения
    """
    if "bbox" not in violation:
        # Если нет информации о прямоугольнике, просто масштабируем кадр
        return cv2.resize(frame, size)
    
    # Расширяем прямоугольник для контекста
    x1, y1, x2, y2 = violation["bbox"]
    h, w = frame.shape[:2]
    
    # Расширяем на 20% в каждую сторону
    width = x2 - x1
    height = y2 - y1
    
    x1 = max(0, int(x1 - width * 0.2))
    y1 = max(0, int(y1 - height * 0.2))
    x2 = min(w, int(x2 + width * 0.2))
    y2 = min(h, int(y2 + height * 0.2))
    
    # Вырезаем и масштабируем
    roi = frame[y1:y2, x1:x2]
    
    # Добавляем красную рамку
    cv2.rectangle(roi, (0, 0), (roi.shape[1] - 1, roi.shape[0] - 1), (0, 0, 255), 3)
    
    # Добавляем информацию о нарушении
    track_id = violation.get("track_id", "unknown")
    violation_time = violation.get("violation_time", 0.0)
    text = f"ID: {track_id}, Time: {violation_time:.1f}s"
    
    draw_status_text(roi, text, (10, 30), 0.7, (0, 0, 255), 2, True)
    
    return cv2.resize(roi, size)

def create_mosaic(frames: List[np.ndarray], 
                 grid_size: Tuple[int, int] = (2, 2), 
                 frame_size: Tuple[int, int] = (320, 240)) -> np.ndarray:
    """
    Создание мозаики из нескольких кадров
    
    Args:
        frames: Список кадров
        grid_size: Размер сетки (строки, столбцы)
        frame_size: Размер каждого кадра (ширина, высота)
        
    Returns:
        Мозаика из кадров
    """
    rows, cols = grid_size
    width, height = frame_size
    
    # Создаем пустую мозаику
    mosaic = np.zeros((height * rows, width * cols, 3), dtype=np.uint8)
    
    # Заполняем мозаику кадрами
    for i, frame in enumerate(frames):
        if i >= rows * cols:
            break
            
        # Вычисляем позицию в мозаике
        row = i // cols
        col = i % cols
        
        # Масштабируем кадр
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        # Размещаем кадр в мозаике
        mosaic[row * height:(row + 1) * height, col * width:(col + 1) * width] = frame
    
    return mosaic

def overlay_transparent(background: np.ndarray, 
                       overlay: np.ndarray, 
                       x: int, 
                       y: int) -> np.ndarray:
    """
    Наложение изображения с прозрачностью
    
    Args:
        background: Фоновое изображение
        overlay: Накладываемое изображение (с альфа-каналом)
        x: Координата X
        y: Координата Y
        
    Returns:
        Результат наложения
    """
    # Фрагмент фона, на который будем накладывать
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]
    
    # Проверяем, что overlay полностью помещается на background
    if x >= bg_w or y >= bg_h or x + ov_w <= 0 or y + ov_h <= 0:
        return background
    
    # Обрезаем overlay, если он выходит за границы
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + ov_w)
    y2 = min(bg_h, y + ov_h)
    
    # Вычисляем смещение для overlay
    ov_x1 = x1 - x
    ov_y1 = y1 - y
    ov_x2 = ov_x1 + (x2 - x1)
    ov_y2 = ov_y1 + (y2 - y1)
    
    # Проверяем, что после обрезки остается что накладывать
    if ov_x1 >= ov_x2 or ov_y1 >= ov_y2:
        return background
    
    # Получаем альфа-канал (если есть)
    if overlay.shape[2] == 4:
        alpha = overlay[ov_y1:ov_y2, ov_x1:ov_x2, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        rgb = overlay[ov_y1:ov_y2, ov_x1:ov_x2, :3]
    else:
        # Если нет альфа-канала, просто копируем
        alpha = np.ones((y2 - y1, x2 - x1, 1))
        rgb = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
    
    # Выполняем наложение
    background[y1:y2, x1:x2] = background[y1:y2, x1:x2] * (1 - alpha) + rgb * alpha
    
    return background
