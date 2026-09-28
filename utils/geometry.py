import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Проверка, находится ли точка внутри полигона
    
    Args:
        point: Координаты точки (x, y)
        polygon: Список координат вершин полигона [(x1, y1), (x2, y2), ...]
        
    Returns:
        True, если точка внутри полигона, иначе False
    """
    if len(polygon) < 3:
        return False
        
    # Преобразуем в numpy массив для использования с OpenCV
    polygon_np = np.array(polygon, dtype=np.int32)
    point_np = (int(point[0]), int(point[1]))
    
    # Используем функцию OpenCV для проверки
    result = cv2.pointPolygonTest(polygon_np, point_np, False)
    return result >= 0

def get_polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """
    Вычисление площади полигона
    
    Args:
        polygon: Список координат вершин полигона [(x1, y1), (x2, y2), ...]
        
    Returns:
        Площадь полигона
    """
    if len(polygon) < 3:
        return 0.0
        
    # Формула площади Гаусса (метод шнуровки)
    area = 0.0
    j = len(polygon) - 1
    
    for i in range(len(polygon)):
        area += (polygon[j][0] + polygon[i][0]) * (polygon[j][1] - polygon[i][1])
        j = i
    
    return abs(area) / 2.0

def draw_polygon(frame: np.ndarray, 
                polygon: List[Tuple[int, int]], 
                color: Tuple[int, int, int], 
                thickness: int = 2,
                filled: bool = False,
                alpha: float = 0.3) -> np.ndarray:
    """
    Отрисовка полигона на кадре
    
    Args:
        frame: Кадр для отрисовки
        polygon: Список координат вершин полигона [(x1, y1), (x2, y2), ...]
        color: Цвет в формате BGR
        thickness: Толщина линии
        filled: Заполнять ли полигон цветом
        alpha: Прозрачность заполнения (0.0 - 1.0)
        
    Returns:
        Кадр с отрисованным полигоном
    """
    if len(polygon) < 3:
        return frame
    
    # Создаем копию входного кадра для предотвращения изменения оригинала
    result = frame.copy()
    
    # Преобразуем polygon в формат numpy для отрисовки
    polygon_np = np.array([polygon], dtype=np.int32)
    
    if filled:
        # Создаем отдельную маску для заполнения полигона
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, polygon_np, 255)
        
        # Создаем цветную маску
        color_mask = np.zeros_like(frame)
        color_mask[:] = color
        
        # Применяем заполнение только там, где маска не равна нулю
        colored_region = cv2.bitwise_and(color_mask, color_mask, mask=mask)
        
        # Смешиваем с оригинальным кадром с учетом прозрачности
        cv2.addWeighted(
            result, 1.0, 
            colored_region, alpha, 
            0, result
        )
    
    # Рисуем контур полигона
    cv2.polylines(result, polygon_np, True, color, thickness)
    
    return result

def filter_keypoints_in_zone(keypoints: List[Tuple[float, float, float]], 
                            zone: List[Tuple[int, int]]) -> List[Tuple[float, float, float]]:
    """
    Фильтрация ключевых точек, находящихся в заданной зоне
    
    Args:
        keypoints: Список ключевых точек [(x1, y1, conf1), (x2, y2, conf2), ...]
        zone: Список координат вершин зоны [(x1, y1), (x2, y2), ...]
        
    Returns:
        Список ключевых точек внутри зоны
    """
    if not zone or len(zone) < 3:
        return keypoints
    
    result = []
    for kp in keypoints:
        x, y, conf = kp
        if conf > 0 and point_in_polygon((x, y), zone):
            result.append(kp)
    
    return result

def get_distance_to_polygon(point: Tuple[float, float], 
                           polygon: List[Tuple[float, float]]) -> float:
    """
    Вычисление минимального расстояния от точки до полигона
    
    Args:
        point: Координаты точки (x, y)
        polygon: Список координат вершин полигона [(x1, y1), (x2, y2), ...]
        
    Returns:
        Минимальное расстояние до полигона (отрицательное, если точка внутри)
    """
    if len(polygon) < 3:
        return float('inf')
        
    # Преобразуем в numpy массив для использования с OpenCV
    polygon_np = np.array(polygon, dtype=np.int32)
    point_np = (float(point[0]), float(point[1]))
    
    # Используем функцию OpenCV для вычисления расстояния
    return cv2.pointPolygonTest(polygon_np, point_np, True)

def create_zone_mask(frame_shape: Tuple[int, int, int], 
                    polygons: List[List[Tuple[int, int]]],
                    colors: List[Tuple[int, int, int]],
                    alphas: List[float]) -> np.ndarray:
    """
    Создание маски для визуализации нескольких зон на кадре
    
    Args:
        frame_shape: Форма кадра (высота, ширина, каналы)
        polygons: Список полигонов [[(x1, y1), (x2, y2), ...], ...]
        colors: Список цветов для каждого полигона
        alphas: Список значений прозрачности для каждого полигона
        
    Returns:
        Маска для наложения на кадр
    """
    if not polygons:
        return np.zeros(frame_shape, dtype=np.uint8)
    
    mask = np.zeros(frame_shape, dtype=np.uint8)
    
    for i, polygon in enumerate(polygons):
        if i < len(colors) and i < len(alphas) and len(polygon) >= 3:
            color = colors[i]
            alpha = alphas[i]
            
            # Создаем отдельную маску для полигона
            polygon_mask = np.zeros_like(mask)
            polygon_np = np.array([polygon], dtype=np.int32)
            cv2.fillPoly(polygon_mask, polygon_np, color)
            
            # Накладываем на основную маску с учетом прозрачности
            cv2.addWeighted(mask, 1.0, polygon_mask, alpha, 0, mask)
    
    return mask

def apply_zone_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Применение маски зон к кадру
    
    Args:
        frame: Исходный кадр
        mask: Маска зон
        
    Returns:
        Кадр с наложенной маской зон
    """
    return cv2.addWeighted(frame, 1.0, mask, 1.0, 0)

def draw_zones_on_frame(frame: np.ndarray, 
                       zones_config: Dict[str, Any], 
                       roi_visible: bool = True,
                       stairs_visible: bool = True,
                       handrails_visible: bool = True) -> np.ndarray:
    """
    Отрисовка всех зон из конфигурации на кадре
    
    Args:
        frame: Кадр для отрисовки
        zones_config: Конфигурация зон
        roi_visible: Отображать зону интереса
        stairs_visible: Отображать зоны лестниц
        handrails_visible: Отображать зоны поручней
        
    Returns:
        Кадр с отрисованными зонами
    """
    # Создаем копию входного кадра для предотвращения изменения оригинала
    result = frame.copy()
    
    # Настройки визуализации
    viz_config = zones_config.get("visualization", {})
    
    # Зона интереса (ROI)
    if roi_visible and "roi_zone" in zones_config and zones_config["roi_zone"]:
        roi_zone = zones_config["roi_zone"]
        # Проверяем, что зона не пустая и содержит минимум 3 точки
        if roi_zone and len(roi_zone) >= 3:
            roi_color = viz_config.get("roi_color", (255, 0, 0))  # Красный по умолчанию
            roi_alpha = viz_config.get("roi_alpha", 0.3)
            
            result = draw_polygon(
                result, 
                roi_zone, 
                roi_color, 
                2, 
                True, 
                roi_alpha
            )
    
    # Лестницы и поручни
    if (stairs_visible or handrails_visible) and "stair_zones" in zones_config:
        stair_zones = zones_config["stair_zones"]
        # Проверка, что stair_zones - это список
        if isinstance(stair_zones, list):
            stair_color = viz_config.get("stair_color", (128, 128, 128))  # Серый по умолчанию
            stair_alpha = viz_config.get("stair_alpha", 0.2)
            
            handrail_color = viz_config.get("handrail_color", (0, 255, 0))  # Зеленый по умолчанию
            handrail_alpha = viz_config.get("handrail_alpha", 0.4)
            
            for stair in stair_zones:
                # Отрисовка лестницы
                if stairs_visible and "coordinates" in stair:
                    stair_coords = stair["coordinates"]
                    # Проверяем, что координаты не пустые и содержат минимум 3 точки
                    if stair_coords and len(stair_coords) >= 3:
                        result = draw_polygon(
                            result, 
                            stair_coords, 
                            stair_color, 
                            2, 
                            True, 
                            stair_alpha
                        )
                
                # Отрисовка поручней
                if handrails_visible and "handrails" in stair and isinstance(stair["handrails"], list):
                    for handrail in stair["handrails"]:
                        if "coordinates" in handrail:
                            handrail_coords = handrail["coordinates"]
                            # Проверяем, что координаты не пустые и содержат минимум 3 точки
                            if handrail_coords and len(handrail_coords) >= 3:
                                result = draw_polygon(
                                    result, 
                                    handrail_coords, 
                                    handrail_color, 
                                    2, 
                                    True, 
                                    handrail_alpha
                                )
    
    return result

def create_zone_editor_frame(frame: np.ndarray, 
                            current_polygon: List[Tuple[int, int]],
                            editing_mode: str = "roi") -> np.ndarray:
    """
    Создание кадра для редактора зон
    
    Args:
        frame: Исходный кадр
        current_polygon: Текущий редактируемый полигон
        editing_mode: Режим редактирования ('roi', 'stair', 'handrail')
        
    Returns:
        Кадр для редактора зон
    """
    result = frame.copy()
    
    # Определяем цвет в зависимости от режима редактирования
    if editing_mode == "roi":
        color = (255, 0, 0)  # Красный для зоны интереса
    elif editing_mode == "stair":
        color = (128, 128, 128)  # Серый для лестницы
    elif editing_mode == "handrail":
        color = (0, 255, 0)  # Зеленый для поручня
    else:
        color = (255, 255, 255)  # Белый для неизвестного режима
    
    # Рисуем текущий полигон
    if current_polygon and len(current_polygon) > 0:
        # Рисуем линии
        for i in range(len(current_polygon)):
            pt1 = current_polygon[i]
            pt2 = current_polygon[(i + 1) % len(current_polygon)]
            cv2.line(result, pt1, pt2, color, 2)
        
        # Рисуем точки
        for point in current_polygon:
            cv2.circle(result, point, 5, color, -1)
    
    # Добавляем подсказку
    text = f"Режим: {editing_mode}. Левый клик - добавить точку, Правый клик - удалить последнюю точку, Enter - завершить"
    cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result
