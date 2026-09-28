from .video_utils import get_frame_at_position

# Импортируем основные утилиты, если файлы существуют
try:
    from .geometry import point_in_polygon, draw_polygon, draw_zones_on_frame
except ImportError:
    pass