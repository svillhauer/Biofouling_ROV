"""Pixel/metric geometry for turning segmentation masks into coverage figures."""
import math


def polygon_area_px(points_norm, image_width, image_height):
    """Shoelace area of a polygon given in normalized [0, 1] coords, in pixels^2.

    Returns None for degenerate polygons (fewer than 3 points).
    """
    if not points_norm or len(points_norm) < 3:
        return None
    area = 0.0
    n = len(points_norm)
    for i in range(n):
        x1n, y1n = points_norm[i]
        x2n, y2n = points_norm[(i + 1) % n]
        area += x1n * y2n - x2n * y1n
    return abs(area) / 2 * image_width * image_height


def frame_footprint_m2(range_m: float, hfov_deg: float, vfov_deg: float) -> float:
    """Area of wall (m^2) visible to a pinhole camera facing it head-on at ``range_m``."""
    width_m = 2 * range_m * math.tan(math.radians(hfov_deg / 2))
    height_m = 2 * range_m * math.tan(math.radians(vfov_deg / 2))
    return width_m * height_m


def covered_area_m2(percent_coverage: float, range_m: float, hfov_deg: float, vfov_deg: float) -> float:
    """Convert a percent-of-frame coverage into absolute area on the wall (m^2)."""
    return (percent_coverage / 100) * frame_footprint_m2(range_m, hfov_deg, vfov_deg)
