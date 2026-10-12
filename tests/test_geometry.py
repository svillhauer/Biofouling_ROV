import math

import pytest

from biofouling.geometry import covered_area_m2, frame_footprint_m2, polygon_area_px


def test_unit_square_covers_whole_frame():
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert polygon_area_px(square, 1920, 1080) == pytest.approx(1920 * 1080)


def test_polygon_area_is_orientation_independent():
    tri_ccw = [(0, 0), (0.5, 0), (0, 0.5)]
    assert polygon_area_px(tri_ccw, 100, 100) == pytest.approx(polygon_area_px(tri_ccw[::-1], 100, 100))
    assert polygon_area_px(tri_ccw, 100, 100) == pytest.approx(0.125 * 100 * 100)


@pytest.mark.parametrize("points", [None, [], [(0, 0), (1, 1)]])
def test_degenerate_polygon_has_no_area(points):
    assert polygon_area_px(points, 640, 480) is None


def test_footprint_at_90_degree_fov_is_2r_squared():
    # tan(45 deg) == 1, so each side is 2 * range
    assert frame_footprint_m2(1.5, 90, 90) == pytest.approx(3.0 * 3.0)


def test_footprint_scales_with_range_squared():
    near = frame_footprint_m2(1.0, 80, 64)
    far = frame_footprint_m2(2.0, 80, 64)
    assert far == pytest.approx(4 * near)


def test_covered_area_is_fraction_of_footprint():
    footprint = frame_footprint_m2(2.0, 80, 64)
    assert covered_area_m2(25.0, 2.0, 80, 64) == pytest.approx(0.25 * footprint)
    assert covered_area_m2(0.0, 2.0, 80, 64) == 0.0
    assert math.isclose(covered_area_m2(100.0, 2.0, 80, 64), footprint)
