import numpy as np

from biofouling.yolo_format import parse_yolo_seg_line, to_yolo_seg_line


def test_roundtrip_preserves_polygon():
    poly = np.array([[10.0, 20.0], [300.0, 40.0], [150.0, 220.0]])
    line = to_yolo_seg_line(poly, width=640, height=480, class_id=3)
    assert line.startswith("3 ")
    np.testing.assert_allclose(parse_yolo_seg_line(line, 640, 480), poly, atol=1e-2)


def test_coords_are_normalized_and_clamped():
    line = to_yolo_seg_line([(-5, 0), (700, 240), (320, 999)], width=640, height=480, class_id=0)
    coords = [float(v) for v in line.split()[1:]]
    assert len(coords) == 6
    assert all(0.0 <= c <= 1.0 for c in coords)
    assert coords[0] == 0.0 and coords[2] == 1.0 and coords[5] == 1.0
