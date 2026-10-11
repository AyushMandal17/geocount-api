import math

from app.engine import Detection, count_by_class, rotated_iou, rotated_nms, tile_origins
from app.query import format_answer, parse_query


def det(cx, cy, conf=0.9, cls=0, name="plane", w=40.0, h=20.0):
    return Detection(cls, name, conf, (cx, cy, w, h, 0.3))


def test_tiles_cover_edge():
    s = tile_origins(2000, 1024, 0.2)
    assert s[0] == 0 and s[-1] + 1024 == 2000
    assert tile_origins(900, 1024, 0.2) == [0]


def test_iou():
    assert math.isclose(rotated_iou(det(100, 100).xywhr, det(100, 100).xywhr), 1.0, rel_tol=1e-3)
    assert rotated_iou(det(0, 0).xywhr, det(500, 500).xywhr) == 0.0


def test_nms_duplicates_distinct_and_classwise():
    assert len(rotated_nms([det(100, 100, 0.9), det(101, 100, 0.8), det(300, 300, 0.7)])) == 2
    assert len(rotated_nms([det(100, 100, 0.9, 0, "plane"), det(100, 100, 0.8, 1, "ship")])) == 2


def test_nms_drops_fragment_inside_big_box():
    big = det(100, 100, 0.9, w=100.0, h=100.0)
    frag = det(100, 100, 0.5, w=30.0, h=30.0)  # IoU tiny, but fully inside
    assert len(rotated_nms([big, frag])) == 1


def test_parse_and_answers():
    assert parse_query("How many cars are there?") == ["small vehicle"]
    assert set(parse_query("count the vehicles")) == {"small vehicle", "large vehicle"}
    assert parse_query("how many large vehicles?") == ["large vehicle"]
    assert parse_query("what is this") is None
    assert format_answer({"ship": 1}) == "There is 1 ship."
    assert format_answer({"ship": 14}) == "There are 14 ships."
    assert format_answer({"ship": 0}) == "No ships detected."
    dets = [det(1, 1), det(2, 2)]
    assert count_by_class(dets) == {"plane": 2}
    assert count_by_class(dets, ["plane", "ship"]) == {"plane": 2, "ship": 0}
