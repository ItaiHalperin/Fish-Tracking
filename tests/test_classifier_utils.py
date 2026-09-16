import pytest
import yaml
import torch
import numpy as np
from PIL import Image

from classifier.labels import (
    parse_composite, compose, flip_pose_lr, flip_facing_lr, flip_composite_lr
)
from classifier.angles import (
    roll_name, roll_vocab, snap_to_class, deg_to_vec, vec_to_deg,
    angular_loss, angular_error_deg, load_angle_map
)
from classifier.config import load_config, dump_config, AugConfig, TrainConfig
from classifier.imageio import to_pil

# --- labels.py ---

def test_parse_composite():
    pose, facing = parse_composite("curved_left_facing_away")
    assert pose == "curved_left"
    assert facing == "away"

def test_parse_composite_error():
    with pytest.raises(ValueError):
        parse_composite("not_a_composite")

def test_compose_roundtrip():
    pose, facing = "curved_left", "away"
    composite = compose(pose, facing)
    assert parse_composite(composite) == (pose, facing)

def test_flip_pose_lr():
    assert flip_pose_lr("curved_left") == "curved_right"
    assert flip_pose_lr("straight_right") == "straight_left"

def test_flip_pose_lr_neutral():
    assert flip_pose_lr("straight") == "straight"

def test_flip_facing_lr():
    assert flip_facing_lr("left") == "right"
    assert flip_facing_lr("right") == "left"
    assert flip_facing_lr("away") == "away"

def test_flip_composite_lr_roundtrip():
    original = "curved_left_facing_right"
    flipped = flip_composite_lr(original)
    assert flipped == "curved_right_facing_left"
    assert flip_composite_lr(flipped) == original

# --- angles.py ---

def test_roll_name_cardinals():
    assert roll_name(0) == "belly_down"
    assert roll_name(90) == "right_flank"
    assert roll_name(180) == "belly_up"
    assert roll_name(270) == "left_flank"

def test_roll_name_unknown():
    assert roll_name(45) == "roll_45"

def test_roll_vocab():
    angle_map = {
        "c1": (0, 90),
        "c2": (45, 180),
        "c3": (90, 90),
        "c4": (135, 0)
    }
    assert roll_vocab(angle_map) == [0.0, 90.0, 180.0]

def test_snap_to_class():
    angle_map = {
        "cls1": (0, 0),
        "cls2": (90, 180)
    }
    
    # exact match
    best, score = snap_to_class(0, 0, angle_map)
    assert best == "cls1"
    assert score == pytest.approx(1.0)
    
    # nearest
    best, score = snap_to_class(10, 10, angle_map)
    assert best == "cls1"
    assert score == pytest.approx(1.0 - 20/360)

def test_deg_vec_roundtrip():
    degs = torch.tensor([0.0, 45.0, 90.0, 180.0, 270.0, 359.0])
    vecs = deg_to_vec(degs)
    recovered = vec_to_deg(vecs)
    torch.testing.assert_close(recovered, degs)

def test_angular_loss():
    # perfect alignment
    pred = deg_to_vec(torch.tensor([45.0]))
    target = torch.tensor([45.0])
    assert angular_loss(pred, target).item() == pytest.approx(0.0, abs=1e-5)
    
    # opposite
    pred_opp = deg_to_vec(torch.tensor([45.0]))
    target_opp = torch.tensor([225.0])
    assert angular_loss(pred_opp, target_opp).item() == pytest.approx(2.0, abs=1e-5)

def test_angular_error_deg():
    # wraps correctly
    err = angular_error_deg(torch.tensor([359.0]), torch.tensor([1.0]))
    assert err.item() == pytest.approx(2.0)

def test_load_angle_map(tmp_path):
    yaml_content = """
cls_A:
    heading: 45
    roll: 90
cls_B:
    heading: 180
    roll: 0
"""
    p = tmp_path / "angle_map.yaml"
    p.write_text(yaml_content)
    am = load_angle_map(p)
    assert am == {"cls_A": (45.0, 90.0), "cls_B": (180.0, 0.0)}


# --- config.py ---

def test_config_defaults():
    ac = AugConfig()
    assert ac.degrees == 0.0
    tc = TrainConfig(name="test", model_type="multihead")
    assert tc.epochs == 50
    assert tc.imgsz == 224

def test_load_config(tmp_path):
    yaml_content = """
name: test_run
model_type: resnet18
epochs: 10
aug:
  degrees: 15.0
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    tc = load_config(p)
    assert tc.name == "test_run"
    assert tc.model_type == "resnet18"
    assert tc.epochs == 10
    assert tc.aug.degrees == 15.0

def test_dump_load_config_roundtrip(tmp_path):
    tc = TrainConfig(name="roundtrip", model_type="test", epochs=5)
    tc.aug.hsv_h = 0.5
    p = tmp_path / "dump.yaml"
    dump_config(tc, p)
    
    loaded = load_config(p)
    assert loaded.name == "roundtrip"
    assert loaded.epochs == 5
    assert loaded.aug.hsv_h == 0.5

# --- imageio.py ---

def test_to_pil_from_pil():
    img = Image.new("L", (10, 10))
    res = to_pil(img)
    assert isinstance(res, Image.Image)
    assert res.mode == "RGB"

def test_to_pil_from_path(tmp_path):
    p = tmp_path / "test.png"
    img = Image.new("RGB", (10, 10), color="red")
    img.save(p)
    
    res = to_pil(p)
    assert isinstance(res, Image.Image)
    assert res.size == (10, 10)

def test_to_pil_from_numpy():
    # BGR array
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[0, 0] = [255, 0, 0] # Blue in BGR
    
    res = to_pil(arr)
    assert isinstance(res, Image.Image)
    
    # Check that it got converted to RGB (first pixel should be [0, 0, 255] red)
    arr_res = np.array(res)
    assert np.array_equal(arr_res[0, 0], [0, 0, 255])
