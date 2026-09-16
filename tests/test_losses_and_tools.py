import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter
import subprocess
import sys
import shutil

from classifier.losses import (
    class_balanced_weights,
    counts_for,
    FocalLoss,
    build_loss
)

def test_class_balanced_weights_sums_to_num_classes():
    counts = [10, 100, 1000]
    weights = class_balanced_weights(counts, beta=0.99)
    assert weights.sum().item() == pytest.approx(len(counts))

def test_class_balanced_weights_equal_counts_gives_uniform():
    counts = [50, 50, 50]
    weights = class_balanced_weights(counts, beta=0.99)
    assert torch.allclose(weights, torch.ones(3))

def test_class_balanced_weights_rare_class_gets_higher_weight():
    counts = [10, 1000]
    weights = class_balanced_weights(counts, beta=0.99)
    assert weights[0] > weights[1]

def test_class_balanced_weights_beta_zero_gives_uniform():
    counts = [1, 10, 100, 1000]
    weights = class_balanced_weights(counts, beta=0.0)
    assert torch.allclose(weights, torch.ones(4))

def test_counts_for_matches_counter():
    labels = ["cat", "dog", "cat", "bird", "bird", "bird"]
    vocab = ["bird", "cat", "dog"]
    counts = counts_for(labels, vocab)
    assert counts == [3, 2, 1]

def test_counts_for_missing_class_is_zero():
    labels = ["cat", "dog"]
    vocab = ["cat", "dog", "bird"]
    counts = counts_for(labels, vocab)
    assert counts == [1, 1, 0]

def test_focal_loss_runs_forward():
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 3)
    target = torch.tensor([0, 1, 2, 0])
    loss = loss_fn(logits, target)
    assert loss.dim() == 0
    assert loss.item() >= 0

def test_focal_loss_gamma_zero_equals_ce():
    logits = torch.randn(4, 3)
    target = torch.tensor([0, 1, 2, 0])
    loss_fn = FocalLoss(gamma=0.0)
    focal = loss_fn(logits, target)
    ce = F.cross_entropy(logits, target)
    assert focal.item() == pytest.approx(ce.item())

def test_focal_loss_gradient_flows():
    logits = torch.randn(4, 3, requires_grad=True)
    target = torch.tensor([0, 1, 2, 0])
    loss_fn = FocalLoss(gamma=2.0)
    loss = loss_fn(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)

def test_build_loss_ce_returns_cross_entropy():
    loss_fn = build_loss("ce", weight=None, gamma=2.0)
    assert isinstance(loss_fn, nn.CrossEntropyLoss)

def test_build_loss_focal_returns_focal():
    loss_fn = build_loss("focal", weight=None, gamma=2.0)
    assert isinstance(loss_fn, FocalLoss)
    assert loss_fn.gamma == 2.0

def test_build_loss_unknown_raises():
    with pytest.raises(ValueError, match="Unknown loss kind"):
        build_loss("magic", weight=None, gamma=2.0)

# Tests for flatten_crops.py

@pytest.fixture
def dummy_crops_dir(tmp_path):
    crops_dir = tmp_path / "crops" / "vid1"
    crops_dir.mkdir(parents=True)
    
    id1_dir = crops_dir / "id_1"
    id1_dir.mkdir()
    (id1_dir / "frame_0001.jpg").touch()
    (id1_dir / "frame_0002.jpg").touch()
    
    id2_dir = crops_dir / "id_2"
    id2_dir.mkdir()
    (id2_dir / "frame_0001.jpg").touch()
    
    return crops_dir

def run_flatten_crops(crops_dir, output_dir=None, move=False):
    script_path = "dataset_tools/flatten_crops.py"
    cmd = [sys.executable, script_path, "--crops-dir", str(crops_dir)]
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    if move:
        cmd.append("--move")
        
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with output: {result.stderr}"

def test_flatten_copies_crops(dummy_crops_dir):
    run_flatten_crops(dummy_crops_dir)
    
    flat_dir = dummy_crops_dir.parent / "vid1_flat"
    assert flat_dir.exists()
    
    files = list(flat_dir.glob("*.jpg"))
    assert len(files) == 3
    
    # Original files should still exist
    assert (dummy_crops_dir / "id_1" / "frame_0001.jpg").exists()

def test_flatten_naming_convention(dummy_crops_dir):
    flat_dir = dummy_crops_dir.parent / "vid1_flat"
    run_flatten_crops(dummy_crops_dir, output_dir=flat_dir, move=True)
    
    expected_files = {
        "id_1_frame_0001.jpg",
        "id_1_frame_0002.jpg",
        "id_2_frame_0001.jpg"
    }
    actual_files = {p.name for p in flat_dir.glob("*.jpg")}
    assert actual_files == expected_files
    
    # Files should be moved, not copied
    assert not (dummy_crops_dir / "id_1" / "frame_0001.jpg").exists()

