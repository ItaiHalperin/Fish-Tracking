import pytest
import shutil
import random
from pathlib import Path
from PIL import Image
import torch
from unittest import mock

from classifier.data import (
    _track_of,
    scan,
    build_vocab,
    split_by_track,
    split_report,
    FishCropDataset,
    FishRollDataset,
    weighted_sampler,
    _materialize,
    load_split_from_dir,
    Sample,
    Vocab
)
from classifier import labels as L

# Use fixtures for test data
@pytest.fixture
def dummy_dir(tmp_path):
    d = tmp_path / "dummy_data"
    d.mkdir()
    
    classes = ["regular_facing_right", "head_down_facing_left", "unclear_facing_right"]
    
    # Create some files
    files = [
        ("regular_facing_right", "video1__id_1_frame_0.jpg"),
        ("regular_facing_right", "video1__id_1_frame_1.jpg"),
        ("regular_facing_right", "video1__id_2_frame_0.jpg"),
        ("head_down_facing_left", "video1__id_1_frame_2.jpg"), # Same track as regular_facing_right (video1, 1)
        ("head_down_facing_left", "video2__id_1_frame_0.jpg"),
        ("unclear_facing_right", "video3__id_1_frame_0.jpg"),
        ("regular_facing_right", "random_image.jpg"), # No match
    ]
    
    for cls in classes:
        (d / cls).mkdir(parents=True, exist_ok=True)
        
    for cls, filename in files:
        img_path = d / cls / filename
        Image.new('RGB', (10, 10), 'red').save(img_path)
        
    return d

@pytest.fixture
def mock_samples(dummy_dir):
    return scan(dummy_dir, skip=["unclear_facing_right"])


def test_track_of_parses_standard_filename():
    track = _track_of(Path("video1__id_3_frame_42.jpg"))
    assert track == ("video1", "3")
    
def test_track_of_returns_stem_for_nonmatching():
    track = _track_of(Path("random_image.jpg"))
    assert track == ("random_image", "")

def test_scan_collects_samples_from_class_dirs(dummy_dir):
    samples = scan(dummy_dir, skip=[])
    assert len(samples) == 7
    # Verify sample fields
    for s in samples:
        assert hasattr(s, 'path')
        assert hasattr(s, 'composite')
        assert hasattr(s, 'pose')
        assert hasattr(s, 'facing')
        assert hasattr(s, 'track')

def test_scan_skips_excluded_classes(dummy_dir):
    samples = scan(dummy_dir, skip=["unclear_facing_right"])
    assert len(samples) == 6
    assert not any(s.composite == "unclear_facing_right" for s in samples)
    
def test_build_vocab_from_samples(mock_samples):
    vocab = build_vocab(mock_samples)
    assert len(vocab.poses) > 0
    assert len(vocab.facings) > 0
    assert len(vocab.composites) > 0
    assert "regular_facing_right" in vocab.composites
    assert "head_down_facing_left" in vocab.composites

def test_split_by_track_no_leak(mock_samples):
    splits = split_by_track(mock_samples, val_ratio=0.2, test_ratio=0.2, seed=42)
    
    train_tracks = {s.track for s in splits["train"]}
    val_tracks = {s.track for s in splits["val"]}
    test_tracks = {s.track for s in splits["test"]}
    
    assert train_tracks.isdisjoint(val_tracks)
    assert train_tracks.isdisjoint(test_tracks)
    assert val_tracks.isdisjoint(test_tracks)

def test_split_by_track_every_class_in_train(mock_samples):
    # Need to make sure we have enough tracks for testing this reliably
    # We will just duplicate mock_samples with different track IDs
    many_samples = []
    classes = ["regular_facing_right", "head_down_facing_left"]
    for cls in classes:
        for i in range(10): # 10 tracks per class
            many_samples.append(Sample(Path(f"v__id_{i}_frame_0.jpg"), cls, "pose", "facing", (f"v_{cls}", str(i))))
            
    splits = split_by_track(many_samples, val_ratio=0.2, test_ratio=0.2, seed=42)
    
    train_classes = {s.composite for s in splits["train"]}
    for cls in classes:
        assert cls in train_classes

def test_split_by_track_deterministic(mock_samples):
    splits1 = split_by_track(mock_samples, val_ratio=0.2, test_ratio=0.2, seed=42)
    splits2 = split_by_track(mock_samples, val_ratio=0.2, test_ratio=0.2, seed=42)
    
    for split in ["train", "val", "test"]:
        tracks1 = {s.track for s in splits1[split]}
        tracks2 = {s.track for s in splits2[split]}
        assert tracks1 == tracks2

def test_split_report_format(mock_samples):
    splits = split_by_track(mock_samples, val_ratio=0.2, test_ratio=0.2, seed=42)
    report = split_report(splits)
    
    assert "class" in report
    assert "train" in report
    assert "val" in report
    assert "test" in report
    assert "TOTAL" in report
    assert "regular_facing_right" in report

def test_weighted_sampler_returns_sampler(mock_samples):
    sampler = weighted_sampler(mock_samples, seed=42)
    assert isinstance(sampler, torch.utils.data.WeightedRandomSampler)
    assert len(sampler) == len(mock_samples)

def test_materialize_creates_output_structure(dummy_dir, tmp_path, mock_samples):
    out_dir = tmp_path / "out"
    splits = split_by_track(mock_samples, val_ratio=0.2, test_ratio=0.2, seed=42)
    
    summary = _materialize(splits, out_dir, train_cap=None, seed=42)
    
    assert out_dir.exists()
    assert (out_dir / "train").exists()
    
    loaded_samples = load_split_from_dir(out_dir, "train")
    assert len(loaded_samples) == len(splits["train"])
    
    # Check that train_cap works
    out_dir2 = tmp_path / "out2"
    _materialize(splits, out_dir2, train_cap=1, seed=42)
    loaded_capped = load_split_from_dir(out_dir2, "train")
    
    # At most 1 per class in train
    train_cap_counts = {}
    for s in loaded_capped:
        train_cap_counts[s.composite] = train_cap_counts.get(s.composite, 0) + 1
        
    for count in train_cap_counts.values():
        assert count <= 1

def dummy_transform(img):
    return "transformed"

def test_fish_crop_dataset(dummy_dir, mock_samples):
    vocab = build_vocab(mock_samples)
    dataset = FishCropDataset(mock_samples, vocab, dummy_transform, flip_p=0.0)
    
    assert len(dataset) == len(mock_samples)
    
    item = dataset[0]
    assert len(item) == 3
    assert item[0] == "transformed"
    assert isinstance(item[1], int)
    assert isinstance(item[2], int)

def test_fish_roll_dataset(dummy_dir, mock_samples):
    angle_map = {
        "regular_facing_right": (0.0, 0.0),
        "head_down_facing_left": (90.0, 0.0),
        "random_image": (0.0, 0.0)
    }
    # Add random_image to mock_samples composites to prevent KeyError
    dataset = FishRollDataset(mock_samples, angle_map, [0.0], dummy_transform, flip_p=0.0)
    
    assert len(dataset) == len(mock_samples)
    item = dataset[0]
    assert len(item) == 2
    assert item[0] == "transformed"
    assert isinstance(item[1], int)
