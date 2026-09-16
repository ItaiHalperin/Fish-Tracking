import pytest
import numpy as np
import torch
from unittest.mock import patch
from pathlib import Path

from core.device import get_device
from core.image_utils import crop_with_padding
from core.video_utils import find_video_files

def test_get_device_returns_string():
    device = get_device()
    assert device in ('mps', 'cuda', 'cpu')

@patch('torch.backends.mps.is_available', return_value=True)
def test_get_device_with_mps_mocked(mock_mps):
    assert get_device() == 'mps'

@patch('torch.backends.mps.is_available', return_value=False)
@patch('torch.cuda.is_available', return_value=True)
def test_get_device_with_cuda_mocked(mock_cuda, mock_mps):
    assert get_device() == 'cuda'

@patch('torch.backends.mps.is_available', return_value=False)
@patch('torch.cuda.is_available', return_value=False)
def test_get_device_cpu_fallback(mock_cuda, mock_mps):
    assert get_device() == 'cpu'

def test_crop_with_padding_basic():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = (40, 40, 60, 60) # w=20, h=20, pad=0.1 -> pad_w=2, pad_h=2 -> (38, 38, 62, 62)
    crop = crop_with_padding(image, box, padding=0.10)
    assert crop.shape == (24, 24, 3)

def test_crop_with_padding_zero_padding():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = (40, 40, 60, 60)
    crop = crop_with_padding(image, box, padding=0.0)
    assert crop.shape == (20, 20, 3)

def test_crop_with_padding_clamped_to_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = (0, 0, 10, 10)
    crop = crop_with_padding(image, box, padding=0.5) # pad=5 -> cx1=0, cy1=0, cx2=15, cy2=15
    assert crop.shape == (15, 15, 3)
    
    box = (90, 90, 100, 100)
    crop = crop_with_padding(image, box, padding=0.5) # pad=5 -> cx1=85, cy1=85, cx2=100, cy2=100
    assert crop.shape == (15, 15, 3)

def test_crop_with_padding_empty_crop_with_min_size():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = (200, 200, 210, 210) # outside image
    crop = crop_with_padding(image, box, padding=0.1, min_size=(32, 32))
    assert crop.shape == (32, 32, 3)
    assert np.all(crop == 0)

def test_crop_with_padding_empty_crop_without_min_size():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = (200, 200, 210, 210)
    crop = crop_with_padding(image, box, padding=0.1)
    assert crop.size == 0

def test_crop_with_padding_full_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = (0, 0, 100, 100)
    crop = crop_with_padding(image, box, padding=0.1)
    assert crop.shape == (100, 100, 3)

def test_find_video_files_finds_supported_extensions(tmp_path):
    (tmp_path / "vid1.mp4").touch()
    (tmp_path / "vid2.avi").touch()
    (tmp_path / "vid3.mov").touch()
    files = find_video_files(tmp_path)
    assert len(files) == 3
    assert {f.name for f in files} == {"vid1.mp4", "vid2.avi", "vid3.mov"}

def test_find_video_files_ignores_non_video(tmp_path):
    (tmp_path / "vid1.mp4").touch()
    (tmp_path / "doc.txt").touch()
    (tmp_path / "img.jpg").touch()
    files = find_video_files(tmp_path)
    assert len(files) == 1
    assert files[0].name == "vid1.mp4"

def test_find_video_files_empty_dir(tmp_path):
    files = find_video_files(tmp_path)
    assert files == []

def test_find_video_files_case_insensitive(tmp_path):
    (tmp_path / "vid1.MP4").touch()
    (tmp_path / "vid2.Avi").touch()
    files = find_video_files(tmp_path)
    assert len(files) == 2
    assert {f.name for f in files} == {"vid1.MP4", "vid2.Avi"}
