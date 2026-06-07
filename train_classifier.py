#!/usr/bin/env python3
"""
Train a 6-class fish position classifier on folder-labeled crops.

Expects this layout (ultralytics ImageFolder convention):

    labels/
      train/
        1_head_down/*.jpg
        2_head_up_diag/*.jpg
        ...
      val/
        1_head_down/*.jpg
        ...
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="labels", help="Root dir containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    model = YOLO("yolo11n-cls.pt")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name="fish_position_classifier",
        fliplr=0.0,
        flipud=0.0,
        degrees=10.0,
    )


if __name__ == "__main__":
    main()
