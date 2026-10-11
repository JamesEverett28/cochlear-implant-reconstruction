"""Prompt SAM2 independently on sampled frames with detected electrode centres."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

from electrode_segmentation.rendering import draw_mask
from electrode_segmentation.utils import get_checkpoint_and_config


VIDEO_PATH = Path("videos/3-view/angled-light/side1-crop.mp4")
DETECTOR_PATH = Path("experiments/side1/base/weights/best.pt")
OUT_PATH = Path("out.mp4")
VIDEO_STRIDE = 5
CONFIDENCE = 0.25
# When enabled, SAM2 receives abs(current_frame - first_frame); output stays original.
USE_FRAME_DIFFERENCE = True


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        detector = YOLO(DETECTOR_PATH)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        checkpoint, model_cfg = get_checkpoint_and_config("large")
        predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))

        out = cv2.VideoWriter(
            str(OUT_PATH),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps / VIDEO_STRIDE,
            (frame_width, frame_height),
        )
        if not out.isOpened():
            raise RuntimeError(f"Could not open output video: {OUT_PATH}")

        try:
            reference_frame: np.ndarray | None = None
            for frame_idx in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"Could not read frame {frame_idx}")
                if reference_frame is None:
                    reference_frame = frame.copy()
                if frame_idx % VIDEO_STRIDE:
                    continue

                detections = detector(frame, conf=CONFIDENCE, verbose=False)[0].obb
                if len(detections):
                    points = detections.xywhr.cpu().numpy()[:, :2]
                    sam2_frame = (
                        cv2.absdiff(frame, reference_frame)
                        if USE_FRAME_DIFFERENCE
                        else frame
                    )
                    predictor.set_image(cv2.cvtColor(sam2_frame, cv2.COLOR_BGR2RGB))
                    masks, _, _ = predictor.predict(
                        point_coords=points,
                        point_labels=np.ones(len(points), dtype=np.int32),
                        multimask_output=False,
                    )
                    draw_mask(frame, masks.any(axis=0))

                out.write(frame)
                print(f"Processed frame {frame_idx + 1}/{frame_count}", end="\r")
        finally:
            out.release()
    finally:
        cap.release()


if __name__ == "__main__":
    main()
