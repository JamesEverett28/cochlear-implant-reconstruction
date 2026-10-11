"""Segment the first third of the resized crop from Label Studio AABB prompts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from electrode_segmentation.rendering import draw_mask
from electrode_segmentation.utils import get_checkpoint_and_config


JSON_PATH = Path("datasets/side1/label-studio.json")
VIDEO_PATH = Path("videos/3-view/angled-light/side1-crop-square-1000.mp4")
OUT_PATH = Path("out.mp4")
VIDEO_STRIDE = 5


def rotated_box_to_aabb(
    value: dict[str, float], frame_width: int, frame_height: int
) -> np.ndarray:
    """Convert a Label Studio rotated rectangle into a slightly padded AABB."""

    x = value["x"] / 100 * frame_width
    y = value["y"] / 100 * frame_height
    width = value["width"] / 100 * frame_width
    height = value["height"] / 100 * frame_height
    angle = np.deg2rad(value["rotation"])
    centre = np.array([x + width / 2, y + height / 2], dtype=np.float32)
    corners = np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ],
        dtype=np.float32,
    )
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    corners = corners @ rotation.T + centre
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    box_width = x1 - x0
    box_height = y1 - y0
    return np.array(
        [
            np.clip(x0 - 0.1 * box_width, 0, frame_width - 1),
            np.clip(y0 - 0.1 * box_height, 0, frame_height - 1),
            np.clip(x1 + 0.1 * box_width, 0, frame_width - 1),
            np.clip(y1 + 0.1 * box_height, 0, frame_height - 1),
        ],
        dtype=np.float32,
    )


def load_boxes(
    json_path: Path, frame_width: int, frame_height: int, final_frame: int
) -> dict[int, list[np.ndarray]]:
    """Load Label Studio boxes for every fifth source frame in the clip."""

    task = json.loads(json_path.read_text(encoding="utf-8"))[0]
    boxes: dict[int, list[np.ndarray]] = defaultdict(list)
    for result in task["annotations"][0]["result"]:
        if result["type"] != "videorectangle":
            continue
        for value in result["value"]["sequence"]:
            if not value.get("enabled", True):
                continue
            frame_idx = value["frame"] - 1
            if frame_idx > final_frame or frame_idx % VIDEO_STRIDE:
                continue
            boxes[frame_idx].append(
                rotated_box_to_aabb(value, frame_width, frame_height)
            )
    return dict(boxes)


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        clip_frame_count = (frame_count + 2) // 3
        boxes = load_boxes(
            JSON_PATH, frame_width, frame_height, clip_frame_count - 1
        )

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
            for frame_idx in range(0, clip_frame_count, VIDEO_STRIDE):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"Could not read frame {frame_idx}")

                frame_boxes = boxes.get(frame_idx)
                if frame_boxes:
                    predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    masks, _, _ = predictor.predict(
                        box=np.stack(frame_boxes), multimask_output=False
                    )
                    mask_stack = masks[:, 0] if masks.ndim == 4 else masks
                    draw_mask(frame, mask_stack.any(axis=0))

                out.write(frame)
                print(
                    f"Processed frame {frame_idx + 1}/{clip_frame_count}", end="\r"
                )
        finally:
            out.release()
    finally:
        cap.release()


if __name__ == "__main__":
    main()
