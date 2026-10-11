"""Track all annotated electrodes as one SAM2 object using rasterized OBB masks."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.memory_attention import MemoryAttention

from electrode_segmentation.rendering import draw_mask
from electrode_segmentation.utils import get_checkpoint_and_config
from utils.video import extract_frames


JSON_PATH = Path("datasets/side1/label-studio.json")
VIDEO_PATH = Path("videos/3-view/angled-light/side1-crop.mp4")
OUT_PATH = Path("out.mp4")
VIDEO_STRIDE = 5
PROMPT_EVERY_EXTRACTED_FRAMES = 4
MAX_OBJECTS = 5
OBJECT_ID = 0


_ORIGINAL_MEMORY_ATTENTION_FORWARD = MemoryAttention.forward


def _mps_float32_memory_attention(
    self: MemoryAttention,
    curr: torch.Tensor | list[torch.Tensor],
    memory: torch.Tensor,
    curr_pos: torch.Tensor | list[torch.Tensor] | None = None,
    memory_pos: torch.Tensor | None = None,
    num_obj_ptr_tokens: int = 0,
):
    """Work around SAM 2's bfloat16-memory / float32-query MPS mismatch."""

    dtype = curr[0].dtype if isinstance(curr, list) else curr.dtype
    if memory.dtype != dtype:
        memory = memory.to(dtype=dtype)
    if memory_pos is not None and memory_pos.dtype != dtype:
        memory_pos = memory_pos.to(dtype=dtype)

    return _ORIGINAL_MEMORY_ATTENTION_FORWARD(
        self,
        curr,
        memory,
        curr_pos=curr_pos,
        memory_pos=memory_pos,
        num_obj_ptr_tokens=num_obj_ptr_tokens,
    )


MemoryAttention.forward = _mps_float32_memory_attention


def rotated_box_to_polygon(
    value: dict[str, float],
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Convert a Label Studio rotated rectangle into a pixel polygon."""

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
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float32,
    )
    corners = corners @ rotation.T + centre
    corners[:, 0] = np.clip(corners[:, 0], 0, frame_width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, frame_height - 1)
    return np.rint(corners).astype(np.int32)


def load_prompt_masks(
    json_path: Path,
    frame_width: int,
    frame_height: int,
    final_source_frame: int,
) -> dict[int, np.ndarray]:
    """Return union OBB masks keyed by source frame for one SAM2 object."""

    task = json.loads(json_path.read_text(encoding="utf-8"))[0]
    polygons_by_frame: dict[int, list[np.ndarray]] = defaultdict(list)
    rectangle_results = [
        result
        for result in task["annotations"][0]["result"]
        if result["type"] == "videorectangle"
    ]
    prompt_source_stride = VIDEO_STRIDE * PROMPT_EVERY_EXTRACTED_FRAMES

    for result in rectangle_results[:MAX_OBJECTS]:
        for value in result["value"]["sequence"]:
            if not value.get("enabled", True):
                continue

            source_frame = value["frame"] - 1
            if source_frame > final_source_frame:
                continue
            if source_frame % prompt_source_stride != 0:
                continue

            polygons_by_frame[source_frame].append(
                rotated_box_to_polygon(value, frame_width, frame_height)
            )

    masks: dict[int, np.ndarray] = {}
    for source_frame, polygons in polygons_by_frame.items():
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, polygons, color=1)
        masks[source_frame] = mask.astype(bool)
    return masks


def write_first_third(
    cap: cv2.VideoCapture,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    fps: float,
    clip_path: Path,
) -> int:
    """Write the first third of ``cap`` to ``clip_path`` and return its length."""

    first_third_frames = (frame_count + 2) // 3
    writer = cv2.VideoWriter(
        str(clip_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create temporary clip: {clip_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    try:
        for _ in range(first_third_frames):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Could not read the requested first third of the video")
            writer.write(frame)
    finally:
        writer.release()

    return first_third_frames


def masks2mask(
    cap: cv2.VideoCapture,
    out_path: str | Path,
    masks: dict[int, np.ndarray],
    stride: int,
    model_size: str = "small",
) -> None:
    """Track the union masks as one object and render the resulting segmentation."""

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / stride,
        (frame_width, frame_height),
    )
    if not out.isOpened():
        raise RuntimeError(f"Could not open output video: {out_path}")

    checkpoint, model_cfg = get_checkpoint_and_config(model_size)
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device)

    try:
        with TemporaryDirectory() as frame_dir:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            extract_frames(cap, frame_dir, stride)
            inference_state = predictor.init_state(
                video_path=frame_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=True,
            )

            for source_frame, mask in masks.items():
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=source_frame // stride,
                    obj_id=OBJECT_ID,
                    mask=mask,
                )

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for frame_idx, _, logits in predictor.propagate_in_video(inference_state):
                source_frame_idx = frame_idx * stride
                cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"Could not read frame {source_frame_idx}")

                draw_mask(frame, (logits[0, 0] > 0).cpu().numpy())
                out.write(frame)
    finally:
        out.release()


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        with TemporaryDirectory() as temp_dir:
            clip_path = Path(temp_dir) / "side1-crop-first-third.mp4"
            first_third_frames = write_first_third(
                cap, frame_count, frame_width, frame_height, fps, clip_path
            )
            masks = load_prompt_masks(
                JSON_PATH, frame_width, frame_height, first_third_frames - 1
            )
            if not masks:
                raise RuntimeError("No Label Studio OBBs matched the requested prompt frames")

            print(
                f"Processing {first_third_frames} source frames with "
                f"{len(masks)} union-mask prompts for one object."
            )
            clip_cap = cv2.VideoCapture(clip_path)
            if not clip_cap.isOpened():
                raise RuntimeError(f"Could not open temporary clip: {clip_path}")
            try:
                masks2mask(clip_cap, OUT_PATH, masks, VIDEO_STRIDE)
            finally:
                clip_cap.release()
    finally:
        cap.release()


if __name__ == "__main__":
    main()
