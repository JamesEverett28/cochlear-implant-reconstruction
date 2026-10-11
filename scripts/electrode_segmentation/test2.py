"""Segment every labelled electrode in independent 30-frame SAM2 windows."""

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


JSON_PATH = Path("datasets/side1/label-studio.json")
VIDEO_PATH = Path("videos/3-view/angled-light/side1-crop.mp4")
OUT_PATH = Path("out.mp4")
VIDEO_STRIDE = 5
BLOCK_SAMPLED_FRAMES = 30


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


def rotated_box_to_aabb(
    value: dict[str, float], frame_width: int, frame_height: int
) -> np.ndarray:
    """Convert a Label Studio rotated rectangle into a pixel xyxy AABB."""

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
    return np.array(
        [
            np.clip(x0, 0, frame_width - 1),
            np.clip(y0, 0, frame_height - 1),
            np.clip(x1, 0, frame_width - 1),
            np.clip(y1, 0, frame_height - 1),
        ],
        dtype=np.float32,
    )


def load_sampled_boxes(
    json_path: Path, frame_width: int, frame_height: int, final_source_frame: int
) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Load every object's boxes on frames retained by ``VIDEO_STRIDE``."""

    task = json.loads(json_path.read_text(encoding="utf-8"))[0]
    boxes_by_object: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for result in task["annotations"][0]["result"]:
        if result["type"] != "videorectangle":
            continue
        for value in result["value"]["sequence"]:
            if not value.get("enabled", True):
                continue
            source_frame = value["frame"] - 1
            if source_frame > final_source_frame or source_frame % VIDEO_STRIDE:
                continue
            boxes_by_object[result["id"]].append(
                (source_frame, rotated_box_to_aabb(value, frame_width, frame_height))
            )
    return dict(boxes_by_object)


def select_block_prompts(
    boxes_by_object: dict[str, list[tuple[int, np.ndarray]]],
    block_start: int,
    block_stop: int,
) -> dict[int, dict[str, np.ndarray]]:
    """Use the first and last available box for each object in one block."""

    prompts: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    for object_id, object_boxes in boxes_by_object.items():
        in_block = [
            (source_frame, box)
            for source_frame, box in object_boxes
            if block_start <= source_frame < block_stop
        ]
        if not in_block:
            continue
        first_source_frame, first_box = in_block[0]
        last_source_frame, last_box = in_block[-1]
        prompts[first_source_frame - block_start][object_id] = first_box
        prompts[last_source_frame - block_start][object_id] = last_box
    return dict(prompts)


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
        str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
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


def extract_block_frames(
    cap: cv2.VideoCapture,
    frame_dir: Path,
    block_start: int,
    block_stop: int,
) -> int:
    """Extract one block's sampled frames as sequential JPEGs for SAM2."""

    cap.set(cv2.CAP_PROP_POS_FRAMES, block_start)
    extracted = 0
    for source_frame in range(block_start, block_stop, VIDEO_STRIDE):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame {source_frame}")
        if not cv2.imwrite(str(frame_dir / f"{extracted:06d}.jpg"), frame):
            raise RuntimeError("Could not write a temporary SAM2 frame")
        extracted += 1
        for _ in range(VIDEO_STRIDE - 1):
            if source_frame + _ + 1 >= block_stop:
                break
            if not cap.grab():
                raise RuntimeError("Could not advance through the input video")
    return extracted


def process_block(
    predictor: object,
    cap: cv2.VideoCapture,
    out: cv2.VideoWriter,
    block_start: int,
    block_stop: int,
    prompts: dict[int, dict[str, np.ndarray]],
) -> None:
    """Run one isolated SAM2 state and append its union masks to ``out``."""

    with TemporaryDirectory() as temp_dir:
        frame_dir = Path(temp_dir)
        frame_count = extract_block_frames(cap, frame_dir, block_start, block_stop)
        inference_state = predictor.init_state(
            # SAM2 recognises a JPEG directory only when passed as a string, not Path.
            video_path=str(frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=True,
        )
        for relative_source_frame, frame_boxes in prompts.items():
            for object_id, box in frame_boxes.items():
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=relative_source_frame // VIDEO_STRIDE,
                    obj_id=object_id,
                    box=box,
                )

        cap.set(cv2.CAP_PROP_POS_FRAMES, block_start)
        for frame_idx, _, logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=frame_count,
        ):
            source_frame = block_start + frame_idx * VIDEO_STRIDE
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Could not read frame {source_frame}")
            union_mask = (logits[:, 0] > 0).any(dim=0).cpu().numpy()
            draw_mask(frame, union_mask)
            out.write(frame)


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
            boxes_by_object = load_sampled_boxes(
                JSON_PATH, frame_width, frame_height, first_third_frames - 1
            )
            if not boxes_by_object:
                raise RuntimeError("No Label Studio boxes matched the requested frames")

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            checkpoint, model_cfg = get_checkpoint_and_config("small")
            predictor = build_sam2_video_predictor(model_cfg, checkpoint, device)
            out = cv2.VideoWriter(
                str(OUT_PATH),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps / VIDEO_STRIDE,
                (frame_width, frame_height),
            )
            if not out.isOpened():
                raise RuntimeError(f"Could not open output video: {OUT_PATH}")
            clip_cap = cv2.VideoCapture(clip_path)
            if not clip_cap.isOpened():
                raise RuntimeError(f"Could not open temporary clip: {clip_path}")
            try:
                block_source_length = BLOCK_SAMPLED_FRAMES * VIDEO_STRIDE
                for block_start in range(0, first_third_frames, block_source_length):
                    block_stop = min(block_start + block_source_length, first_third_frames)
                    prompts = select_block_prompts(
                        boxes_by_object, block_start, block_stop
                    )
                    if not prompts:
                        continue
                    objects = {object_id for frame_boxes in prompts.values() for object_id in frame_boxes}
                    print(
                        f"Processing frames {block_start}:{block_stop} with "
                        f"{len(objects)} electrodes and "
                        f"{sum(map(len, prompts.values()))} boundary prompts."
                    )
                    process_block(predictor, clip_cap, out, block_start, block_stop, prompts)
            finally:
                clip_cap.release()
                out.release()
    finally:
        cap.release()


if __name__ == "__main__":
    main()
