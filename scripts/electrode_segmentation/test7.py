"""Prompt SAM2 with Label Studio electrode centres on the resized crop video."""

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
VIDEO_PATH = Path("videos/3-view/angled-light/side1-crop-square-1000.mp4")
OUT_PATH = Path("out.mp4")
VIDEO_STRIDE = 5
PROMPT_STRIDE = 25  # One prompt every five frames supplied to SAM2.
OBJECT_ID = 0  # All electrode-centre prompts jointly describe the full implant.

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


def scaled_box_centre(
    value: dict[str, float], frame_width: int, frame_height: int
) -> np.ndarray:
    """Map a Label Studio box centre to the aspect-ratio-preserving resize."""

    point = np.array(
        [
            (value["x"] + value["width"] / 2) / 100 * frame_width,
            (value["y"] + value["height"] / 2) / 100 * frame_height,
        ],
        dtype=np.float32,
    )
    point[0] = np.clip(point[0], 0, frame_width - 1)
    point[1] = np.clip(point[1], 0, frame_height - 1)
    return point


def load_prompt_points(
    json_path: Path, frame_width: int, frame_height: int, final_frame: int
) -> dict[int, np.ndarray]:
    """Return all electrode centres on each frame that receives a SAM2 prompt."""

    task = json.loads(json_path.read_text(encoding="utf-8"))[0]
    points_by_frame: dict[int, list[np.ndarray]] = defaultdict(list)
    for result in task["annotations"][0]["result"]:
        if result["type"] != "videorectangle":
            continue
        for value in result["value"]["sequence"]:
            if not value.get("enabled", True):
                continue
            source_frame = value["frame"] - 1
            if source_frame > final_frame or source_frame % PROMPT_STRIDE:
                continue
            points_by_frame[source_frame].append(
                scaled_box_centre(value, frame_width, frame_height)
            )
    return {
        frame: np.stack(points).astype(np.float32)
        for frame, points in points_by_frame.items()
    }


def write_first_third(
    cap: cv2.VideoCapture,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    fps: float,
    clip_path: Path,
) -> int:
    """Write the first third of the video and return its source-frame length."""

    clip_frames = (frame_count + 2) // 3
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
        for _ in range(clip_frames):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Could not read the requested first third of the video")
            writer.write(frame)
    finally:
        writer.release()
    return clip_frames


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
            temp_path = Path(temp_dir)
            clip_path = temp_path / "side1-crop-resized-first-third.mp4"
            clip_frame_count = write_first_third(
                cap, frame_count, frame_width, frame_height, fps, clip_path
            )
            prompt_points = load_prompt_points(
                JSON_PATH, frame_width, frame_height, clip_frame_count - 1
            )
            if not prompt_points:
                raise RuntimeError("No Label Studio boxes matched the requested prompt frames")

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
                frame_dir = temp_path / "frames"
                extract_frames(clip_cap, frame_dir, VIDEO_STRIDE)
                inference_state = predictor.init_state(
                    video_path=str(frame_dir),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                    async_loading_frames=True,
                )
                for source_frame, points in prompt_points.items():
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=source_frame // VIDEO_STRIDE,
                        obj_id=OBJECT_ID,
                        points=points,
                        labels=np.ones(len(points), dtype=np.int32),
                    )

                print(
                    f"Tracking one implant using {len(prompt_points)} annotated frames "
                    f"over the first {clip_frame_count} source frames."
                )
                clip_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for sampled_frame, _, logits in predictor.propagate_in_video(inference_state):
                    source_frame = sampled_frame * VIDEO_STRIDE
                    clip_cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
                    ret, frame = clip_cap.read()
                    if not ret:
                        raise RuntimeError(f"Could not read frame {source_frame}")
                    draw_mask(frame, (logits[0, 0] > 0).cpu().numpy())
                    out.write(frame)
                    print(
                        f"Processed frame {source_frame + 1}/{clip_frame_count}", end="\r"
                    )
            finally:
                clip_cap.release()
                out.release()
    finally:
        cap.release()


if __name__ == "__main__":
    main()
