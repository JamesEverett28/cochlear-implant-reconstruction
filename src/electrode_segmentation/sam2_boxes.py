import os 
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch 
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from electrode_segmentation.rendering import draw_mask
from .utils import get_checkpoint_and_config

def boxes2mask(
    cap: cv2.VideoCapture,
    out_path: str | Path,
    boxes: dict[int, npt.NDArray[np.float32]],
    extract_stride: int,
    scale: float | None = None,
    model_size: str = "small"
):

    """
    Generate masks independently from axis-aligned bounding boxes.
    """

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        out_path, 
        fourcc, 
        fps / extract_stride, 
        (frame_w, frame_h)
    )

    if not out.isOpened():
        raise RuntimeError(f"Could not open output video: {out_path}")

    sam2_checkpoint, model_cfg = get_checkpoint_and_config(model_size)

    predictor = SAM2ImagePredictor(
        build_sam2(model_cfg, sam2_checkpoint, device=device)
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % extract_stride == 0:

            frame_boxes = boxes.get(frame_idx)

            if frame_boxes is not None and len(frame_boxes):

                predictor_frame = frame

                if scale is not None:

                    predictor_frame = cv2.resize(frame, None, fx=scale, fy=scale)
                    frame_boxes = frame_boxes * scale

                predictor.set_image(
                    cv2.cvtColor(predictor_frame, cv2.COLOR_BGR2RGB)
                )

                combined_mask = np.zeros((frame_h, frame_w), dtype=bool)

                for box in frame_boxes:

                    masks, _, _ = predictor.predict(
                        box=box,
                        multimask_output=False,
                    )

                    mask = masks[0].astype(np.uint8)

                    if scale is not None:

                        mask = cv2.resize(
                            mask,
                            (frame_w, frame_h),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    combined_mask |= mask.astype(bool)

                draw_mask(frame, combined_mask)

            out.write(frame)

            print(f"\rProcessed frame {frame_idx} / {frame_count}", end="", flush=True)

        frame_idx += 1

    out.release()
