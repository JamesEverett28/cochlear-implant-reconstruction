import os 
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import numpy.typing as npt
import torch 
import cv2
from sam2.build_sam import build_sam2_video_predictor

from .rendering import draw_mask
from utils.video import extract_frames
from .utils import get_checkpoint_and_config

def points2mask(
    cap: cv2.VideoCapture,
    out_path: str | Path,
    points: dict[int, npt.NDArray[np.float32]],
    stride: int,
    model_size: str = "large"
):

    """
    Generate a mask from a list of points.
    """

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )


    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        out_path, 
        fourcc, 
        fps / stride, 
        (frame_w, frame_h)
    )

    if not out.isOpened():
        raise RuntimeError(f"Could not open output video: {out_path}")

    sam2_checkpoint, model_cfg = get_checkpoint_and_config(model_size)

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device)

    with TemporaryDirectory() as frame_dir:

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        extract_frames(cap, frame_dir, stride)

        inference_state = predictor.init_state(
            video_path=frame_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=True,
        )

        for frame_idx, frame_points in points.items():
            right_to_left = frame_points[
                np.argsort(frame_points[:, 0])[::-1]
            ]

            labels = np.ones(len(right_to_left), dtype=np.int32)

            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx // stride,
                obj_id=0,
                points=right_to_left,
                labels=labels,
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_idx, _, logits in predictor.propagate_in_video(inference_state):

            source_frame_idx = frame_idx * stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)

            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Could not read frame {source_frame_idx}")

            mask = (logits[0, 0] > 0).cpu().numpy()
            
            draw_mask(frame, mask)
            out.write(frame)

    out.release()





    




    

    








