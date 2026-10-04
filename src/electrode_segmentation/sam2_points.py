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

def points2mask(
    cap: cv2.VideoCapture,
    out_path: str | Path,
    points: list[npt.NDArray[np.float32]],
    stride: int
):

    """
    Generate a mask from a list of prompt points.
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
        fps, 
        (frame_w, frame_h)
    )

    if not out.isOpened():
        raise RuntimeError(f"Could not open output video: {out_path}")

    sam2_checkpoint = "dependencies/sam2/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "dependencies/sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device)

    with TemporaryDirectory() as frame_dir:

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        extract_frames(cap, frame_dir, stride)

        inference_state = predictor.init_state(video_path=frame_dir)

        for frame_idx, frame_points in enumerate(points):

            if frame_idx % stride != 0:
                continue

            labels = np.ones(len(frame_points), dtype=np.int32)

            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=0,
                points=frame_points,
                labels=labels
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_idx, _, logits in predictor.propagate_in_video(inference_state):

            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Could not read frame {frame_idx}")

            mask = (logits[0] > 0).cpu().numpy()

            draw_mask(frame, mask)
            out.write(frame)

    out.release()





    




    

    









