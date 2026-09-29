from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import numpy.typing as npt
import cv2

if TYPE_CHECKING:
    from .inference import FramePrediction


def mask_from_obbs(
    obbs: list[npt.NDArray[np.float32]],
    img_h: int,
    img_w: int,
    normalized: bool = True
) -> npt.NDArray[np.uint8]:

    """
    Create a binary mask from obb coords.
    """

    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for obb in obbs:

        obb = obb.copy()

        if normalized:
            
            obb[:, 0] *= img_w
            obb[:, 1] *= img_h

        obb = np.round(obb).astype(np.int32)

        cv2.fillPoly(mask, [obb], 255)

    return mask



def draw_frame_predictions(
    frame: npt.NDArray[np.uint8],
    obbs: npt.NDArray[np.float32],
    scale: int
) -> npt.NDArray[np.uint8]:

    """
    Draw obb predictions on corresponding frame.
    """

    # Upsample just for display 
    frame = cv2.resize(
        src = frame,
        dsize = None, 
        fx = scale,
        fy = scale,
        interpolation = cv2.INTER_CUBIC
    )

    for obb in obbs:

        obb = obb.copy()

        obb = np.round(obb * scale).astype(np.int32)

        cv2.polylines(
            frame,
            [obb],
            isClosed = True,
            color = (0, 255, 0),
            thickness = 1,
            lineType = cv2.LINE_AA
        )

    return frame



def draw_video_predictions(
    in_path: str | Path,
    out_path: str | Path,
    predictions: list[FramePrediction]
) -> None:

    """
    Draw obb predictions on corresponding video.
    """

    in_path = Path(in_path)
    out_path = Path(out_path)

    out_path.parent.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError("Could not load video")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS)

    stride = predictions[1].frame_num - predictions[0].frame_num
    out_fps = in_fps / stride
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Upsample just for display 
    scale = 4
    out = cv2.VideoWriter(
        out_path, 
        fourcc, 
        out_fps, 
        (frame_width*scale, frame_height*scale), 
        isColor=True
    )

    if not out.isOpened():
        raise RuntimeError("Could not open output video")

    frame_num = 1
    first_pred = predictions[0].frame_num
    final_pred = predictions[-1].frame_num

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if (frame_num-1) % stride == 0:

            if frame_num >= first_pred and frame_num <= final_pred:

                obbs = predictions[(frame_num - first_pred) // stride].obbs

            else:

                obbs = []

            annotated_frame = draw_frame_predictions(frame, obbs, scale)

            out.write(annotated_frame)

        frame_num += 1

    cap.release()
    out.release()





    



