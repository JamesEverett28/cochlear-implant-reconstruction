from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import numpy.typing as npt
import cv2

from ultralytics.utils.ops import xywhr2xyxyxyxy

if TYPE_CHECKING:
    from .inference import FramePrediction



def rasterize_obbs(
    obbs: list[npt.NDArray[np.float32]],
    img_h: int,
    img_w: int,
) -> npt.NDArray[np.uint8]:

    """
    Produce a mask of rectangles from xywhr obbs.
    """

    prior = np.zeros((img_h, img_w), dtype=np.uint8)

    for obb in obbs:

        obb = xywhr2xyxyxyxy(obb)

        obb = np.round(obb).astype(np.int32)

        cv2.fillConvexPoly(prior, obb, 255)

    return prior


def rasterize_gaussian_blobs(
    obbs: list[npt.NDArray[np.float32]],
    img_h: int,
    img_w: int,
    sigma_scale: float = 0.5,
    truncate: float = 3.0,
) -> npt.NDArray[np.uint8]:
    
    """
    Produce orientated Gaussian blobs from xywhr obbs.

    sigma_scale:
        sigma relative to half-width / half-height.

        1.0 => sigma_x = w/2
        0.5 => sigma_x = w/4

    """

    prior = np.zeros((img_h, img_w), dtype=np.float32)

    for cx, cy, w, h, r in obbs:

        sigma_x = max(w * 0.5 * sigma_scale, 1e-6)
        sigma_y = max(h * 0.5 * sigma_scale, 1e-6)

        c = np.cos(r)
        s = np.sin(r)

        # Axis-aligned extent of the rotated truncate-sigma ellipse
        rx = truncate * np.sqrt(
            (sigma_x * c) ** 2 +
            (sigma_y * s) ** 2
        )
        ry = truncate * np.sqrt(
            (sigma_x * s) ** 2 +
            (sigma_y * c) ** 2
        )

        # Clamp ROI to image
        x0 = max(0, int(np.floor(cx - rx)))
        x1 = min(img_w, int(np.ceil(cx + rx)) + 1)
        y0 = max(0, int(np.floor(cy - ry)))
        y1 = min(img_h, int(np.ceil(cy + ry)) + 1)

        if x0 >= x1 or y0 >= y1:
            continue

        # Only generate coordinates inside this OBB's ROI
        yy, xx = np.mgrid[y0:y1, x0:x1]

        dx = xx - cx
        dy = yy - cy

        # Rotate into OBB coordinate system
        x_rot = c * dx + s * dy
        y_rot = -s * dx + c * dy

        blob = np.exp(
            -0.5 * (
                (x_rot / sigma_x) ** 2 +
                (y_rot / sigma_y) ** 2
            )
        )

        # Combine overlapping priors using maximum
        roi = prior[y0:y1, x0:x1]
        np.maximum(roi, blob, out=roi)

    return np.round(prior * 255).astype(np.uint8)



def rasterize_prior(
    obbs: list[npt.NDArray[np.float32]],
    img_h: int,
    img_w: int,
    rasterize_method: str,
    sigma_scale: float = 0.5,
    truncate: float = 3.0,

) -> npt.NDArray[np.uint8]:

    """
    Rasterize xywhr obbs according to specified method.
    """

    if rasterize_method == "direct":

        return rasterize_obbs(
            obbs,
            img_h,
            img_w,
        )

    elif rasterize_method == "gaussian":

        return rasterize_gaussian_blobs(
            obbs,
            img_h, 
            img_w,
            sigma_scale,
            truncate
        )

    else:

        raise ValueError(f"Unknown rasterize method {rasterize_method}")



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

        corners = xywhr2xyxyxyxy(obb)

        corners = np.round(corners * scale).astype(np.int32)

        cv2.polylines(
            frame,
            [corners],
            isClosed = True,
            color = (0, 255, 0),
            thickness = 1,
            lineType = cv2.LINE_AA
        )

    return frame



def draw_video_predictions(
    cap: cv2.VideoCapture,
    out_path: str | Path,
    predictions: list[FramePrediction]
) -> None:

    """
    Draw obb predictions on corresponding video.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

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





    



