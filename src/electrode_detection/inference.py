from pathlib import Path 
from dataclasses import dataclass

from ultralytics import YOLO
from ultralytics.engine.results import OBB
import numpy.typing as npt
import numpy as np
import cv2

from .fourchannel_model import FourChannelsOBBYolo
from .rendering import mask_from_obbs



@dataclass
class FramePrediction:
    frame_num: int
    obbs: list[npt.NDArray[np.float32]]
    confidence: npt.NDArray[np.float32]


def frame_inference(
    model: YOLO | FourChannelsOBBYolo,
    bgr_frame: npt.NDArray[np.uint8],
    prev_obbs: list[npt.NDArray[np.float32]] | None = None
) -> OBB:

    """
    Get predictions from bgr frame and optional prior.
    """

    h, w = bgr_frame.shape[:2]

    if type(model) == FourChannelsOBBYolo:

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        prior = mask_from_obbs(prev_obbs, h, w, normalized=False)

        channels = np.dstack((rgb_frame, prior))

    else:

        channels = bgr_frame

    return model(channels)[0].obb



def video_inference(
    model: YOLO | FourChannelsOBBYolo,
    in_path: str | Path,
    stride: int,
    frame_range: tuple[int, int] | None = None
) -> list[FramePrediction]:

    """
    Run inference on a full video.
    """

    in_path = Path(in_path)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError("Could not load video")

    frame_num = 1 if frame_range is None else frame_range[0]
    predictions = []
    prev_obbs = np.empty((0, 4, 2), dtype=np.float32)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)

    while True:

        ret, frame = cap.read()
        if not ret:
            break
        if frame_range is not None and frame_num > frame_range[1]:
            break

        if (frame_num-1) % stride == 0:

            obbs = frame_inference(model, frame, prev_obbs)

            predictions.append(
                FramePrediction(
                    frame_num = frame_num,
                    obbs = list(obbs.xyxyxyxy.cpu().numpy()),
                    confidence = obbs.conf.cpu().numpy()
                )
            )

            prev_obbs = predictions[-1].obbs

        frame_num += 1

    cap.release()

    return predictions

