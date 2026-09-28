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
class FramePredition:
    frame_idx: int
    obbs: npt.NDArray[np.float32]
    confidence: npt.NDArray[np.float32]


def frame_inference(
    model: YOLO | FourChannelsOBBYolo,
    bgr_frame: npt.NDArray[np.uint8],
    prev_obbs: list[npt.NDArray[np.float64]] | None = None
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
) -> list[FramePredition]:

    """
    Run inference on a full video.
    """

    in_path = Path(in_path)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError("Could not load video")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    predictions = []
    prev_obbs = np.empty((0, 4, 2), dtype=np.float32)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:

            obbs = frame_inference(model, frame, prev_obbs)

            predictions.append(
                FramePredition(
                    frame_idx = frame_idx,
                    obbs = obbs.xyxyxyxy.cpu().numpy(),
                    confidence = obbs.conf.cpu().numpy()
                )
            )

            prev_obbs = predictions[-1].obbs

        frame_idx += 1
        print(f"Processed {frame_idx}/{frame_count}")

    cap.release()

    return predictions

