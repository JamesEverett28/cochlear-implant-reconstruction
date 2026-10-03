from pathlib import Path 
from dataclasses import dataclass

from ultralytics import YOLO
from ultralytics.engine.results import OBB as UltralyticsOBB
import numpy.typing as npt
import numpy as np
import cv2

from .fourchannel_model import FourChannelsOBBYolo
from .obb import OBB
from .prior import Tracks, get_prior_from_obbs



@dataclass
class FramePrediction:
    frame_num: int
    obbs: list[OBB]
    confidence: npt.NDArray[np.float32]


def frame_inference(
    model: YOLO | FourChannelsOBBYolo,
    bgr_frame: npt.NDArray[np.uint8],
    prior: npt.NDArray[np.uint8],
    confidence: float
) -> UltralyticsOBB:

    """
    Get predictions from bgr frame and optional prior.
    """

    if isinstance(model, FourChannelsOBBYolo):

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        channels = np.dstack((rgb_frame, prior))

    else:

        channels = bgr_frame

    return model(channels, conf=confidence)[0].obb



def video_inference(
    model: YOLO | FourChannelsOBBYolo,
    cap: cv2.VideoCapture,
    stride: int,
    rasterize_method: str,
    predict_method: str,
    tracks_dropout: int,
    confidence: float,
    frame_range: tuple[int, int] | None = None
) -> list[FramePrediction]:

    """
    Run inference on a full video.
    """

    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_num = 1 if frame_range is None else frame_range[0]
    predictions = []
    tracks = Tracks(
        stride=stride, 
        dropout=tracks_dropout
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)

    while True:

        ret, frame = cap.read()
        if not ret:
            break
        if frame_range is not None and frame_num > frame_range[1]:
            break

        if (frame_num-1) % stride == 0:

            pred_obbs = tracks.predict_obbs(frame_num, predict_method)

            prior = get_prior_from_obbs(
                obbs=pred_obbs,
                img_h=img_h,
                img_w=img_w,
                rasterize_method=rasterize_method,
                perturb=False
            )

            obbs = frame_inference(model, frame, prior, confidence)

            predictions.append(
                FramePrediction(
                    frame_num = frame_num,
                    obbs = [OBB(xywhr) for xywhr in obbs.xywhr.cpu().numpy()],
                    confidence = obbs.conf.cpu().numpy()
                )
            )

            tracks.update(predictions[-1])

        frame_num += 1

    cap.release()

    return predictions
