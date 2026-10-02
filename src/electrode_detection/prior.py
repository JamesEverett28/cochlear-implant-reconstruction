from __future__ import annotations
from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt
from ultralytics.utils.ops import xyxyxyxy2xywhr

from .rendering import rasterize_prior
if TYPE_CHECKING:
    from .inference import FramePrediction



def perturb_obb(
    obb: npt.NDArray[np.float32],
    pos_std: float = 0.05,
    size_std: float = 0.05,
    angle_std: float = np.deg2rad(3),
) -> npt.NDArray[np.float32]:
    """
    Perturb xywhr obb slightly to prevent 'perfect' training priors.
    """

    cx, cy, w, h, r = obb

    # Additive position noise
    cx += np.random.normal(0, pos_std * w)
    cy += np.random.normal(0, pos_std * h)

    # Additive size noise
    w += np.random.normal(0, size_std * w)
    h += np.random.normal(0, size_std * h)

    # Additive angular noise
    r += np.random.normal(0, angle_std)

    return np.array([cx, cy, w, h, r], dtype=np.float32)



def linear_predict(
    frame_num: int,
    tracks: Tracks,

) -> list[npt.NDArray[np.float32]]:

    # Predict obbs at a given frame by extrapolating the centre from two previous frames.

    preds = []

    for track in tracks:

        if len(track) < 2:

            continue

        c0 = track[-2].mean(axis=0)
        c1 = track[-1].mean(axis=0)

        c3 = c1 + (c1 - c0) * (frame_num - track[-1].frame_num) / (track[-1].frame_num - track[-2].frame_num)

        pred = track[-1] + c3

        preds.append(pred)

    return preds



class Track:

    # Object to track electrode instances between frames.

    def __init__(
        self,
        frame_num: int,
        obb: npt.NDArray[np.float32],
    ) -> None:

        self.frame_nums = [frame_num]
        self.obbs = [obb]

    def update(
        frame_num: int,
        obb: npt.NDArray[np.float32]
    ):

        self.frame_nums.append(frame_num)
        self.obbs.append(obb)




class Tracks:

    # Object to hold and update a set of Track objects corresponding to a single video \
    # as well as predicting obb locations in future frames based on current tracks.

    def __init__(
        self,
        stride: int, 
        dropout: int,

    ) -> None:

        self.tracks = []
        self.stride = stride
        self.dropout = dropout



    def update(
        predictions: FramePrediction
    ) -> None:

        pairs = []

        for pred_idx, pred_obb in enumerate(predictions.obbs):
            for track_idx, track in enumerate(self.tracks):

                intersection, _ = cv2.intersectConvexConvex(pred_obb, track.obbs[-1])

                if intersection > 0:

                    pairs.append(intersection, pred_idx, track_idx)

        pairs.sort(
            key = lambda x: x[0], 
            reverse=True
        )

        matched_pred_idxs, matched_track_idxs = set(), set()

        for _, pred_idx, track_idx in pairs:   
            if pred_idx not in matched_pred_idxs and track_idx not in matched_track_idxs:

                self.tracks[track_idx].update(
                    predictions.frame_num,
                    predictions.obbs[pred_idx]
                )

        updated_tracks = []
        
        for track_idx, track in enumerate(self.tracks):

            if track_idx in matched_track_idxs:

                updated_tracks.append(self.tracks[track_idx])

            elif predictions.frame_num - track[-1].frame_num <= self.dropout * self.stride:

                updated_tracks.append(self.tracks[track_idx])

        for pred_idx, pred_obb in enumerate(predictions.obbs):

            if pred_idx not in matched_pred_idxs:

                updated_tracks.append(Track(
                    predictions.frame_num,
                    pred_obb
                ))

        self.tracks = updated_tracks



    def predict_obbs(
        frame_num: int,
        predict_method: str = "linear"
    ) -> list[npt.NDArray[np.float32]]:


        PREDICT_METHODS = {
            "linear": linear_predict,
        }

        try:
            predict_method = PREDICT_METHODS[predict_method]

        except KeyError:
            raise ValueError(f"Unknown predict method {predict_method}")

        return predict_method(frame_num, self)

        

def get_prior_from_obbs(
    obbs: list[npt.NDArray[np.float32]],
    img_h: int,
    img_w: int,
    rasterize_method: str,
    perturb: bool,
    sigma_scale: float = 0.5,
    truncate: float = 3.0,
    pos_std: float = 0.05,
    size_std: float = 0.05,
    angle_std: float = np.deg2rad(3),
) -> npt.NDArray[np.uint8]:

    """
    Get prior image from input obbs.
    """

    for i in range(len(obbs)):

        if len(obbs[i].shape) > 1:

            obbs[i] = xyxyxyxy2xywhr(obbs[i])

        if perturb:

            obbs[i] = perturb_obb(
                obb=obbs[i],
                pos_std=pos_std,
                size_std=size_std,
                angle_std=angle_std
            )

    prior = rasterize_prior(
        obbs=obbs,
        img_h=img_h,
        img_w=img_w,
        rasterize_method=rasterize_method,
        sigma_scale=sigma_scale,
        truncate=truncate
    )

    return prior

    





       


        
    









        




def get_prior():

    pass 

