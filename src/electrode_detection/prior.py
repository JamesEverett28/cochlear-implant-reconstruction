from __future__ import annotations
from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt

from .obb import OBB
from .rendering import rasterize_prior
if TYPE_CHECKING:
    from .inference import FramePrediction



def perturb_obb(
    obb: OBB,
    pos_std: float = 0.05,
    size_std: float = 0.05,
    angle_std: float = np.deg2rad(3),
) -> OBB:
    """
    Perturb obb slightly to prevent 'perfect' training priors.
    """

    cx, cy, w, h, r = obb.xywhr

    # Additive position noise
    cx += np.random.normal(0, pos_std * w)
    cy += np.random.normal(0, pos_std * h)

    # Additive size noise
    w = np.abs(w + np.random.normal(0, size_std * w))
    h = np.abs(h + np.random.normal(0, size_std * h))

    # Additive angular noise
    r += np.random.normal(0, angle_std)

    return OBB(np.array([cx, cy, w, h, r], dtype=np.float32))



def linear_predict(
    frame_num: int,
    tracks: Tracks,

) -> list[OBB]:

    # Predict obbs at a given frame by extrapolating the centre from two previous frames.

    preds = []

    for track in tracks.tracks:

        if len(track.obbs) < 2:

            continue

        c0 = track.obbs[-2].xywhr[:2]
        c1 = track.obbs[-1].xywhr[:2]

        del_c = (c1 - c0) * (frame_num - track.frame_nums[-1]) / (track.frame_nums[-1] - track.frame_nums[-2])

        pred = track.obbs[-1].copy()
        pred.xywhr[:2] += del_c

        preds.append(pred)

    return preds



class Track:

    # Object to track electrode instances between frames.

    def __init__(
        self,
        frame_num: int,
        obb: OBB,
    ) -> None:

        self.frame_nums = [frame_num]
        self.obbs = [obb]

    def update(
        self,
        frame_num: int,
        obb: OBB
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
        self,
        predictions: FramePrediction
    ) -> None:

        pairs = []

        pred_corners = [obb.corners for obb in predictions.obbs]
        track_corners = [track.obbs[-1].corners for track in self.tracks]

        for pred_idx, pred_obb in enumerate(pred_corners):
            for track_idx, track_obb in enumerate(track_corners):

                intersection, _ = cv2.intersectConvexConvex(pred_obb, track_obb)

                if intersection > 0:

                    pairs.append((intersection, pred_idx, track_idx))

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

                matched_pred_idxs.add(pred_idx)
                matched_track_idxs.add(track_idx)

        updated_tracks = []
        
        for track_idx, track in enumerate(self.tracks):

            if track_idx in matched_track_idxs:

                updated_tracks.append(self.tracks[track_idx])

            elif predictions.frame_num - track.frame_nums[-1] <= self.dropout * self.stride:

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
    ) -> list[OBB]:


        PREDICT_METHODS = {
            "linear": linear_predict,
        }

        try:
            predict_method = PREDICT_METHODS[predict_method]

        except KeyError:
            raise ValueError(f"Unknown predict method {predict_method}")

        return predict_method(frame_num, self)

        

def get_prior_from_obbs(
    obbs: list[OBB],
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

    if perturb:

        obbs = [obb.copy() for obb in obbs]

        for i in range(len(obbs)):

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
