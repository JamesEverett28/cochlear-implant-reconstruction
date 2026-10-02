from __future__ import annotations
from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .inference import FramePrediction



def xyxyxyxy_to_xywhr(
    obb: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:

    """
    Convert (un-normalized) xyxyxyxy obb to xywhr obb, with r in radians.
    """

    center = obb.mean(axis=0)

    edge_w = obb[1] - obb[0]
    edge_h = obb[2] - obb[1]

    w = np.linalg.norm(edge_w)
    h = np.linalg.norm(edge_h)

    r = np.arctan2(edge_w[1], edge_w[0])

    return np.array([center[0], center[1], w, h, r], dtype=np.float32)



def perturb_xywhr(
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



def rasterize_gaussian_blobs(
    obbs: list[npt.NDArray[np.float32]],
    img_h: int,
    img_w: int,
    sigma_scale: float = 0.5,
    truncate: float = 3.0
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



def linear_predict(
    frame_num: int,
    tracks: Tracks,

) -> list[npt.NDArray[np.float32]]:

    # Predict obbs at a given frame by extrapolating the centre from two previous frames.

    preds = []

    for track in tracks:

        if len(track) < 2:

            continue

        c0 = track[-2].obb.mean(axis=0)
        c1 = track[-1].obb.mean(axis=0)

        c3 = c1 + (c1 - c0) * (frame_num - track[-1].frame_num) / (track[-1].frame_num - track[-2].frame_num)

        pred = track[-1].obb + c3

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

                intersection, _ = cv2.intersectConvexConvex(pred_obb, track.obb)

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



    def predict_obb(
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

        



def get_prior(
    frame_num: int,
    tracks: Tracks,
    img_h: int,
    img_w: int,
    predict_method: str = "linear",
    sigma_scale: float = 0.5,
    perturb: bool = False,
    pos_std: float = 0.05,
    size_std: float = 0.05,
    angle_std: float = np.deg2rad(3),
) -> npt.NDArray[np.uint8]:


    pred_obbs = tracks.predict_obbs(frame_num, predict_method)

    for i in range(len(pred_obbs)):

        pred_obbs[i] = xyxyxyxy_to_xywhr(pred_obbs[i])

        if perturb:

            pred_obbs[i] = perturb_xywhr(
                obb=pred_obbs[i],
                pos_std=pos_std,
                size_std=size_std,
                angle_std=angle_std
            )

    prior = rasterize_gaussian_blobs(
        obbs=pred_obbs,
        img_h=img_h,
        img_w=img_w,
        sigma_scale=sigma_scale
    )

    return prior

    





       


        
    









        




def get_prior():

    pass 

