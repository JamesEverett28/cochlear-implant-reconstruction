from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import cv2

if TYPE_CHECKING:
    from .inference import FramePredition



@dataclass
class PerformanceMetrics:
    precision: float
    recall: float
    median_matched_iou: float
    mean_matched_iou: float



def get_iou(
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32]
) -> float:

    """
    Get iou between two rects of shape (4, 2).
    """

    area_a = abs(cv2.contourArea(a))
    area_b = abs(cv2.contourArea(b))

    intersection, _ = cv2.intersectConvexConvex(a, b)

    union = area_a + area_b - intersection

    return intersection / union



def get_frame_metrics(
    predictions: list[npt.NDArray[np.float32]],
    labels: list[npt.NDArray[np.float32]],
    iou_thresh = float
) -> tuple[list[float], int, int, int]:

    """
    Get IoUs, TP, FP, FN for a particular frame
    """

    pairs = []

    for a_idx, a in enumerate(predictions):
        for b_idx, b in enumerate(labels):

            iou = get_iou(a, b)

            if iou >= iou_thresh:

                pairs.append((iou, a_idx, b_idx))

    pairs.sort(
        key = lambda x: x[0],
        reverse = True 
    )

    matched_a_idxs, matched_b_idxs = set(), set()
    ious = []

    for iou, a_idx, b_idx in pairs:

        if a_idx not in matched_a_idxs and b_idx not in matched_b_idxs:

            ious.append(iou)

            matched_a_idxs.add(a_idx)
            matched_b_idxs.add(b_idx)

    tp = len(ious)
    fp = len(predictions) - tp
    fn = len(labels) - tp

    return ious, tp, fp, fn




def get_video_metrics(
    predictions: list[FramePredition],
    labels: dict[int, list[npt.NDArray[np.float32]]],
    iou_thresh: float
) -> list[PerformanceMetrics]:

    """
    Get precision, recall, median matched iou, mean matched iou for a sequence of frames
    """

    if len(predictions) != len(labels):

        raise RuntimeError("Number of prediction frames must match number of labelled frames.")

    ious = []
    tp, fp, fn = 0, 0, 0

    for i in range(len(predictions)):

        ious_i, tp_i, fp_i, fn_i = get_frame_metrics(
            predictions=predictions[i],
            labels=labels[i]
        )


    
    