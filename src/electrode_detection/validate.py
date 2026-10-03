from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np
import cv2

from .obb import OBB

if TYPE_CHECKING:
    from .inference import FramePrediction




@dataclass
class PerformanceMetrics:
    precision: float
    recall: float
    median_matched_iou: float
    mean_matched_iou: float



def get_iou(
    a: OBB,
    b: OBB
) -> float:

    """
    Get iou between two obbs.
    """

    area_a = abs(cv2.contourArea(a.corners))
    area_b = abs(cv2.contourArea(b.corners))

    intersection, _ = cv2.intersectConvexConvex(a.corners, b.corners)

    union = area_a + area_b - intersection

    return intersection / union



def get_frame_metrics(
    predictions: list[OBB], 
    labels: list[OBB], 
    iou_thresh: float
) -> tuple[list[float], int, int, int]:

    """
    Get IoUs, TP, FP, FN for a single frame, given some iou threshold.
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
    predictions: list[FramePrediction],
    labels: dict[int, list[OBB]],
    iou_thresh: float
) -> PerformanceMetrics:

    """
    Get precision, recall, median matched iou, mean matched iou for a sequence of frames.
    """

    ious = []
    tp, fp, fn = 0, 0, 0

    for prediction in predictions:

        pred_obbs = prediction.obbs 
        label_obbs = labels.get(prediction.frame_num, [])

        ious_i, tp_i, fp_i, fn_i = get_frame_metrics(
            predictions=pred_obbs,
            labels=label_obbs,
            iou_thresh=iou_thresh
        )

        ious.extend(ious_i)
        tp += tp_i
        fp += fp_i
        fn += fn_i

    return PerformanceMetrics(
        precision = tp / (tp + fp),
        recall = tp / (tp + fn),
        median_matched_iou = np.median(ious),
        mean_matched_iou = np.mean(ious)
    )
