from pathlib import Path

from ultralytics import YOLO
import cv2
import numpy as np

from electrode_segmentation.sam2_boxes import boxes2mask
from electrode_detection.inference import video_inference

model_path = "obb_experiments/side1/base/weights/best.pt"
out_path = Path("sam2_experiments/boxes2mask.mp4")
video_in_path=Path("videos/3-view/angled-light/side1-crop-start.mp4")

stride = 5
scale = 5
model_size = "base-plus"

def main() -> None:

    cap = cv2.VideoCapture(video_in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not load video: {video_in_path}")

    try:

        model = YOLO(model_path)

        predictions = video_inference(
            model=model,
            cap=cap,
            stride=stride,
        )

        boxes = {}

        for prediction in predictions:
            if not prediction.obbs:
                continue

            boxes[prediction.frame_num - 1] = np.array(
                [
                    [
                        obb.corners[:, 0].min(),
                        obb.corners[:, 1].min(),
                        obb.corners[:, 0].max(),
                        obb.corners[:, 1].max(),
                    ]
                for obb in prediction.obbs
                ], 
                dtype=np.float32
            )

        boxes2mask(
            cap=cap,
            out_path=out_path,
            boxes=boxes,
            extract_stride=stride,
            scale=scale,
            model_size=model_size
        )

    finally:
        cap.release()


if __name__ == "__main__":

    main()









