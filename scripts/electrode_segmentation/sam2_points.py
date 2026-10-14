from pathlib import Path

from ultralytics import YOLO
import cv2
import numpy as np

from electrode_segmentation.sam2_points import points2mask
from electrode_detection.inference import video_inference

model_path = "obb_experiments/side1/base/weights/best.pt"
out_path = Path("sam2_experiments/points2mask.mp4")
video_in_path=Path("videos/3-view/angled-light/side1-crop-start.mp4")

stride = 5
prompt_stride = 5
block_size = 50
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

        points = {}

        for prediction in predictions:
            if not prediction.obbs:
                continue

            points[prediction.frame_num - 1] = np.array(
                [obb.xywhr[:2] for obb in prediction.obbs],
                dtype=np.float32
            )

        points2mask(
            cap=cap,
            out_path=out_path,
            points=points,
            extract_stride=stride,
            prompt_stride=prompt_stride,
            block_size=block_size,
            scale=scale,
            model_size=model_size
        )

    finally:
        cap.release()


if __name__ == "__main__":

    main()









