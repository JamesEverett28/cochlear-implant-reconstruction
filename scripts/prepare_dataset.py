from pathlib import Path

import cv2

from electrode_detection.dataset import generate_dataset_from_json

json_path = Path("datasets/side1-insertion/label-studio.json")
video_path = Path("videos/3-view/angled-light/side1-crop.mp4")
dataset_dir = Path("datasets/side1-insertion/4channels")
stride = 5
fourth_channel = True

def main() -> None:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
            raise RuntimeError("Could not load video")

    generate_dataset_from_json(
        json_path=json_path,
        cap=cap,
        dataset_dir=dataset_dir,
        stride=stride,
        fourth_channel=fourth_channel
    )

if __name__ == "__main__":

    main()

    


