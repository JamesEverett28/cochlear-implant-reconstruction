from pathlib import Path

import numpy as np

from electrode_detection.experiments import Experiment, run_experiment

base = Experiment(
    project="side1",
    name="base",
    json_path=Path("datasets/side1/label-studio.json"),
    video_in_path=Path("videos/3-view/angled-light/side1-crop.mp4"),
    train_frame_range=(1, 1060),
    val_frame_range=(1190, 1731), 
    model_path=Path("yolo26n-obb.pt"),  
    epochs=100,
    batch=16,
    imgsz=256,
    patience=10,
    fourth_channel=False,
    stride=5,
    confidence=0.25,
    predict_method="linear",
    tracks_dropout=1,
    rasterize_method="direct",
    pos_std=0.05,
    size_std=0.05,
    angle_std=np.deg2rad(3),
    blank_prior_prob=0.2,
    iou_thresh=0.95
)


EXPERIMENTS = [
    base,
]


def main() -> None:

    for experiment in EXPERIMENTS:

        metrics = run_experiment(experiment)
        print(f"{experiment.project} / {experiment.name}: {metrics}")


if __name__ == "__main__":

    main()

