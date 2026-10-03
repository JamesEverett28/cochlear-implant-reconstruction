from pathlib import Path
from dataclasses import replace

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
    replace(base, name="exp1", epochs=200),
    replace(base, name="exp2", epochs=300),
    replace(base, name="exp3", model_path=Path("yolo26s-obb.pt")),
    replace(base, name="exp4", model_path=Path("yolo26m-obb.pt")),
    replace(base, name="exp5", fourth_channel=True),
    replace(base, name="exp6", fourth_channel=True, rasterize_method="gaussian"),
    replace(base, name="exp7", fourth_channel=True, rasterize_method="guassian", stride=1),
    replace(base, name="exp8", fourth_channel=True, rasterize_method="gaussian", pos_std=0.1, size_std=1, angle_std=np.deg2rad(6)),
    replace(base, name="exp9", fourth_channel=True, rasterize_method="gaussian", confidence=0.15),
    replace(base, name="exp10", fourth_channel=True, rasterize_method="gaussian", confidence=0.35),
    replace(base, name="exp11", fourth_channel=True, rasterize_method="gaussian", blank_prior_prob=0.1),
    replace(base, name="exp12", fourth_channel=True, rasterize_method="gaussian", blank_prior_prob=0.3),
    replace(base, name="exp13", fourth_channel=True, rasterize_method="gaussian", iou_thresh=0.9),
]


def main() -> None:

    for experiment in EXPERIMENTS:

        metrics = run_experiment(experiment)
        print(f"{experiment.project} / {experiment.name}: {metrics}")


if __name__ == "__main__":

    main()

