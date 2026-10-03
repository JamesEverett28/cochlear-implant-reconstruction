import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO

from .dataset import generate_dataset_from_json
from .fourchannel_model import FourChannelsOBBYolo
from .validate import PerformanceMetrics, validate


METRICS_CONFIG_FIELDS = (
    "project",
    "name",
    "model_path",
    "epochs",
    "fourth_channel",
    "stride",
    "confidence",
    "predict_method",
    "tracks_dropout",
    "rasterize_method",
    "pos_std",
    "size_std",
    "angle_std",
    "blank_prior_prob",
    "iou_thresh",
)


@dataclass(frozen=True)
class Experiment:

    """
    All inputs and settings needed for one training/validation run.
    """

    project: str  # Grouping directory for related experiments.
    name: str  # Unique name for this experiment within its project.
    json_path: Path  # Label Studio annotation export.
    video_in_path: Path  # Source video used for dataset generation and validation.
    train_frame_range: tuple[int, int]  # Inclusive frame range used for training.
    val_frame_range: tuple[int, int]  # Inclusive frame range used for validation.
    model_path: Path = Path("yolo26n-obb.pt")  # Pretrained YOLO weights used to initialise training.
    epochs: int = 200  # Number of training epochs.
    batch: int = 16  # Images per optimisation batch.
    imgsz: int = 256  # Square input resolution used by YOLO.
    fourth_channel: bool = True  # Whether to train with the prior as a fourth image channel.
    stride: int = 5  # Number of video frames between sampled frames.
    confidence: float = 0.25  # Minimum confidence accepted during inference.
    predict_method: str = "linear"  # Method used to extrapolate tracked OBBs.
    tracks_dropout: int = 1  # Sampled frames a missing track remains active.
    rasterize_method: str = "direct"  # Method used to turn OBB priors into an image.
    pos_std: float = 0.05  # Position-noise standard deviation relative to object size.
    size_std: float = 0.05  # Size-noise standard deviation relative to object size.
    angle_std: float = np.deg2rad(3)  # Rotation-noise standard deviation in radians.
    blank_prior_prob: float = 0.2 # Probability of setting prior blank during training.
    iou_thresh: float = 0.95  # IoU threshold used to score a detection as matched.


def append_metrics(
    experiment: Experiment,
    metrics: PerformanceMetrics,
    metrics_path: Path,
) -> None:
    
    """
    Append the configuration and validation metrics for one experiment.
    """

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        field: getattr(experiment, field)
        for field in METRICS_CONFIG_FIELDS
    }
    row.update(asdict(metrics))

    write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0

    with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:

        writer = csv.DictWriter(metrics_file, fieldnames=row.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(row)



def run_experiment(
    experiment: Experiment,
    datasets_dir: Path = Path("datasets"),
    save_dir: Path = Path("experiments"),
) -> PerformanceMetrics:
    
    """
    Train one experiment, validate it, render predictions, and log its metrics.
    """

    dataset_dir = datasets_dir / experiment.project / experiment.name
    run_dir = save_dir / experiment.project / experiment.name
    video_out_path = run_dir / "val_predictions.mp4"
    metrics_out_path = save_dir / experiment.project / "val_metrics.csv"

    cap = cv2.VideoCapture(experiment.video_in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not load video: {experiment.video_in_path}")

    try:
        generate_dataset_from_json(
            json_path=experiment.json_path,
            cap=cap,
            dataset_dir=dataset_dir,
            stride=experiment.stride,
            fourth_channel=experiment.fourth_channel,
            rasterize_method=experiment.rasterize_method,
            pos_std=experiment.pos_std,
            size_std=experiment.size_std,
            angle_std=experiment.angle_std,
            frame_range=experiment.train_frame_range,
            blank_prior_prob=experiment.blank_prior_prob
        )

        model: YOLO | FourChannelsOBBYolo

        if experiment.fourth_channel:

            model = FourChannelsOBBYolo(experiment.model_path)

        else:
            
            model = YOLO(experiment.model_path)

        model.train(
            data=dataset_dir / "data.yaml",
            batch=experiment.batch,
            imgsz=experiment.imgsz,
            epochs=experiment.epochs,
            save_dir=run_dir,
            device="mps",
            plots=False,
            val=False,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,
            mosaic=0.0,
            mixup=0.0,
            cutmix=0.0,
            copy_paste=0.0,
            erasing=0.0,
            bgr=0.0,
            auto_augment=None,
        )

        metrics = validate(
            model=model,
            cap=cap,
            stride=experiment.stride,
            rasterize_method=experiment.rasterize_method,
            predict_method=experiment.predict_method,
            tracks_dropout=experiment.tracks_dropout,
            confidence=experiment.confidence,
            json_path=experiment.json_path,
            iou_thresh=experiment.iou_thresh,
            video_out_path=video_out_path,
            frame_range=experiment.val_frame_range,
        )
    finally:
        cap.release()

    append_metrics(experiment, metrics, metrics_out_path)
    return metrics
