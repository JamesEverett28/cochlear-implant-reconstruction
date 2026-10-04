from electrode_detection.inference import video_inference
from electrode_segmentation.sam2_points import points2mask
from electrode_detection.dataset import generate_dataset_from_json
from electrode_detection.fourchannel_model import FourChannelsOBBYolo
from electrode_detection.experiments import Experiment
from electrode_detection.rendering import draw_video_predictions

import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path



base = Experiment(
    project="side1",
    name="base",
    json_path=Path("datasets/side1/label-studio.json"),
    video_in_path=Path("videos/3-view/angled-light/side1-start.mp4"),
    train_frame_range=(1, 1060),
    val_frame_range=(1190, 1731), 
    model_path=Path("yolo26n-obb.pt"),  
    epochs=50,
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

dataset_dir = Path("datasets") / base.project / base.name
run_dir = Path("experiments") / base.project / base.name


cap = cv2.VideoCapture(base.video_in_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not load video: {base.video_in_path}")

try:

    """
    generate_dataset_from_json(
        json_path=base.json_path,
        cap=cap,
        dataset_dir=dataset_dir,
        stride=base.stride,
        fourth_channel=base.fourth_channel,
        rasterize_method=base.rasterize_method,
        pos_std=base.pos_std,
        size_std=base.size_std,
        angle_std=base.angle_std,
        blank_prior_prob=base.blank_prior_prob,
        frame_range=base.train_frame_range
    )

    model: YOLO | FourChannelsOBBYolo

    if base.fourth_channel:

        model = FourChannelsOBBYolo(base.model_path)

    else:
        
        model = YOLO(base.model_path)

    model.train(
        data=dataset_dir / "data.yaml",
        batch=base.batch,
        imgsz=base.imgsz,
        epochs=base.epochs,
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
    """

    model = YOLO("experiments/side1/base/weights/best.pt")

    predictions = video_inference(
        model=model,
        cap=cap,
        stride=base.stride,
        rasterize_method=base.rasterize_method,
        predict_method=base.predict_method,
        tracks_dropout=base.tracks_dropout,
        confidence=base.confidence
    )


    points = {}

    for prediction in predictions:
        if not prediction.obbs:
            continue

        points[prediction.frame_num - 1] = np.stack(
            [obb.xywhr[:2] for obb in prediction.obbs],
            axis=0,
        )


    points2mask(
        cap=cap,
        out_path="out.mp4",
        points=points,
        stride=base.stride,
        model_size="large"
    )

finally:
    cap.release()