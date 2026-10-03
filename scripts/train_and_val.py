from pathlib import Path

from ultralytics import YOLO
import cv2

from electrode_detection.fourchannel_model import FourChannelsOBBYolo
from electrode_detection.dataset import generate_dataset_from_json, annotations_dict_from_json
from electrode_detection.inference import video_inference
from electrode_detection.validate import get_video_metrics
from electrode_detection.rendering import draw_video_predictions

train_frame_range = (4, 49)
val_frame_range = (64, 89)

project = "side1"
name = "experiment_1"
json_path = Path("datasets/side1-insertion/label-stuio.json")
video_in_path = Path("videos/3-view/angled-light/side1-crop.mp4")
metrics_out_path = Path("runs/obb/metrics.csv")
video_out_path = Path(f"runs/obb/{project}/{name}/predictions.mp4")
project = "side1"
name = "experiment_1"
epochs = 200
perturb = True 
fourth_channel = True
rasterize_method = "gaussian"
stride = 5
confidence = 0.25
predict_method = "linear"
tracks_dropout = 1
iou_thresh = 0.95
dataset_dir = Path(f"datasets/{project}/{name}")

def main() -> None:

    cap = cv2.VideoCapture(video_in_path)
    if not cap.isOpened():
            raise RuntimeError("Could not load video")

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    generate_dataset_from_json(
        json_path=json_path,
        cap=cap,
        dataset_dir=dataset_dir,
        stride=stride,
        fourth_channel=fourth_channel,
        frame_range=train_frame_range
    )

    if fourth_channel:

        model = FourChannelsOBBYolo("yolo26n-obb.pt")

    else:

        model = YOLO("yolo26n-obb.pt")

    model.train(
        data=dataset_dir / "data.yaml",
        batch=16,
        imgsz=256,
        epochs=epochs,
        project=project,
        name=name,
        device="mps",
        patience=10,
        val=False,

        # disable augmentations
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


val_predictions = video_inference(
    model=model,
    cap=cap,
    stride=stride,
    rasterize_method=rasterize_method,
    predict_method=predict_method,
    tracks_dropout=tracks_dropout,
    confidence=confidence,
    frame_range=val_frame_range
)

draw_video_predictions(
    cap=cap,
    out_path=video_out_path
)

val_labels = annotations_dict_from_json(
    json_path=json_path,
    stride=stride,
    img_h=img_h,
    img_w=img_w,
    frame_range=val_frame_range
)

video_metrics = get_video_metrics(
    predictions=val_predictions,
    labels=val_labels,
    iou_thresh=iou_thresh,
)

# TODO: write video_metrics to metrics_out_path






    