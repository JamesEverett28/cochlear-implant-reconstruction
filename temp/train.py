from ultralytics import YOLO
from .fourchannels_yolo import FourChannelsOBBYolo
from pathlib import Path

data_path = Path("datasets/side1-insertion/4channels/data.yaml")
project = "side1-insertion"
name = "epochs200-4channel"


model = FourChannelsOBBYolo("yolo26n-obb.pt")
result = model.train(
    data=data_path,
    batch=16,
    imgsz=256,
    epochs=200,
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



