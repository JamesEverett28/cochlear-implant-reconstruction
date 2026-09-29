import cv2
from pathlib import Path
import numpy as np

from .fourchannels_yolo import FourChannelsOBBYolo

def rasterize_obbs(obbs, img_w, img_h, normalized=True):

    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for obb in obbs:

        obb = obb.copy()

        if normalized:
            
            obb[:, 0] *= img_w
            obb[:, 1] *= img_h

        obb = np.round(obb).astype(np.int32)

        cv2.fillPoly(mask, [obb], 255)

    return mask

scale = 4
stride = 5

in_path = Path("videos/3-view/angled-no-light/side1-crop.mp4")
out_path = Path("predictions/angled-no-light/side1-crop-4channels.mp4") 
out_path.parent.mkdir(exist_ok=True, parents=True)

model = FourChannelsOBBYolo("runs/obb/side1-insertion/epochs200-4channels/weights/best.pt")

cap = cv2.VideoCapture(in_path)
if not cap.isOpened():
    raise RuntimeError("Could not load video")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
in_fps = cap.get(cv2.CAP_PROP_FPS)

out_fps = in_fps / stride
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(out_path, fourcc, out_fps, (frame_width*scale, frame_height*scale), isColor=True)
if not out.isOpened():
    raise RuntimeError("Could not open output video")

frame_num = 0
prev_boxes = np.empty((0, 4, 2))

while True:

    ret, frame = cap.read()
    if not ret:
        break

    if frame_num % stride == 0:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        prior = rasterize_obbs(prev_boxes, frame_width, frame_height, normalized=False)
    
        channels = np.dstack((rgb, prior))

        result = model(channels)[0]

        # upsample just for display
        large = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        boxes = result.obb.xyxyxyxy.cpu().numpy()

        prev_boxes = boxes.copy()

        for box in boxes:

            points = np.round(box * scale).astype(np.int32)

            cv2.polylines(
                large,
                [points],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA
            )

        out.write(large)

    frame_num += 1
    print(f"Processed {frame_num}/{frame_count}")

cap.release()
out.release()


