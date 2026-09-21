import json
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

def label_studio_rect_to_obb(x, y, w, h, r, img_w, img_h):

    x = x / 100 * img_w
    y = y / 100 * img_h
    w = w / 100 * img_w
    h = h / 100 * img_h

    theta = np.deg2rad(r)

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])

    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    corners = corners @ R.T
    corners += np.array([x, y])
    corners[:, 0] /= img_w
    corners[:, 1] /= img_h

    return corners


stride = 5

labels_json = Path("datasets/side1-insertion/label-studio.json")
video_path = Path("videos/3-view/angled-light/side1-crop.mp4")
images_dir = Path("datasets/side1-insertion/standard/images")
labels_dir = Path("datasets/side1-insertion/standard/labels")

images_dir.mkdir(exist_ok=True)
labels_dir.mkdir(exist_ok=True)

frame_annotations = defaultdict(list)

cap  = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Could not load video")

img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with labels_json.open("r", encoding="utf-8") as f:
    task = json.load(f)

for electrode in task[0]["annotations"][0]["result"]:

    if electrode["type"] != "videorectangle":
        continue

    for instance in electrode["value"]["sequence"]:

        frame_num = instance["frame"]
        if frame_num % stride != 0:
            continue

        x, y, w, h, r = [instance[u] for u in ["x", "y", "width", "height", "rotation"]]
        obb = label_studio_rect_to_obb(x, y, w, h, r, img_w, img_h)

        frame_annotations[frame_num].append(obb)

for frame_num in sorted(frame_annotations):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
    ret, frame = cap.read()
    if not ret:
        print(f"Could not read frame {frame_num}")
        continue

    file_stem = f"frame_{frame_num:06d}"
    image_path = images_dir / f"{file_stem}.jpg"
    label_path = labels_dir / f"{file_stem}.txt"

    cv2.imwrite(image_path, frame)

    with label_path.open("w", encoding="utf-8") as f:

        for obb in frame_annotations[frame_num]:
            
            coords = " ".join(
                f"{coord:.6f}" for coord in obb.ravel()
            )

            f.write(f"0 {coords}\n")

cap.release()


