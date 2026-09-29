import cv2
import numpy as np
from pathlib import Path 

in_path = Path("videos/3-view/angled-light/combined.avi")
out_path = in_path.parent / "combined-diff-gray2.avi"
cap  = cv2.VideoCapture(in_path)

if not cap.isOpened():
    raise RuntimeError("Could not load video")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height), isColor=False)

if not out.isOpened():
    raise RuntimeError("Could not open output video")

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_num == 0:
        frame_0 = frame

    diff = cv2.absdiff(frame, frame_0)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.multiply(diff, 2)
    out.write(diff)

    frame_num += 1
    print(f"Processed {frame_num}/{frame_count}", end="\r")

cap.release()
out.release()