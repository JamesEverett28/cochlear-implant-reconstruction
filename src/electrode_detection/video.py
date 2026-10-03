from pathlib import Path
import subprocess

import cv2


def frame_diff(
    in_path: str | Path,
    out_path: str | Path,
    gray_scale: bool,
    thresh: int | None = None
) -> None:

    """
    Subtract the first frame from each subsequent frame in a video.
    Optionally gray scale and threshold.
    """

    in_path = Path(in_path)
    out_path = Path(out_path)

    out_path.parent.mkdir(exist_ok=True, parents=True)

    if thresh is not None and not gray_scale:
        raise RuntimeError("Can only threshold frames if gray-scaled")

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError("Could not load video")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        out_path, 
        fourcc, 
        fps, 
        (frame_width, frame_height), 
        isColor = not gray_scale
    )

    if not out.isOpened():
        raise RuntimeError("Could not open output video")

    ref_frame = None

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if ref_frame is None:
            ref_frame = frame

        diff = cv2.absdiff(frame, ref_frame)

        if gray_scale:
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            if thresh is not None:
                _, diff = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

        out.write(diff)

    cap.release()
    out.release()


def crop_video(
    in_path: str | Path,
    out_path: str | Path,
    w:int, h:int, x:int, y:int,
    crf: int = 18,
    preset: str = "fast"
) -> None:

    """
    Crop a video using ffmpeg.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    in_path = str(in_path)
    out_path = str(out_path)

    crop = f"{w}:{h}:{x}:{y}"

    subprocess.run(
        [
            "ffmpeg",
            "-i", in_path,
            "-vf", f"crop={crop}",
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            out_path,
        ],
        check=True,
    )