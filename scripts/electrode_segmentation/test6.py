"""Crop a video and upscale it to 1000 px while preserving its aspect ratio."""

from pathlib import Path
import subprocess


IN_PATH = Path("videos/3-view/angled-light/side1.mp4")
OUT_PATH = Path("videos/3-view/angled-light/side1-crop-square-1000.mp4")

# Crop dimensions and origin, matching scripts/utils/crop_video.py.
W, H, X, Y = 255, 80, 195, 200
OUTPUT_SIZE = 1000


def main() -> None:
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

    # -2 preserves the aspect ratio and rounds the calculated height to an
    # even value required by the H.264 encoder.
    filters = (
        f"crop={W}:{H}:{X}:{Y},"
        f"scale={OUTPUT_SIZE}:-2"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(IN_PATH),
            "-vf",
            filters,
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            str(OUT_PATH),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
