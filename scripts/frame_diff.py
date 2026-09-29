from pathlib import Path

from electrode_detection.video import frame_diff

in_path = Path("videos/3-view/angled-light/combined.avi")
out_path = in_path.parent / "combined-diff-gray2.avi"
gray_scale = False
thresh = None

def main() -> None:

    frame_diff(
        in_path=in_path,
        out_path=out_path,
        gray_scale=gray_scale,
        thresh=thresh
    )

if __name__ == "__main__":

    main()

    