from pathlib import Path

from utils.video import crop_video

in_path = Path("videos/3-view/angled-light/side1.mp4")
out_path = Path("videos/3-view/angled-light/side1-start1.mp4")

#w, h, x, y = 255, 80, 195, 200

w, h, x, y = 400, 200, 100, 110

def main() -> None:

    crop_video(
        in_path=in_path,
        out_path=out_path,
        w=w, h=h, x=x, y=y
    )

if __name__ == "__main__":

    main()