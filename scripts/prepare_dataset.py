from pathlib import Path

from electrode_detection.dataset import generate_dataset_from_json

json_path = Path("datasets/side1-insertion/label-studio.json")
video_path = Path("videos/3-view/angled-light/side1-crop.mp4")
dataset_dir = Path("datasets/side1-insertion/4channels")
stride = 5
fourth_channel = True

def main() -> None:

    generate_dataset_from_json(
        json_path=json_path,
        video_path=video_path,
        dataset_dir=dataset_dir,
        stride=stride,
        fourth_channel=fourth_channel
    )

if __name__ == "__main__":

    main()

    


