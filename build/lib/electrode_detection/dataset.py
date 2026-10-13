import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from collections import defaultdict

import numpy as np
import cv2

from .obb import OBB
from .prior import get_prior_from_obbs



def export_ls_json(
    refresh_token: str,
    out_path: str | Path,
    project_id: int,
    base_url: str = "http://localhost:8080",
) -> None:
    
    """
    Export interpolated Label Studio annotations for one project.
    """

    base_url = base_url.rstrip("/")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    refresh_request = Request(
        f"{base_url}/api/token/refresh",
        data=json.dumps({"refresh": refresh_token}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(refresh_request) as response:
        access_token = json.load(response)["access"]

    query = urlencode({
        "exportType": "JSON",
        "interpolate_key_frames": "true",
    })
    export_request = Request(
        f"{base_url}/api/projects/{project_id}/export?{query}",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    with urlopen(export_request) as response:
        out_path.write_bytes(response.read())



def annotations_dict_from_json(
    json_path: str | Path,
    stride: int,
    img_h: int,
    img_w: int,
    frame_range: tuple[int, int] | None = None,
) -> dict[int, list[OBB]]:

    """
    Create frame annotations dict from label-studio json.
    """

    json_path = Path(json_path)
    
    frame_annotations = defaultdict(list)

    with json_path.open("r", encoding="utf-8") as f:
        task = json.load(f)
    
        for electrode in task[0]["annotations"][0]["result"]:
    
            if electrode["type"] != "videorectangle":
                continue
    
            for instance in electrode["value"]["sequence"]:
    
                frame_num = instance["frame"]
                if (frame_num-1) % stride != 0:
                    continue
                if frame_range is not None and (frame_num < frame_range[0] or frame_num > frame_range[1]):
                    continue
    
                x, y, w, h, r = [instance[u] for u in ["x", "y", "width", "height", "rotation"]]

                x = (x + w/2) / 100 * img_w
                y = (y + h/2) / 100 * img_h
                w = w / 100 * img_w
                h = h / 100 * img_h

                frame_annotations[frame_num].append(
                    OBB(np.array([x, y, w, h, np.deg2rad(r)], dtype=np.float32))
                )

    return frame_annotations


def write_dataset_config(dataset_dir: str | Path, fourth_channel: bool) -> Path:

    """
    Create the Ultralytics dataset.yaml configuration beside generated samples.
    """

    dataset_yaml = Path(dataset_dir) / "data.yaml"
    channels = "channels: 4\n\n" if fourth_channel else ""
    dataset_yaml.write_text(
        "train: images\n"
        "val: images\n\n"
        f"{channels}"
        "names:\n"
        "  0: Electrode\n",
        encoding="utf-8",
    )
    return dataset_yaml


def generate_dataset_from_json(
    json_path: str | Path,
    cap: cv2.VideoCapture,
    dataset_dir : str | Path,
    stride: int,
    fourth_channel: bool,
    rasterize_method: str = "direct",
    blank_prior_prob:float = 0.2,
    pos_std: float = 0.05,
    size_std: float = 0.05,
    angle_std: float = np.deg2rad(3),
    frame_range: tuple[int, int] | None = None,

) -> None:

    """
    Generate jpg/tiff and txt files from label-studio json
    """

    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_annotations = annotations_dict_from_json(
        json_path=json_path,
        stride=stride,
        img_h=img_h,
        img_w=img_w,
        frame_range=frame_range
    )

    images_dir = Path(dataset_dir) / "images"
    labels_dir = Path(dataset_dir) / "labels"

    images_dir.mkdir(exist_ok=True, parents=True)
    labels_dir.mkdir(exist_ok=True)

    for image in images_dir.iterdir():
        image.unlink()

    for label in labels_dir.iterdir():
        label.unlink()

    write_dataset_config(dataset_dir, fourth_channel)

    for frame_num in sorted(frame_annotations):
    
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {frame_num}")
            continue

        file_stem = f"frame_{frame_num:06d}"
        label_path = labels_dir / f"{file_stem}.txt"

        if not fourth_channel:

            image_path = images_dir / f"{file_stem}.jpg"
            cv2.imwrite(image_path, frame)

        else:

            image_path = images_dir / f"{file_stem}.tiff"

            prior = get_prior_from_obbs(
                obbs=frame_annotations[frame_num],
                img_h=img_h,
                img_w=img_w,
                rasterize_method=rasterize_method,
                perturb=True,
                pos_std=pos_std,
                size_std=size_std,
                angle_std=angle_std,
                blank_prior_prob=blank_prior_prob
            )

            channels = [
                frame[:, :, 2],
                frame[:, :, 1],
                frame[:, :, 0],
                prior
            ]

            cv2.imwritemulti(
                image_path, 
                channels, 
                [cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_LZW]
            )

        with label_path.open("w", encoding="utf-8") as f:

            for obb in frame_annotations[frame_num]:

                corners = obb.corners_normalized(img_h, img_w)
                
                coords = " ".join(
                    f"{coord:.6f}" for coord in corners.ravel()
                )

                f.write(f"0 {coords}\n")
