import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import cv2

from .rendering import rasterize_prior



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



def ls_rect_to_obb(
    x:float, y:float, w:float, h:float, r:float,
    img_w: int, 
    img_h: int,
    normalize: bool
) -> npt.NDArray[np.float32]:

    """
    Convert label-studio xywhr format to corner coords between 0-1.
    """

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
    ], dtype=np.float32)

    corners = corners @ R.T
    corners += np.array([x, y])

    if normalize:

        corners[:, 0] /= img_w
        corners[:, 1] /= img_h

    return corners



def annotations_dict_from_json(
    json_path: str | Path,
    stride: int,
    img_h: int,
    img_w: int,
    frame_range: tuple[int, int] | None = None,
    normalize: bool = True
) -> dict[int, list[npt.NDArray[np.float32]]]:

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

                obb = ls_rect_to_obb(x, y, w, h, r, img_w, img_h, normalize=normalize)

                frame_annotations[frame_num].append(obb)
    
    return frame_annotations


def generate_dataset_from_json(
    json_path: str | Path,
    video_path: str | Path,
    dataset_dir : str | Path,
    stride: int,
    fourth_channel: bool,
    frame_range: tuple[int, int] | None = None,
    rasterize_method: str = "direct",
    sigma_scale: float = 0.5,
    truncate: float = 3.0,
    perturb: bool = True,
    
) -> None:

    """
    Generate jpg/tiff and txt files from label-studio json
    """

    cap  = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not load video")

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

            prev_obbs = frame_annotations.get(frame_num - stride, [])
            prior = rasterize_obbs(prev_obbs, img_h, img_w, normalized=True)

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
                
                coords = " ".join(
                    f"{coord:.6f}" for coord in obb.ravel()
                )

                f.write(f"0 {coords}\n")

    cap.release()