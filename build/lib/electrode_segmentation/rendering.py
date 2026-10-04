import numpy as np
import numpy.typing as npt

def draw_mask(
    frame: npt.NDArray[np.uint8], # (H, W, 3) BGR
    mask: npt.NDArray[np.bool], # (H, W)
    colour: tuple[int, int, int] = (0, 255, 0), # BGR
    alpha: float = 0.4
):

    """
    Overlay a translucent mask onto frame in place.
    """

    if mask.shape != frame.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match frame shape {frame.shape[:2]}"
        )

    colour = np.array(colour, dtype=np.float32)

    frame[mask] = (
        (1 - alpha) * frame[mask] + alpha * colour
    ).astype(np.uint8)

