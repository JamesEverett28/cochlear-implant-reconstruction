from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from ultralytics.utils.ops import xywhr2xyxyxyxy

@dataclass
class OBB:

    xywhr: npt.NDArray[np.float32] # cx, cy, w, h, radians; pixels



    @property
    def corners(self) -> npt.NDArray[np.float32]:

        return xywhr2xyxyxyxy(self.xywhr)



    def corners_normalized(
        self,
        img_h: int,
        img_w: int
    ) -> npt.NDArray[np.float32]:

        corners = self.corners.copy()

        corners[:, 0] /= img_w
        corners[:, 1] /= img_h

        return corners



    def copy(self) -> OBB:

        return OBB(self.xywhr.copy())
