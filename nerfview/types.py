from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np


@dataclass
class CameraState(object):
    fov: float
    aspect: float
    camera_t_world: np.ndarray #  4x4

    # @property
    def projection(self, image_size: Tuple[int, int]) -> np.ndarray:
      w, h = image_size
      f = h / 2.0 / np.tan(self.fov / 2.0)

      return np.array([f, f, w/2., h/2.], dtype=np.float32)



