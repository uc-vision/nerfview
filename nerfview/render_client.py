
from dataclasses import dataclass
import threading
import time
from typing import Callable, Literal, Optional, Tuple

import numpy as np

import viser
import viser.transforms as vt

from nerfview.types import CameraState


@dataclass
class RenderConfig:
    jpeg_quality: int = 70
    max_render_res: int = 2048
    fast_render_scale: float = 0.5

class RenderClient():
    def __init__(
        self,
        client: viser.ClientHandle,
        render_fn: Callable[[CameraState, Tuple[int, int]], Tuple[np.ndarray, Optional[np.ndarray]]],
        config: RenderConfig,
    ):

        self.client = client
        self.last_render = time.time()
        self.last_moved = time.time()

        self.config = config
        self.render_fn = render_fn


    def camera_moved(self, _: viser.CameraHandle):
        self.last_moved = time.time()
        self.render(self.config.fast_render_scale)
        

    def get_camera_state(self) -> CameraState:
        camera = self.client.camera
        camera_t_world = np.concatenate(
            [
                np.concatenate(
                    [vt.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],            
            0,
        )
        
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            camera_t_world=camera_t_world,
        )


    def get_image_size(self, max_size:int, aspect: float) -> Tuple[int, int]:    
        if aspect > 1:
            return max_size, int(max_size / aspect)
        else:
            return int(max_size * aspect), max_size



    def render(self, image_scale: float):
        self.last_render = time.time()

        camera = self.get_camera_state()
        image_size = self.get_image_size(image_scale * self.config.max_render_res, camera.aspect)
        img, depth = self.render_fn(camera, image_size)
  
        self.client.scene.set_background_image(
            image=img, format="jpeg", 
            jpeg_quality=self.config.jpeg_quality, 
            depth=depth)

