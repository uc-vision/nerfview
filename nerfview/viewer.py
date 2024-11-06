import dataclasses
import time
from threading import Lock
from typing import Any, Callable, Literal, Optional, Tuple, Union

import numpy as np
import viser
import viser.transforms as vt
from jaxtyping import Float32, UInt8

from ._renderer import Renderer, RenderTask


@dataclasses.dataclass
class CameraState(object):
    fov: float
    aspect: float
    c2w: Float32[np.ndarray, "4 4"]

    def get_K(self, img_wh: Tuple[int, int]) -> Float32[np.ndarray, "3 3"]:
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return K


Status = Literal["rendering", "preparing", "training", "paused", "completed"]


VIEWER_LOCK = Lock()


def with_viewer_lock(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        with VIEWER_LOCK:
            return fn(*args, **kwargs)

    return wrapper


class Viewer(object):
    """This is the main class for working with nerfview viewer.

    On instantiation, it (a) binds to a viser server and (b) creates a set of
    GUIs depending on its mode. After user connecting to the server, viewer
    renders and servers images in the background based on the camera movement.

    Args:
        server (viser.ViserServer): The viser server object to bind to.
        render_fn (Callable): A function that takes a camera state and image
            resolution as input and returns an image as a uint8 numpy array.
            Optionally, it can return a tuple of two images, where the second image
            is a float32 numpy depth map.
        mode (Literal["training", "rendering"]): The mode of the viewer.
            Support rendering and training. Defaults to "rendering".
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable[
            [CameraState, Tuple[int, int]],
            Union[
                UInt8[np.ndarray, "H W 3"],
                Tuple[UInt8[np.ndarray, "H W 3"], Optional[Float32[np.ndarray, "H W"]]],
            ],
        ],
        mode: Literal["rendering", "training"] = "rendering"
    ):
        # Public states.
        self.server = server
        self.render_fn = render_fn
        self.mode = mode
        self.lock = VIEWER_LOCK
        self.status: Status = mode
        
        # Private states.
        self._renderers: dict[int, Renderer] = {}

        self._step: int = 0
        self._last_update_step: int = 0
        self._last_move_time: float = 0.0
      

        server.on_client_disconnect(self._disconnect_client)
        server.on_client_connect(self._connect_client)

        self._define_guis()


    def _metrics_text(self, metrics: dict[str, Any]) -> str:
        def f(key: str, value: Any) -> str:
            if isinstance(value, float):
                return f"{key}: {value:.3f}"
            elif isinstance(value, int):
                return f"{key}: {value:d}"
            else:
                return f"{key}: {value}"

        return f"""<sub>{" ".join(f(k, v) for k, v in metrics.items())}</sub>"""


    def _define_guis(self):
        with self.server.gui.add_folder("Stats") as self._stats_folder:
            self._stats_text = self.server.gui.add_markdown(self._metrics_text(self.metrics))

        with self.server.gui.add_folder(
            "Training", visible=self.mode == "training"
        ) as self._training_folder:
            self._pause_train_button = self.server.gui.add_button("Pause")
            self._pause_train_button.on_click(self._on_pause_train)
            self._resume_train_button = self.server.gui.add_button("Resume")
            self._resume_train_button.visible = False
            self._resume_train_button.on_click(self._on_resume_train)


        with self.server.gui.add_folder("Rendering") as self._rendering_folder:
            self._max_img_res_slider = self.server.gui.add_slider(
                "Max Img Res", min=64, max=2048, step=1, initial_value=2048
            )
            self._max_img_res_slider.on_update(self.rerender)

    def _on_pause_train(self, _):
        self.status = "paused"
        self._pause_train_button.visible = False
        self._resume_train_button.visible = True

    def _on_resume_train(self, _):
        self.status = "training"
        self._pause_train_button.visible = True
        self._resume_train_button.visible = False


    def rerender(self, _):
        clients = self.server.get_clients()
        for client_id in clients:
            camera_state = self.get_camera_state(clients[client_id])
            assert camera_state is not None
            self._renderers[client_id].submit(RenderTask("rerender", camera_state))

    def _disconnect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id].running = False
        self._renderers.pop(client_id)

    def _connect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id] = Renderer(
            viewer=self, client=client, lock=self.lock
        )
        self._renderers[client_id].start()

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            self._last_move_time = time.time()
            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                self._renderers[client_id].submit(RenderTask("move", camera_state))

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate(
                    [vt.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],            # self._train_util_slider = self.server.gui.add_slider(
            #     "Train Util", min=0.0, max=1.0, step=0.05, initial_value=0.9
            # )
            # self._train_util_slider.on_update(self.rerender)
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
        )

    def update_training(self, metrics: dict[str, Any]):
      
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = self._stats_text_fn(metrics)

        if len(self._renderers) == 0:
            return
        
        # Stop training while user moves camera to make viewing smoother.
        while time.time() - self._last_move_time < 0.1:
            time.sleep(0.05)

            
        if self.state.status == "training":        
          clients = self.server.get_clients()
          for client_id in clients:
              camera_state = self.get_camera_state(clients[client_id])
              assert camera_state is not None
              self._renderers[client_id].submit(
                  RenderTask("update", camera_state)
              )
          with self.server.atomic(), self._stats_folder:
              self._stats_text.content = self._stats_text_fn(metrics)

    def complete(self):
        self.state.status = "completed"
        self._pause_train_button.disabled = True
        self._resume_train_button.disabled = True
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = f"""<sub>
                Step: {self._step}\\
                Training Completed!
                </sub>"""
