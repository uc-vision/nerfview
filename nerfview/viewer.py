from functools import partial
import time
from threading import Lock
from typing import Any, Callable, Optional, Tuple

import numpy as np
import viser
from nerfview.types import CameraState

from .render_client import RenderClient, RenderConfig


class Viewer(object):
  """This is the main class for working with nerfview viewer.

  On instantiation, it (a) binds to a viser server and (b) creates a set of
  GUIs depending on its mode. After user connecting to the server, viewer
  renders and servers images in the background based on the camera movement.

  Args:
    server (viser.ViserServer): The viser server object to bind to.
    render_fn (Callable): A function that takes a camera state and image
      resolution as input and returns an image and (optionally) a depth map.
  """

  def __init__(
    self,
    server: viser.ViserServer,
    render_fn: Callable[
      [CameraState, Tuple[int, int]],
      Tuple[np.ndarray, Optional[np.ndarray]], # image h,w,3, depth h,w
    ],
    config: RenderConfig = RenderConfig()
  ):
    # Public states.
    self.server = server
    self.render_fn = render_fn

    self.paused = False        
    self.renderers: dict[int, RenderClient] = {}

    self.config = config
    self.last_update = time.time()

    server.on_client_disconnect(self.disconnect_client)
    server.on_client_connect(self.connect_client)

    self.define_guis()


  def metrics_text(self, metrics: dict[str, Any]) -> str:
    def f(key: str, value: Any) -> str:
      if isinstance(value, float):
        return f"{key}: {value:.3f}"
      elif isinstance(value, int):
        return f"{key}: {value:d}"
      else:
        return f"{key}: {value}"

    return f"""<sub>{" ".join(f(k, v) for k, v in metrics.items())}</sub>"""


  def define_guis(self):
    with self.server.gui.add_folder("Stats") as self.stats_folder:
      self.stats_text = self.server.gui.add_markdown(self.metrics_text(self.metrics))

    with self.server.gui.add_folder(
      "Training", visible=self.mode == "training"
    ) as self.training_folder:
      self.pause_train_button = self.server.gui.add_button("Pause")
      self.pause_train_button.on_click(partial(self.on_pause_train, True))
      self.resume_train_button = self.server.gui.add_button("Resume")
      self.resume_train_button.visible = False
      self.resume_train_button.on_click(partial(self.on_pause_train, False))


    with self.server.gui.add_folder("Rendering") as self._rendering_folder:
      self._max_img_res_slider = self.server.gui.add_slider(
        "Max Img Res", min=64, max=2048, step=1, initial_value=2048
      )
      self._max_img_res_slider.on_update(self.on_max_img_res)

  def on_pause_train(self, pause: bool):
    self.paused = pause

    self.pause_train_button.visible = not pause
    self.resume_train_button.visible = pause

  def on_max_img_res(self, res: int):
    self.config.max_render_res = res

  def disconnect_client(self, client: viser.ClientHandle):
    client_id = client.client_id
    self.renderers.pop(client_id)

  def connect_client(self, client: viser.ClientHandle):
    client_id = client.client_id
    self.renderers[client_id] = RenderClient(client=client, render_fn=self.render_fn, config=self.config)


  def update_metrics(self, metrics: dict[str, Any]):
    with self.server.atomic(), self.stats_folder:
      self.stats_text.content = self.metrics_text(metrics)


  @property
  def last_moved(self) -> float:
    times = [r.last_moved for r in self.renderers.values()]
    return max(times, default=0.0)


  def update(self, scene_changed: bool = False):
    for renderer in self.renderers.values():
      if scene_changed or renderer.last_render > self.last_update:
        renderer.render(1.0)

    self.last_update = time.time()
