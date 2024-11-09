import time
from typing import Any, Callable, Optional, Tuple, Literal

import numpy as np
import viser

from .render_client import RenderClient, RenderConfig
from .types import CameraState




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
    config: RenderConfig = RenderConfig(),
    mode: Literal['rendering', 'training'] = 'rendering'
  ):

    self.server = server
    self.render_fn = render_fn

    self.status: Literal['training', 'paused', 'completed', 'rendering'] = 'training'    
    self.renderers: dict[int, RenderClient] = {}

    self.config = config
    self.mode = mode
    self.last_update = time.time()
    self.step = 0
    self.metrics = {"Step": self.step}

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
    with self.server.gui.add_folder("Stats", visible=self.mode == "training") as self.stats_folder:
      self.stats_text = self.server.gui.add_markdown(self.metrics_text(self.metrics))

    with self.server.gui.add_folder(
      "Training", visible=self.mode == "training"
    ) as self.training_folder:
      self.pause_train_button = self.server.gui.add_button("Pause")
      self.pause_train_button.on_click(self.on_pause_train)
      self.resume_train_button = self.server.gui.add_button("Resume")
      self.resume_train_button.visible = False
      self.resume_train_button.on_click(self.on_pause_train)

    with self.server.gui.add_folder("Rendering") as self._rendering_folder:
      self._max_img_res_slider = self.server.gui.add_slider(
        "Max Img Res", min=64, max=2048, step=1, initial_value=2048
      )
      self._max_img_res_slider.on_update(self.on_max_img_res)


  def on_pause_train(self, _):
    self.status = 'paused' if self.status == 'training' else 'training'

    self.pause_train_button.visible = (self.status != 'paused')
    self.resume_train_button.visible = (self.status == 'paused')


  def on_max_img_res(self, _):
    self.config.max_render_res = self._max_img_res_slider.value


  def disconnect_client(self, client: viser.ClientHandle):
    client_id = client.client_id
    self.renderers.pop(client_id)


  def connect_client(self, client: viser.ClientHandle):
    client_id = client.client_id
    self.renderers[client_id] = RenderClient(client=client, render_fn=self.render_fn, config=self.config)

    @client.camera.on_update
    async def _(_: viser.CameraHandle):
        self.renderers[client_id].last_moved = time.time()
        self.renderers[client_id].render(self.config.fast_render_scale)


  def update_metrics(self, metrics: dict[str, Any]):
    with self.server.atomic(), self.stats_folder:
      self.stats_text.content = self.metrics_text(metrics)


  @property
  def last_moved(self) -> float:
    times = [r.last_moved for r in self.renderers.values()]
    return max(times, default=0.0)


  def update(self, step: int, scene_changed: bool = False):
    if self.mode == "rendering":
      raise ValueError("`update` method is only available in training mode.")
    
    if self.status == 'training':
      self.step = step
      self.metrics = {"Step": self.step}
      self.update_metrics(self.metrics)
      
      for renderer in self.renderers.values():
        renderer.config = self.config
        if scene_changed or renderer.last_render > self.last_update:
          renderer.render(1.0)

      self.last_update = time.time()
      
      
  def complete(self):
    self.status = 'completed'
    self.pause_train_button = True
    self.resume_train_button = True
    self._max_img_res_slider.disabled = True
    with self.server.atomic(), self.stats_folder:
      self.stats_text.content = f"""<sub>
          Step: {self.step}\\
          Training Completed!
          </sub>"""
    
