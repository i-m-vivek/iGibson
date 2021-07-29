from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.simulator import Simulator
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
from gibson2.render.profiler import Profiler
import gibson2
import os
import matplotlib.pyplot as plt


from gibson2.robots.fetch_robot import Fetch
from gibson2.robots.tiago_single_robot import Tiago_Single

config = parse_config(os.path.join(gibson2.example_config_path, "turtlebot_demo.yaml"))
settings = MeshRendererSettings(enable_shadow=False, msaa=False)
s = Simulator(
    mode="gui", image_width=256, image_height=256, rendering_settings=settings
)

# scene = StaticIndoorScene("Rs", build_graph=True, pybullet_load_texture=True)
scene = EmptyScene()
s.import_scene(scene)
fetch = Fetch(config)
# fetch = Turtlebot(config)
s.import_robot(fetch)

print(fetch.parts.keys())

fetch.set_position([0, 0, 0])
print(fetch.ordered_joints)

for i in range(10):
    s.step()
    print(fetch.get_position())
print(s.renderer.instances)
