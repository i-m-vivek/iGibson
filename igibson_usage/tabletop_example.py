from gibson2.robots.tiago_single_robot import Tiago_Single
from gibson2.scenes.stadium_scene import StadiumScene
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from gibson2.simulator import Simulator
import numpy as np
import os
from gibson2.utils.utils import parse_config
import gibson2
from gibson2.objects.articulated_object import ArticulatedObject


config = parse_config(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs",
        "tiago_stadium_config.yaml",
    )
)
settings = MeshRendererSettings(
    enable_shadow=False, msaa=False, enable_pbr=True, texture_scale=1.0
)
s = Simulator(
    mode="gui",
    image_width=128,
    image_height=128,
    vertical_fov=90,
    rendering_settings=settings,
    physics_timestep=1 / 240.0,
    render_timestep=1 / 10.0,
    render_to_tensor=False,
)


scene = EmptyScene()
table_path = "/home/vkmittal14/WORKSPACE/TUD/vk_igibson/iGibson/gibson2/data/ig_dataset/objects/table/9ad91992184e2b3e283b00891f680579/9ad91992184e2b3e283b00891f680579.urdf"
table = ArticulatedObject(table_path, 2)
table.load()
table.set_position([2, 0, 0.3])
myrobot = Tiago_Single(config)
s.import_scene(scene, load_texture=False)
s.import_object(table)
s.import_robot(myrobot)


for _ in range(10000):  # at least 100 seconds
    sampl = np.random.uniform(low=-1, high=1, size=(14,)).tolist()
    myrobot.apply_action(sampl)
    s.step()
