from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.robots.fetch_robot import Fetch
from gibson2.robots.freight_robot import Freight

# from gibson2.robots.locobot_robot import locobot_robot
from gibson2.robots.tiago_single_robot import Tiago_Single

# from gibson2.robots.tia
from gibson2.simulator import Simulator
from gibson2.scenes.stadium_scene import StadiumScene

# from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
from gibson2.render.profiler import Profiler
import gibson2
import os


# config = parse_config(os.path.join(gibson2.example_config_path, "turtlebot_demo.yaml"))
# fetch_config = parse_config(
#     os.path.join(gibson2.example_config_path, "fetch_reaching.yaml")
# )
tiago_config = parse_config("tiago_stadium_config.yaml")
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


scene = StadiumScene()
# scene = StaticIndoorScene('Rs',
#   build_graph=True,
#   pybullet_load_texture=True)
s.import_scene(scene, load_texture=False)
# my_robot1 = Freight(config)
# my_robot2 = Fetch(fetch_config)
my_robot2 = Tiago_Single(tiago_config)
# s.import_robot(my_robot1)
s.import_robot(my_robot2)

# my_robot1.set_position([1, 0, 2])
my_robot2.set_position([0, 0, 0])
my_robot2.self_collision = True
# for _ in range(10):
#     obj = YCBObject('003_cracker_box')
#     s.import_object(obj)
#     obj.set_position_orientation(np.random.uniform(
#         low=0, high=2, size=3), [0, 0, 0, 1])

print(s.renderer.instances)

print("Action Space: ", my_robot2.action_space)


action_dim = 10
min_action = -1
max_action = 1
division = 25

all_actions = []
forward = np.linspace(min_action, max_action, division)
backward = np.linspace(max_action, min_action, division)
to_apply = np.concatenate((forward, backward))
for i in range(action_dim):
    actions = np.zeros((2 * division, action_dim))
    actions[:, i] = to_apply
    all_actions.append(actions)

all_actions = np.concatenate(all_actions)
all_actions = all_actions.astype(np.float32)
all_actions = all_actions.tolist()

rgbs = []
print("Applying some action on the robot :)")
for i in range(len(all_actions)):
    # my_robot1.apply_action([0.1, 0.01])
    sampl = np.random.uniform(low=-1, high=1, size=(10,)).tolist()
    my_robot2.apply_action(sampl)

    s.step()
    # with Profiler('Simulator step'):
    #     s.step()
    #     rgb = s.renderer.render_robot_cameras(modes=('rgb'))[0]
    #     rgbs.append(rgb)

# print(rgbs)
# print("Type of image", type(rgbs[0]))
