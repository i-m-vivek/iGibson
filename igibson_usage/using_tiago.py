from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.robots.fetch_robot import Fetch
from gibson2.robots.freight_robot import Freight
from gibson2.robots.tiago_single_robot import Tiago_Single

# from gibson2.robots.tia
from gibson2.simulator import Simulator
from gibson2.scenes.stadium_scene import StadiumScene

from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
from gibson2.render.profiler import Profiler
import gibson2
import os
from gibson2.external.pybullet_tools.utils import (
    joints_from_names,
    set_joint_positions,
)
from gibson2.utils.utils import rotate_vector_3d


config = parse_config("configs/tiago_stadium_config.yaml")

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


def do_sim():
    for i in range(10):
        s.step()


def get_quat(theta):
    quat = [
        0.0,
        0.0,
        np.sin(np.deg2rad(theta / 2)),
        np.cos(np.deg2rad(theta / 2)),
    ]
    return quat


scene = StadiumScene()
s.import_scene(scene, load_texture=False)
my_robot = Tiago_Single(config)
my_robot.self_collision = True
s.import_robot(my_robot)
import pdb
my_robot.set_position([0, 1, 0])
my_robot.set_orientation([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])

pdb.set_trace()
robot_pos = my_robot.get_position()
roll, pitch, yaw = my_robot.get_rpy()
target_pos = np.array([1.25, -2, 0])
# target_pos_local = rotate_vector_3d(target_pos - robot_pos, roll, pitch, yaw)
target_pos_local = np.array([2.0, 2.0, 0.1])
world_in_robot = rotate_vector_3d(np.array([0,0,0]) - robot_pos, roll, pitch, yaw)
robot_to_world = rotate_vector_3d(target_pos_local - world_in_robot, roll, pitch, 2*np.pi -yaw)
print(robot_to_world)

# print("Action Space: ", my_robot.action_space)


# action_dim = 14
# min_action = -1
# max_action = 1
# division = 100

# all_actions = []
# forward = np.linspace(min_action, max_action, division)
# backward = np.linspace(max_action, min_action, division)
# to_apply = np.concatenate((forward, backward))
# for i in range(14):
#     actions = np.zeros((2 * division, action_dim))
#     actions[:, i] = to_apply
#     all_actions.append(actions)

# all_actions = np.concatenate(all_actions)
# all_actions = all_actions.astype(np.float32)
# all_actions = all_actions.tolist()

# rgbs = []
# robot_id = s.robots[0].robot_ids[0]

# arm_joints = joints_from_names(
#     robot_id,
#     [
#         "torso_lift_joint",
#         "arm_1_joint",
#         "arm_2_joint",
#         "arm_3_joint",
#         "arm_4_joint",
#         "arm_5_joint",
#         "arm_6_joint",
#         "arm_7_joint",
#     ],
# )

# rest_position = np.array([0, np.pi, -np.pi / 2, 0, np.pi / 2, 0, 0, 0])


# print(my_robot.parts.keys())
# print("Applying some action on the robot :)")
# joint_position = [
#         0,
#         0,
#         0.15,
#         0,
#         0,
#         0,
#         0,
#         0,
#     ]
# joint_position = np.array(joint_position) + rest_position
# set_joint_positions(robot_id, arm_joints, joint_position.tolist())
# print(my_robot.parts)
# while True:
#     # joint_position = [
#     #     np.random.uniform(low=0.0, high=0.15),
#     #     np.random.uniform(low=-0.15, high=0),
#     #     -np.random.uniform(low=-0.15, high=0.15),
#     #     np.random.uniform(low=-0.15, high=0.15),
#     #     np.random.uniform(low=-0.2, high=0.2),
#     #     np.random.uniform(low=-0.2, high=0.2),
#     #     np.random.uniform(low=-0.05, high=0.05),
#     #     np.random.uniform(low=-0.05, high=0.05),
#     # ]
#     # joint_position = np.array(joint_position) + rest_position
#     # set_joint_positions(robot_id, arm_joints, joint_position.tolist())
#     for i in range(len(all_actions)):
#         # my_robot1.apply_action([0.1, 0.01])
#         # sampl = np.random.uniform(low=-1, high=1, size=(14,)).tolist()
#         # my_robot.apply_action(sampl)
#         my_robot.apply_action(all_actions[i])
#         # print(my_robot.get_rpy())
#         s.step()
# with Profiler('Simulator step'):
#     s.step()
#     rgb = s.renderer.render_robot_cameras(modes=('rgb'))[0]
#     rgbs.append(rgb)

# print(rgbs)
# print("Type of image", type(rgbs[0]))
