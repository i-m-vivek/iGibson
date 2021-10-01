import gym
import numpy as np
import pybullet as p

from gibson2.external.pybullet_tools.utils import (
    joints_from_names,
    set_joint_positions,
    get_movable_joints,
)
from gibson2.robots.robot_locomotor import LocomotorRobot


class Tiago_Single(LocomotorRobot):
    def __init__(self, config):
        # >>>> Because we are using vel control we need the maximum vel of different body parts >>>>
        self.wheel_velocity = config.get("wheel_velocity", 1.0)
        self.torso_lift_velocity = config.get("torso_lift_velocity", 1.0)
        self.head_velocity = config.get("head_velocity", 1.0)
        self.arm_velocity = config.get("arm_velocity", 1.0)
        self.gripper_velocity = config.get("gripper_velocity", 1.0)
        # <<< <<<
        self.wheel_dim = 2
        self.torso_lift_dim = 1
        self.head_dim = 2
        self.arm_dim = 7
        self.gripper_dim = 2
        self.rest_position = [
            0,
            0,
            0,
            0,
            0,
            0,
            np.pi,
            -np.pi / 2,
            0,
            np.pi / 2,
            0,
            0,
            0,
            0,
            0,
        ]

        self.problem_parts = []  # filled on load
        self.joint_mask = []  # filled on load

        action_dim = (
            self.wheel_dim
            + self.torso_lift_dim
            + self.head_dim
            + self.arm_dim
            + self.gripper_dim
        )
        LocomotorRobot.__init__(
            self,
            "tiago/tiago_single.urdf",
            action_dim=action_dim,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control="velocity",
            self_collision=True,
        )

    def set_up_continuous_action_space(self):
        self.action_high = np.array(
            [self.wheel_velocity] * self.wheel_dim
            + [self.torso_lift_velocity] * self.torso_lift_dim
            + [self.head_velocity] * self.head_dim
            + [self.arm_velocity] * self.arm_dim
            + [self.gripper_velocity] * self.gripper_dim
        )
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(
            shape=(self.action_dim,), low=-1.0, high=1.0, dtype=np.float32
        )

    def set_up_discrete_action_space(self):
        assert False, "Tiago_Single does not support discrete actions"

    def robot_specific_reset(self):
        super(Tiago_Single, self).robot_specific_reset()

        # roll the arm to its body
        robot_id = self.robot_ids[0]
        arm_joints = joints_from_names(
            robot_id,
            [
                "torso_lift_joint",
                "arm_1_joint",
                "arm_2_joint",
                "arm_3_joint",
                "arm_4_joint",
                "arm_5_joint",
                "arm_6_joint",
                "arm_7_joint",
            ],
        )

        rest_position = [0, np.pi, -np.pi / 2, 0, np.pi / 2, 0, 0, 0]

        set_joint_positions(robot_id, arm_joints, rest_position)

    def get_end_effector_position(self):
        return self.parts["gripper_grasping_frame"].get_position()

    def get_end_effector_index(self):
        return self.parts["gripper_grasping_frame"].body_part_index

    def load(self):
        ids = super(Tiago_Single, self).load()
        robot_id = self.robot_ids[0]

        # get problematic links

        # moving_parts = ["arm", "wrist", "hand"]
        # problem_part_name = []
        # for part in self.parts:
        #     # print(part)
        #     for x in moving_parts:
        #         # print(x)
        #         if x not in part:
        #             problem_part_name.append(part)
        #             self.problem_parts.append(self.parts[part])
        # print(problem_part_name)
        # disable self collision
        # print(self.problem_parts)

        # self.problem_part_new = []
        # for i in range(len(problem_part_name)):
        #     if "arm" not in problem_part_name[i]:
        #         self.problem_part_new.append(self.problem_parts[i])

        moving_parts = [
            "gripper",
            "head",
            "arm_3_link",
            "arm_4_link",
            "arm_5_link",
            "arm_6_link",
            "arm_7_link",
            "arm_tool_link",
        ]
        # print(moving_parts)
        problem_part_name = []
        # problem_parts = []
        for part in self.parts:
            flag = False
            for x in moving_parts:
                if x in part:
                    flag = True
                    break
            if flag == False:
                problem_part_name.append(part)
                self.problem_parts.append(self.parts[part])
        # print(problem_part_name)
        # print(self.problem_parts)

        for a in self.problem_parts:
            for b in self.problem_parts:
                # if a != b:
                p.setCollisionFilterPair(
                    robot_id, robot_id, a.body_part_index, b.body_part_index, 0
                )

        # calculate joint mask
        all_joints = get_movable_joints(robot_id)
        valid_joints = [j.joint_index for j in self.ordered_joints]
        self.joint_mask = [j in valid_joints for j in all_joints]

        return ids
