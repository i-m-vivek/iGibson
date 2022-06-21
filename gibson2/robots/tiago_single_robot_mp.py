import gym
import numpy as np
import pybullet as p

from gibson2.external.pybullet_tools.utils import (
    joints_from_names,
    set_joint_positions,
    get_movable_joints,
)
from gibson2.robots.robot_locomotor import LocomotorRobot


class Tiago_SingleMP(LocomotorRobot):
    def __init__(self, config):
        # <<< <<<
        self.wheel_dim = 2
        self.arm_dim = 7
        self.rest_position = [
            -0.1,
            -0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        self.problem_parts = []  # filled on load
        self.joint_mask = []  # filled on load

        action_dim = 10
        LocomotorRobot.__init__(
            self,
            "tiago/tiago_single_mp.urdf",
            action_dim=action_dim,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control="position",
            self_collision=True,
        )

    def set_up_continuous_action_space(self):
        self.action_high = np.array(
            [
                25,
                25,
                np.pi,
                2.74889357189,
                1.0908307825,
                1.57079632679,
                2.35619449019,
                2.09439510239,
                1.57079632679,
                2.09439510239,
            ]
        )
        self.action_low = np.array(
            [
                -25,
                -25,
                -np.pi,
                0.0,
                -1.57079632679,
                -3.53429173529,
                -0.392699081699,
                -2.09439510239,
                -1.57079632679,
                -2.09439510239,
            ]
        )
        self.action_space = gym.spaces.Box(
            shape=(self.action_dim,), low=-1.0, high=1.0, dtype=np.float32
        )

    def set_up_discrete_action_space(self):
        assert False, "Tiago_Single does not support discrete actions"

    def robot_specific_reset(self):
        super(Tiago_SingleMP, self).robot_specific_reset()

        # roll the arm to its body
        robot_id = self.robot_ids[0]
        arm_joints = joints_from_names(
            robot_id,
            [
                "X",
                "Y",
                "R",
                "arm_1_joint",
                "arm_2_joint",
                "arm_3_joint",
                "arm_4_joint",
                "arm_5_joint",
                "arm_6_joint",
                "arm_7_joint",
            ],
        )

        rest_position = [
            -0.1,
            -0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        set_joint_positions(robot_id, arm_joints, rest_position)

    def get_end_effector_position(self):
        return self.parts["gripper_link"].get_position()

    def get_end_effector_index(self):
        return self.parts["gripper_link"].body_part_index

    def end_effector_part_index(self):
        """
        Get end-effector link id
        """
        return self.parts["gripper_link"].body_part_index

    def load(self):
        ids = super(Tiago_SingleMP, self).load()
        robot_id = self.robot_ids[0]

        # keep_collisions = [
        #     ["base_link", "arm_3_link"],
        #     ["base_link", "arm_4_link"],
        #     ["base_link", "arm_5_link"],
        #     ["base_link", "arm_6_link"],
        #     ["base_link", "arm_7_link"],
        #     ["base_link", "gripper_link"],
        #     ["base_link", "gripper_right_finger_link"],
        #     ["base_link", "gripper_left_finger_link"],
        #     ["base_link", "gripper_tool_link"],

        # ]
        # torso_lift_link
        # torso_fixed_link
        # torso_fixed_column_link

        # TO FIX these collisions, followings are the imp files
        # * env_example
        # * igibson_env -> def check_collision -> the collion link index provided by bullet are 1 less than that of our robot.parts
        # * using_sim
        keep_collisions = [
            ["base", "arm"],
            ["base", "wrist"],
            ["base", "gripper"],
            ["head", "arm"],
            ["head", "wrist"],
            ["head", "gripper"],
            ["wrist", "gripper"],
            ["torso_fixed_link", "arm_1_link"],
            ["torso_lift_link", "arm_1_link"],
            # # ["torso_fixed_column_link", "arm_1_link"], # this causes the movement of base as well
            # # ["torso_fixed_link", "arm_2_link"], # also causes some weird behaviour but the robot remains stable after loading
            ["torso_lift_link", "arm_2_link"],
            ["torso_fixed_column_link", "arm_2_link"],
            # # ["torso_fixed_link", "arm_3_link"], # this arm_3_link & torso_fixed_link causes collisions which leads to robot falling over
            ["torso_lift_link", "arm_3_link"],
            ["torso_fixed_column_link", "arm_3_link"],
            # ["torso_fixed_link", "arm_4_link"], # also causes some weird behaviour but the robot remains stable after loading
            ["torso_lift_link", "arm_4_link"],
            ["torso_fixed_column_link", "arm_4_link"],
            [
                "torso_fixed_link",
                "arm_5_link",
            ],  # also causes some weird behaviour but the robot remains stable after loading
            ["torso_lift_link", "arm_5_link"],
            ["torso_fixed_column_link", "arm_5_link"],
            ["torso", "arm_6_link"],
            ["torso", "arm_7_link"],
            ["torso", "wrist"],
            ["torso", "gripper"],
        ]
        for parta in self.parts:
            for partb in self.parts:
                flag = True
                for i, j in keep_collisions:
                    if ((i in parta) or (i in partb)) and (
                        ((j in parta) or (j in partb))
                    ):
                        flag = False
                        # print("YES")
                # if parta!=partb:
                if flag:
                    p.setCollisionFilterPair(
                        robot_id,
                        robot_id,
                        self.parts[parta].body_part_index,
                        self.parts[partb].body_part_index,
                        0,
                    )

        self.all_joints = get_movable_joints(robot_id)
        valid_joints = [j.joint_index for j in self.ordered_joints]
        self.joint_mask = [j in valid_joints for j in self.all_joints]
        return ids

    # def load(self):
    #     ids = super(Tiago_Single, self).load()
    #     robot_id = self.robot_ids[0]

    #     # get problematic links

    #     # moving_parts = ["arm", "wrist", "hand"]
    #     # problem_part_name = []
    #     # for part in self.parts:
    #     #     # print(part)
    #     #     for x in moving_parts:
    #     #         # print(x)
    #     #         if x not in part:
    #     #             problem_part_name.append(part)
    #     #             self.problem_parts.append(self.parts[part])
    #     # print(problem_part_name)
    #     # disable self collision
    #     # print(self.problem_parts)

    #     # self.problem_part_new = []
    #     # for i in range(len(problem_part_name)):
    #     #     if "arm" not in problem_part_name[i]:
    #     #         self.problem_part_new.append(self.problem_parts[i])

    #     moving_parts = [
    #         "gripper",
    #         "head",
    #         "arm_3_link",
    #         "arm_4_link",
    #         "arm_5_link",
    #         "arm_6_link",
    #         "arm_7_link",
    #         # "arm_tool_link",
    #     ]
    #     # print(moving_parts)
    #     problem_part_name = []
    #     # problem_parts = []
    #     for part in self.parts:
    #         flag = False
    #         for x in moving_parts:
    #             if x in part:
    #                 flag = True
    #                 break
    #         if flag == False:
    #             problem_part_name.append(part)
    #             self.problem_parts.append(self.parts[part])
    #     # print(problem_part_name)
    #     # print(self.problem_parts)

    #     for a in self.problem_parts:
    #         for b in self.problem_parts:
    #             # if a != b:
    #             p.setCollisionFilterPair(
    #                 robot_id, robot_id, a.body_part_index, b.body_part_index, 0
    #             )

    #     # calculate joint mask
    #     all_joints = get_movable_joints(robot_id)
    #     valid_joints = [j.joint_index for j in self.ordered_joints]
    #     self.joint_mask = [j in valid_joints for j in all_joints]

    #     return ids
