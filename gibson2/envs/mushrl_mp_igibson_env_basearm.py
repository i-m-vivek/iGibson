from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
import numpy as np
import pybullet as p
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box, Discrete
from mushroom_rl.core import Core
import math
from gibson2.utils.utils import rotate_vector_3d


class iGibsonMPEnv(Environment):
    def __init__(
        self,
        config_file,
        horizon=250,
        gamma=0.99,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 10,
        physics_timestep=1 / 240,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
    ):
        self.env = iGibsonEnv(
            config_file,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor,
            automatic_reset=automatic_reset,
        )
        obs_shape = 0
        for k in self.env.observation_space.spaces.keys():
            if k != "occupancy_grid":
                obs_shape += np.prod(self.env.observation_space.spaces[k].shape)

        observation_space = Box(-np.inf, np.inf, (obs_shape,))
        # action_space = Box(-np.inf, np.inf, (7, )) # (x, y, orn, x_ee, y_ee, z_ee, embodiment)
        action_space = Box(
            np.array([-1.25, -1.25, -np.pi, -np.pi, 0.5, 0.1, 0]),
            np.array([1.25, 1.25, np.pi, np.pi, 1.0, 0.8, 1]),
            (7,),
        )  # (x, y, orn, degree_ee, distance_ee, height_ee, embodiment), for the arm everything is in robot frame
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        super().__init__(mdp_info)

        raw_state = self.env.reset()
        state_list = []
        for k in raw_state.keys():
            if k != "occupancy_grid" and k in self.env.observation_space.spaces.keys():
                state_list.append(
                    raw_state[k].reshape(
                        -1,
                    )
                )
        self._state = np.concatenate(state_list).astype(np.float32)

        self.env_action_space = self.env.action_space
        self.motion_planner = MotionPlanningWrapper(self.env)
        self.max_dist_subgoal = 2
        # negative reward if the metapolicy gives bad subgoals for the motion planner
        self.base_mp_reward = self.env.config.get("base_mp_reward", -1)
        self.arm_mp_reward = self.env.config.get("arm_mp_reward", -1)

    def reset(self, state=None):
        if state is None:
            raw_state = self.env.reset()
            state_list = []
            for k in raw_state.keys():
                if (
                    k != "occupancy_grid"
                    and k in self.env.observation_space.spaces.keys()
                ):
                    state_list.append(
                        raw_state[k].reshape(
                            -1,
                        )
                    )
            self._state = np.concatenate(state_list).astype(np.float32)
        else:
            self._state = state
        return self._state

    def step(self, action):
        """
        3 dim action
        (x, y, orn, x_ee, y_ee, z_ee, emb)
        """
        base_reward = 0
        arm_reward = 0
        emb = action[-1] < 0.5
        # emb = True  # over-riding to check whether there is a bug in the code or not.
        if emb == True:
            # added some offset to have a nice -1.25 to 1.25 model prediction.
            # We only need need +ve x, y values.
            action[0] += 1.25
            action[1] += 1.25

            robot_pos = self.env.robots[0].get_position()
            _, _, yaw = self.env.robots[0].get_rpy()
            local_action = np.array([action[0], action[1], robot_pos[2], 1])
            rotation_mtrx = np.array(
                [[np.cos(yaw), -np.sin(yaw), 0, robot_pos[0]],
                [np.sin(yaw), np.cos(yaw), 0, robot_pos[1]],
                [0, 0, 1, robot_pos[2]],
                [0, 0, 0, 1]]
            )
            robot_to_world = np.matmul(rotation_mtrx, local_action)[:3]
            robot_to_world[2] = action[
                2
            ]  # change the z output to orn, and as orn is not in the goal it does not matter
            path = self.motion_planner.plan_base_motion(robot_to_world)

            if path is not None:
                self.motion_planner.dry_run_base_plan(path)
            else:
                base_reward = self.base_mp_reward

        else:
            pos = self.env.robots[0].get_position()
            _, _, yaw = self.env.robots[0].get_rpy()
            orn = yaw + action[3]
            target_ee = pos + np.array(
                [action[4] * math.cos(orn), action[4] * math.sin(orn), action[5]]
            )
            # target_ee = current_ee + action[3:6]
            # target_ee[-1] = action[5]

            joint_pos = self.motion_planner.get_arm_joint_positions(target_ee)
            if joint_pos is not None:
                arm_path = self.motion_planner.plan_arm_motion(joint_pos)
                if arm_path:
                    self.motion_planner.dry_run_arm_plan(arm_path=arm_path)
                else:
                    arm_reward = self.arm_mp_reward
            else:
                arm_reward = self.arm_mp_reward

        apply = np.zeros(self.env.action_space.shape[0])
        raw_state, reward, done, info = self.env.step(apply)

        state_list = []
        for k in raw_state.keys():
            if k != "occupancy_grid" and k in self.env.observation_space.spaces.keys():
                state_list.append(
                    raw_state[k].reshape(
                        -1,
                    )
                )
        self._state = np.concatenate(state_list).astype(np.float32)

        reward += base_reward + arm_reward
        return self._state, reward, done, info

    # With both arm and base motion planner
    # def step(self, action):
    #     """
    #     7 dim action
    #     (x, y, orn, x_ee, y_ee, z_ee, embodiment)
    #     """

    #     base_reward = 0
    #     arm_reward = 0
    #     emb = action[-1] > 0

    #     if emb == 0:
    #         path = self.motion_planner.plan_base_motion(action[:3])
    #         if path is not None:
    #             self.motion_planner.dry_run_base_plan(path)
    #         else:
    #             base_reward = self.base_mp_reward
    #     else:
    #         joint_pos = self.motion_planner.get_arm_joint_positions(action[3:6])
    #         if joint_pos is not None:
    #             arm_path = self.motion_planner.plan_arm_motion(joint_pos)
    #             if arm_path:
    #                 self.motion_planner.dry_run_arm_plan(arm_path)
    #             else:
    #                 arm_reward = self.arm_mp_reward
    #         else:
    #             arm_reward = self.arm_mp_reward

    #     apply = np.zeros(self.env.action_space.shape[0])
    #     # print(apply)
    #     raw_state, reward, done, info = self.env.step(apply)

    #     state_list =[]
    #     for k in raw_state.keys():
    #         if k!= "occupancy_grid" and k in self.env.observation_space.spaces.keys():
    #             state_list.append(raw_state[k].reshape(-1, ))
    #     self._state = np.concatenate(state_list).astype(np.float32)

    #     reward += arm_reward + base_reward
    #     return self._state, reward, done, info
