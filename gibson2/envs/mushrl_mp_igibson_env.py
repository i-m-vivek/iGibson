from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
import numpy as np
import pybullet as p
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box, Discrete
from mushroom_rl.core import Core


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
            -np.inf, np.inf, (3,)
        )  # (x, y, orn, x_ee, y_ee, z_ee, embodiment)
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
        self._state = np.concatenate(state_list)

        self.env_action_space = self.env.action_space
        self.motion_planner = MotionPlanningWrapper(self.env)
        self.max_dist_subgoal = 2
        # negative reward if the metapolicy gives bad subgoals for the motion planner
        self.base_mp_reward = self.env.config.get("base_mp_reward", -1)
        self.arm_mp_reward = self.env.config.get("arm_mp_reward", -1)

    def reset(self, state=None):
        if state is None:
            self._state = self.env.reset()
        else:
            self._state = state
        return self._state

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
    #     self._state = np.concatenate(state_list)

    #     reward += arm_reward + base_reward
    #     return self._state, reward, done, info

    def step(self, action):
        """
        3 dim action
        (x, y, orn, x_ee, y_ee, z_ee, embodiment)
        """

        base_reward = 0

        path = self.motion_planner.plan_base_motion(action[:3])
        if path is not None:
            self.motion_planner.dry_run_base_plan(path)
        else:
            base_reward = self.base_mp_reward

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
        self._state = np.concatenate(state_list)

        reward += base_reward
        return self._state, reward, done, info
