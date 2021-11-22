from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
import numpy as np
import pybullet as p


class iGibsonMPEnv:
    def __init__(
        self,
        config_file,
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
        self.env.reset()
        self.observation_space = self.env.observation_space
        self.action_space = 4 # three for the position of the EE and one for the orientation of the robot. 
        self.env_action_space = self.env.action_space
        self.motion_planner = MotionPlanningWrapper(self.env)
        self.max_dist_subgoal = 2
    # def plan_base_motion(self, goal):
    def reset(self):
        return self.env.reset()

    def step(self, actions):
        """
        Motion planner based movement.
        Returns the next state, reward, done, info
        following OpenAI Gym's convention

        The actions are the location where you want the arm of the robot to be.
        The robot first tries to reach a nearby position using the base and then move its arm.
        If the robot

        :param action: goal location
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information

        https://github.com/StanfordVL/iGibson/blob/6800954d8e6aca0bfabfa3d4d649cf672e369e87/igibson/envs/behavior_mp_env.py#L443
        """

        # TODO: How to decide the base position, should
        # we search for nearest reachable position if the robot
        #  can not reach to the exact position
        

        # TODO: How to decide the new orientation? This is imp...
        # the orintation can be imporved by doing somthing similar to p.raytest
        # https://github.com/StanfordVL/iGibson/blob/6800954d8e6aca0bfabfa3d4d649cf672e369e87/igibson/envs/behavior_mp_env.py#L465

        found_path = False  # ((x,y,z),(roll, pitch, yaw))

        original_position = self.env.robots[0].get_position()
        original_orientation = self.env.robots[0].get_orientation()

        distance_to_try = [0.6, 1.2, 1.8, 2.4]
        for distance in distance_to_try:
            if found_path:
                break
            for _ in range(20):
                yaw = np.random.uniform(-np.pi, np.pi)
                pos = [
                    actions[0] + distance * np.cos(yaw),
                    actions[1] + distance * np.sin(yaw),
                    original_position[-1],
                ]
                orn = [0, 0, yaw - np.pi]
                path = self.motion_planner.plan_base_motion([pos[0], pos[1], yaw - np.pi])
                if path:
                    found_path = True
                    if self.env.mode == "headless":
                        self.env.robots[0].set_position_orientation(
                            pos, p.getQuaternionFromEuler(orn)
                        )
                    else:
                        self.motion_planner.dry_run_base_plan(path)
                    break
        if path:
            joint_pos = self.motion_planner.get_arm_joint_positions(actions)
            if joint_pos:
                arm_path = self.motion_planner.plan_arm_motion(joint_pos)
                if arm_path:
                    self.motion_planner.dry_run_arm_plan(arm_path)
        
        print(self.action_space)
        apply = np.zeros(self.action_space.shape[0])
        print(apply)
        state, reward, done, info = self.env.step(apply)
        # TODO: Play with the reward to incorporate the effect of no reachable motion planner goals.
        return state, reward, done, info

    def step2(self, actions):
        """
        Motion planner based movement.
        Returns the next state, reward, done, info
        following OpenAI Gym's convention

        The actions are the location where you want the arm of the robot to be.
        The robot first tries to reach a nearby position using the base and then move its arm.
        If the robot

        :param action: goal location
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information

        https://github.com/StanfordVL/iGibson/blob/6800954d8e6aca0bfabfa3d4d649cf672e369e87/igibson/envs/behavior_mp_env.py#L443
        """

        # TODO: How to decide the base position, should
        # we search for nearest reachable position if the robot
        #  can not reach to the exact position
        

        # TODO: How to decide the new orientation? This is imp...
        # the orintation can be imporved by doing somthing similar to p.raytest
        # https://github.com/StanfordVL/iGibson/blob/6800954d8e6aca0bfabfa3d4d649cf672e369e87/igibson/envs/behavior_mp_env.py#L465

        found_path = False  # ((x,y,z),(roll, pitch, yaw))
        original_position = self.env.robots[0].get_position()
        original_orientation = self.env.robots[0].get_orientation()

        action_pos = actions[:3]
        action_pos[0] = np.clip(action_pos[0], original_position[0] - self.max_dist_subgoal, original_position[0] + self.max_dist_subgoal)
        action_pos[1] = np.clip(action_pos[1], original_position[1] - self.max_dist_subgoal, original_position[1] + self.max_dist_subgoal)
        action_yaw = np.clip(actions[:-1], a_min = -np.pi, a_max = np.pi)

        distance_to_try = [0.6, 1.2, 1.8, 2.4]
        for distance in distance_to_try:
            if found_path:
                break
            for _ in range(20):
                r = np.random.uniform(-np.pi, np.pi)
                pos = [
                    action_pos[0] + distance * np.cos(r),
                    action_pos[1] + distance * np.sin(r),
                    original_position[-1],
                ]
                orn = [0, 0, action_yaw]
                path = self.motion_planner.plan_base_motion([pos[0], pos[1], action_yaw])
                if path:
                    found_path = True
                    if self.env.mode == "headless":
                        self.env.robots[0].set_position_orientation(
                            pos, p.getQuaternionFromEuler(orn)
                        )
                    else:
                        self.motion_planner.dry_run_base_plan(path)
                    break
        
        # desired_joint_pos = action_pos - self.env.robots[0].get_position()
        
        if path:
            joint_pos = self.motion_planner.get_arm_joint_positions(actions)
            if joint_pos:
                arm_path = self.motion_planner.plan_arm_motion(joint_pos)
                if arm_path:
                    self.motion_planner.dry_run_arm_plan(arm_path)
        
        
        apply = np.zeros(self.env_action_space.shape[0])
        state, reward, done, info = self.env.step(apply)
        # TODO: Play with the reward to incorporate the effect of no reachable motion planner goals.
        
        return state, reward, done, info

    # add more functions that are used by PPO agents
    # def action_space(self):
