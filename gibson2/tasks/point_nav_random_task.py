from gibson2.tasks.point_nav_fixed_task import PointNavFixedTask
from gibson2.utils.utils import l2_distance
import pybullet as p
import logging
import numpy as np


class PointNavRandomTask(PointNavFixedTask):
    """
    Point Nav Random Task
    The goal is to navigate to a random goal position
    """

    def __init__(self, env):
        super(PointNavRandomTask, self).__init__(env)
        self.move_arm_only = self.config.get("move_arm_only", False)
        self.target_dist_min = self.config.get("target_dist_min", 1.0)
        self.target_dist_max = self.config.get("target_dist_max", 10.0)

    def sample_initial_pose_and_target_pos(self, env):
        """
        Sample robot initial pose and target position

        :param env: environment instance
        :return: initial pose and target position
        """
        if "intial_robot_pos" in self.config:
            initial_pos = np.array(self.config["intial_robot_pos"])
        else:
            _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
        # target_orn = np.random.uniform(-np.pi, np.pi)
        if self.move_arm_only:
            yaw = np.random.uniform(0, 2 * np.pi)
            angle_offset = np.random.uniform(-np.pi / 7, -np.pi / 7)
            dist_offset = np.random.uniform(0.3, 0.75)
            x_offset = dist_offset * np.cos(yaw + angle_offset)
            y_offset = dist_offset * np.sin(yaw + angle_offset)

            initial_orn = [0, 0, yaw]
            target_pos = np.array(
                [
                    x_offset,
                    y_offset,
                    np.random.uniform(0.3, 0.9),
                ]
            )
            target_pos = initial_pos + target_pos
            # return initial_pos, initial_orn, target_pos, target_orn
            return initial_pos, initial_orn, target_pos
        else:
            max_trials = 100
            dist = 0.0
            for _ in range(max_trials):
                _, target_pos = env.scene.get_random_point(floor=self.floor_num)
                if env.scene.build_graph:
                    _, dist = env.scene.get_shortest_path(
                        self.floor_num,
                        initial_pos[:2],
                        target_pos[:2],
                        entire_path=False,
                    )
                else:
                    dist = l2_distance(initial_pos, target_pos)
                if self.target_dist_min < dist < self.target_dist_max:
                    break
            if not (self.target_dist_min < dist < self.target_dist_max):
                print("WARNING: Failed to sample initial and target positions")
            if "intial_robot_pos" in self.config:
                initial_orn = np.array(self.config["initial_orn"])
            else:
                initial_orn = np.array([0, 0, np.random.uniform(-np.pi, np.pi)])
            # return initial_pos, initial_orn, target_pos, target_orn
            return initial_pos, initial_orn, target_pos

    def reset_scene(self, env):
        """
        Task-specific scene reset: get a random floor number first

        :param env: environment instance
        """
        self.floor_num = env.scene.get_random_floor()
        super(PointNavRandomTask, self).reset_scene(env)

    def reset_agent(self, env):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.

        :param env: environment instance
        """
        reset_success = False
        max_trials = 100

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for i in range(max_trials):
            (
                initial_pos,
                initial_orn,
                target_pos,
                # target_orn,
            ) = self.sample_initial_pose_and_target_pos(env)
            reset_success = env.test_valid_position(
                env.robots[0], initial_pos, initial_orn
            ) and env.test_valid_position(env.robots[0], target_pos)
            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        p.removeState(state_id)

        self.target_pos = target_pos
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn
        # self.target_orn = target_orn

        super(PointNavRandomTask, self).reset_agent(env)

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, end effector position, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        # task_obs = super(PointNavRandomTask, self).get_task_obs(
        #     env
        # )  # vel_x, vel_y, angularvel_z

        # robot_pos = env.robots[0].get_position()
        # robot_rpy = env.robots[0].get_rpy()

        # proprioceptive_states = np.concatenate([robot_pos, robot_rpy])
        # task_obs = np.append(task_obs, proprioceptive_states)
        goal_pos = self.target_pos
        # task_obs = np.array([goal_pos])
        return goal_pos
