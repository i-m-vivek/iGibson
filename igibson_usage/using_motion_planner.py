from re import sub
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.external.pybullet_tools.utils import NullSpace
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
import gibson2
import argparse
import numpy as np
import os


def run_example(args):
    nav_env = iGibsonEnv(
        # mode="headless",
        config_file=args.config,
        mode=args.mode,
        action_timestep=1.0 / 120.0,
        physics_timestep=1.0 / 120.0,
    )

    motion_planner = MotionPlanningWrapper(nav_env, fine_motion_plan=True)
    state = nav_env.reset()
    # print("Robot Position", nav_env.robots[0].get_position())
    # # <<<<< Base Motion
    # path = motion_planner.plan_base_motion(np.array([-2, -2, 0.1]))
    # print(path)
    # # if the path is not found it will return None otherwise it will return the a list of tuple [(x,y,o), (x,y,o), (x,y,o), (x,y,o), ... ]
    # if path:
    #     print("FOUND THE PATH")
    #     print(path[-1])
    #     motion_planner.dry_run_base_plan(path)
    # # >>>>>

    # <<<<< Arm Motion
    # get the joint positions using the subgoal using IK
    while True:
        joint_pos = None
        while joint_pos is None:
            print("Base Position: ", nav_env.robots[0].get_position())
            print("EE Position: ", nav_env.robots[0].get_end_effector_position())
            subgoal = list(map(float, (input("Enter arm subgoal location: ").split())))
            print(subgoal)
            # subgoal = nav_env.robots[0].get_end_effector_position()
            # subgoal[2] = 0.48
            joint_pos = motion_planner.get_arm_joint_positions(subgoal)
        # check whether the robot can obtain those joint_pos using motion planner
        print(joint_pos)
        if joint_pos is not None:
            print("Found joint pos")
            arm_path = motion_planner.plan_arm_motion(joint_pos)
            if arm_path:
                print("Found the arm path")
                motion_planner.dry_run_arm_plan(arm_path)
            else:
                print("can't find an arm path :(")
        # else:

        #  >>>>>

        for i in range(20):
            action = np.zeros(nav_env.action_space.shape)
            state, reward, done, _ = nav_env.step(action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default=os.path.join("configs/tiago_motion_planning.yaml"),
        help="which config file to use",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "gui", "iggui"],
        default="gui",
        help="which mode for simulation (default: iggui)",
    )

    args = parser.parse_args()
    run_example(args)
