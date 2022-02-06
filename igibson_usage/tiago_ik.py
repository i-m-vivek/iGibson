from gibson2.robots.fetch_robot import Fetch
# from gibson2.robots.tiago_single_robot_dnandha import Tiago_Single
from gibson2.robots.tiago_single_robot import Tiago_Single
from gibson2.simulator import Simulator
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.utils.utils import parse_config
from gibson2.render.profiler import Profiler

import pybullet as p
from gibson2.external.pybullet_tools.utils import get_movable_joints, set_joint_positions, joints_from_names, get_joint_positions, \
    get_max_limits, get_min_limits, get_sample_fn

import numpy as np
import gibson2
import os

def main():
    config = parse_config("configs/tiago_config.yaml")
    s = Simulator(mode='gui', physics_timestep=1 / 240.0)
    scene = EmptyScene()
    s.import_scene(scene)
    robot = Tiago_Single(config=config)
    s.import_robot(robot)

    # robot_id = fetch.robot_ids[0]
    robot_id = robot.robot_ids[0]

    # arm_joints = joints_from_names(
    #         robot_id,
    #         [
    #             "arm_1_joint",
    #             "arm_2_joint",
    #             "arm_3_joint",
    #             "arm_4_joint",
    #             "arm_5_joint",
    #             "arm_6_joint",
    #             "arm_7_joint",
    #         ],
    #     )

    robot.robot_body.reset_position([0, 0, 0])
    robot.robot_body.reset_orientation([0, 0, 1, 0])
    x, y, z = robot.get_end_effector_position()

    visual_marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.1)
    marker = p.createMultiBody(baseVisualShapeIndex=visual_marker)

    all_joints = get_movable_joints(robot_id)
    # max_limits = [0, 0] + get_max_limits(robot_id, arm_joints)
    # min_limits = [0, 0] + get_min_limits(robot_id, arm_joints)
    # rest_position = [0, 0] + list(get_joint_positions(robot_id, arm_joints))
    max_limits = get_max_limits(robot_id, all_joints)
    min_limits = get_min_limits(robot_id, all_joints)
    rest_position = list(get_joint_positions(robot_id, all_joints))
    joint_range = list(np.array(max_limits) - np.array(min_limits))
    # joint_range = [item + 1 for item in joint_range]
    jd = [0.1 for item in joint_range]
    # print([j.joint_name for j in robot.ordered_joints])
    # print(len(robot.ordered_joints))

    valid_joints = [j.joint_index for j in robot.ordered_joints]
    # print([j.joint_name for j in robot.ordered_joints[2:-2]])
    joint_mask = []
    # print(all_joints)
    # print("len all joints", len(all_joints))

    for j in all_joints:
        if j in valid_joints:
            joint_mask += [True]
        else:
            joint_mask += [False]
    print(joint_mask)
    
    def accurateCalculateInverseKinematics(robotid, endEffectorId, targetPos, threshold, maxIter):
        # sample_fn = get_sample_fn(robotid, arm_joints)
        # set_joint_positions(robotid, arm_joints, sample_fn())
        it = 0
        while it < maxIter:
            jointPoses = p.calculateInverseKinematics(
                robotid,
                endEffectorId,
                targetPos,
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=rest_position,
                jointDamping=jd)
            jointPoses = np.asarray(jointPoses)
            # set_joint_positions(robotid, arm_joints, jointPoses[2:10])
            set_joint_positions(robotid, valid_joints, jointPoses[joint_mask])
            ls = p.getLinkState(robotid, endEffectorId)
            newPos = ls[4]

            dist = np.linalg.norm(np.array(targetPos) - np.array(newPos))
            print(dist)
            if dist < threshold:
                print("done")
                break

            it += 1

        print("Num iter: " + str(it) + ", residual: " + str(dist))
        return jointPoses

    while True:
        with Profiler("Simulation step"):
            robot.robot_body.reset_position([0, 0, 0])
            robot.robot_body.reset_orientation([0, 0, 1, 0])
            threshold = 0.15
            maxIter = 100
            joint_pos = accurateCalculateInverseKinematics(
                robot_id,
                # robot.end_effector_part_index(),
                robot.parts['gripper_link'].body_part_index,
                [x, y, z],
                threshold,
                maxIter)[2:10]

            s.step()
            keys = p.getKeyboardEvents()
            for k, v in keys.items():
                if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN)):
                    x += 0.01
                if (k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN)):
                    x -= 0.01
                if (k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN)):
                    y += 0.01
                if (k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN)):
                    y -= 0.01
                if (k == ord('z') and (v & p.KEY_IS_DOWN)):
                    z += 0.01
                if (k == ord('x') and (v & p.KEY_IS_DOWN)):
                    z -= 0.01
            p.resetBasePositionAndOrientation(marker, [x, y, z], [0, 0, 0, 1])

    s.disconnect()


if __name__ == '__main__':
    main()