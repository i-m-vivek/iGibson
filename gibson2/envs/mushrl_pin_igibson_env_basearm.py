# # from gibson2.envs.igibson_env import iGibsonEnv
# from gibson2.envs.igibson_env_v2 import iGibsonEnv
# from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
# import numpy as np
# import pybullet as p
# from mushroom_rl.core import Environment, MDPInfo
# from mushroom_rl.utils.spaces import Box, Discrete
# from mushroom_rl.core import Core
# import math
# from gibson2.utils.utils import rotate_vector_3d
# from gibson2.external.pybullet_tools.utils import set_joint_positions
# from gibson2.external.pybullet_tools.utils import joints_from_names
# import copy


# dt = 0.01  # Define time step


# q_limits = [[0.0, 2.74889357189],
#             [-1.57079632679, 1.0908307825],
#             [-3.53429173529, 1.57079632679],
#             [-0.392699081699, 2.35619449019],
#             [-2.09439510239, 2.09439510239],
#             [-1.57079632679, 1.57079632679],
#             [-2.09439510239, 2.09439510239]]

# q_limit_min = [0.,
#                -1.57079632679,
#                -3.53429173529,
#                -0.392699081699,
#                -2.09439510239,
#                -1.57079632679,
#                -2.09439510239]
# q_limit_max = [2.74889357189,
#                1.0908307825,
#                1.57079632679,
#                2.35619449019,
#                2.09439510239,
#                1.57079632679,
#                2.09439510239]

# def getPosVelJoints(robotId, joint_indexes):  # Function to get the position/velocity of all joints from pybullet

#     jointStates = p.getJointStates(robotId,
#                                    joint_indexes)  # State of all joints (position, velocity, reaction forces, appliedJointMotortoruqe)
#     # print('np.shape(jointStates):', np.shape(jointStates))
#     # print('len(jointStates): ', len(jointStates))
#     # print('jointStates[0]: ', jointStates[0])
#     # print('jointStates[0][0]: ', jointStates[0][0])
#     # baseState = p.getBasePositionAndOrientation(robotId)  # Position and orientation of the free flying base (Position, orientation)
#     # baseVel = p.getBaseVelocity(robotId)  # Velocity of the free flying base  (linear velocity, angular velocity)

#     # Reshaping data into q and qdot
#     joint_pos = np.vstack((np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).transpose()))
#     # q = np.vstack((np.array([baseState[0]]).transpose(), np.array([baseState[1]]).transpose(),
#     #                np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).transpose()))
#     # print('q: ', q) # ([:3] -> base position,
#     #                 # [3:7] -> base orientation in Quatenion,
#     #                 # [7:9] -> wheel right and left,
#     #                 # [9:16] -> position of 7 joints,
#     #                 # [16:] -> position of gripper right finger and left finger
#     # print('np.shape(q): ', np.shape(q)) # (18, 1), baseState[1] is orientation in Quatenion

#     return joint_pos


# def CosSin2theta(q: np.array, robot):  # TODO: 07.12 for base continuous joint

#     '''
#         param: q -> np.array(11, 1)
#         return: q_ref -> np.array(10, 1)
#     '''

#     q_ref = np.zeros((robot.nv, 1))  # (10, 1)
#     continuous_theta_cos = math.acos(q[2])  # Transform cos(theta) to theta
#     continuous_theta_sin = math.asin(q[3])

#     q_ref[2] = continuous_theta_cos
#     for i in range(2):  # Prismatic joint Y and X
#         q_ref[i] = q[i]
#     for j in range(3, 10):  # 7 arm revolute joints
#         q_ref[j] = q[j+1]

#     q_ref = q_ref.reshape(robot.nv, )
#     return q_ref

# def se3ToTransfrom(SE3):
#     # Transform a SE3 to a  (4, 4) transformation matrix.

#     r = numpy2torch(SE3.rotation)
#     t = numpy2torch(SE3.translation)
#     x1 = torch.cat((r, t.reshape(3, 1)), 1)
#     homo = torch.tensor([[0, 0, 0, 1]])
#     Tf = torch.cat((x1.float(), homo.float()), 0)

#     return Tf

# def theta2CosSin(q, robot):  # TODO: 07.12 for base continuous joint

#     '''
#         param: q -> np.array(10, 1)
#         return: q_ref -> np.array(11, 1)
#     '''

#     q_ref = np.zeros((robot.nq, 1))  # (11, 1), Define q which will be used in pinochio
#     # Transform theta to cos(theta) and sin(theta)
#     continuous_theta_cos = math.cos(q[2])  # Transform cos(theta) to theta
#     continuous_theta_sin = math.sin(q[2])

#     q_ref[2] = continuous_theta_cos
#     q_ref[3] = continuous_theta_sin
#     for i in range(2):  # prismatic Y and X joint
#         q_ref[i] = q[i]
#     for j in range(3, 10):  # 7 arm revolute joints
#         q_ref[j+1] = q[j]

#     q_ref = q_ref.reshape(robot.nq, )
#     return q_ref


# def get_quaternion_from_euler(roll, pitch, yaw):
#   """
#   Convert an Euler angle to a quaternion.
   
#   Input
#     :param roll: The roll (rotation around x-axis) angle in radians.
#     :param pitch: The pitch (rotation around y-axis) angle in radians.
#     :param yaw: The yaw (rotation around z-axis) angle in radians.
 
#   Output
#     :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
#   """
#   qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
#   qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
#   qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
#   qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
#   return np.array([qx, qy, qz, qw])



# class iGibsonMPEnv(Environment):
#     def __init__(
#         self,
#         config_file,
#         horizon=250,
#         gamma=0.99,
#         scene_id=None,
#         mode="headless",
#         action_timestep=1 / 10,
#         physics_timestep=1 / 240,
#         device_idx=0,
#         render_to_tensor=False,
#         automatic_reset=False,
#     ):
#         self.env = iGibsonEnv(
#             config_file,
#             scene_id=scene_id,
#             mode=mode,
#             action_timestep=action_timestep,
#             physics_timestep=physics_timestep,
#             device_idx=device_idx,
#             render_to_tensor=render_to_tensor,
#             automatic_reset=automatic_reset,
#         )
#         obs_shape = 0
#         for k in self.env.observation_space.spaces.keys():
#             if k != "occupancy_grid":
#                 obs_shape += np.prod(self.env.observation_space.spaces[k].shape)

#         observation_space = Box(-np.inf, np.inf, (obs_shape,))
#         # action_space = Box(-np.inf, np.inf, (7, )) # (x, y, orn, x_ee, y_ee, z_ee, embodiment)
#         action_space = Box(
#             np.array([-1.25, -1.25, -np.pi, -np.pi, 0.5, 0.1, 0, 0]),
#             np.array([1.25, 1.25, np.pi, np.pi, 1.0, 0.8, 1, 1]),
#             (8,),
#         )  # (x, y, orn, degree_ee, distance_ee, height_ee, embodiment), for the arm everything is in robot frame
#         mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
#         super().__init__(mdp_info)

#         raw_state = self.env.reset()
#         state_list = []
#         for k in raw_state.keys():
#             if k != "occupancy_grid" and k in self.env.observation_space.spaces.keys():
#                 state_list.append(
#                     raw_state[k].reshape(
#                         -1,
#                     )
#                 )
#         self._state = np.concatenate(state_list).astype(np.float32)

#         self.env_action_space = self.env.action_space

#         self.max_dist_subgoal = 2
#         # negative reward if the metapolicy gives bad subgoals for the motion planner
#         self.base_mp_reward = self.env.config.get("base_mp_reward", -1)
#         self.arm_mp_reward = self.env.config.get("arm_mp_reward", -1)

#         # self.tiago_arm_joint_ids = joints_from_names(
#         #         self.robot_id,
#         #         [
#         #             "torso_lift_joint",
#         #             "arm_1_joint",
#         #             "arm_2_joint",
#         #             "arm_3_joint",
#         #             "arm_4_joint",
#         #             "arm_5_joint",
#         #             "arm_6_joint",
#         #             "arm_7_joint",
#         #         ],
#         #     )

#     def reset(self, state=None):
#         if state is None:
#             raw_state = self.env.reset()
#             state_list = []
#             for k in raw_state.keys():
#                 if (
#                     k != "occupancy_grid"
#                     and k in self.env.observation_space.spaces.keys()
#                 ):
#                     state_list.append(
#                         raw_state[k].reshape(
#                             -1,
#                         )
#                     )
#             self._state = np.concatenate(state_list).astype(np.float32)
#         else:
#             self._state = state
#         return self._state

#     # def set_tiago_joint_pos(self, joint_pos): 
#     #     set_joint_positions(self.env.robots[0], self.tiago_arm_joint_ids, joint_pos)
        
#     def step(self, action):
#         """
#         (x, y, orn, x_ee, y_ee, z_ee, base, arm)
#         """
#         # print(action)
#         base_reward = 0
#         arm_reward = 0
#         # emb = action[-1] < 0.5
#         # emb = True  # over-riding to check whether there is a bug in the code or not.
#         if action[-2] == 1:
#             print("using base this time")
#             # added some offset to have a nice -1.25 to 1.25 model prediction.
#             # We only need need +ve x, y values.
#             action_c = copy.deepcopy(action)
#             action_c[0] += 1.25
#             action_c[1] += 1.25

#             robot_pos = self.env.robots[0].get_position()
#             _, _, yaw = self.env.robots[0].get_rpy()
#             local_action = np.array([action_c[0], action_c[1], robot_pos[2], 1])
#             rotation_mtrx = np.array(
#                 [[np.cos(yaw), -np.sin(yaw), 0, robot_pos[0]],
#                 [np.sin(yaw), np.cos(yaw), 0, robot_pos[1]],
#                 [0, 0, 1, robot_pos[2]],
#                 [0, 0, 0, 1]]
#             )
#             robot_to_world = np.matmul(rotation_mtrx, local_action)[:3]
#             robot_to_world[2] = yaw + action_c[2]  # change the z output to orn, and as orn is not in the goal it does not matter
#             path = self.motion_planner.plan_base_motion(robot_to_world)

#             if path is not None:
#                 self.motion_planner.dry_run_base_plan(path)
#             else:
#                 base_reward = self.base_mp_reward

#         else:
#             pos = self.env.robots[0].get_position()
#             _, _, yaw = self.env.robots[0].get_rpy()
#             orn = yaw + action[3]
#             target_ee = pos + np.array(
#                 [action[4] * math.cos(orn), action[4] * math.sin(orn), action[5]]
#             )
#             # target_ee = current_ee + action[3:6]
#             # target_ee[-1] = action[5]

#             joint_pos = self.motion_planner.get_arm_joint_positions(target_ee)
#             # TODO(Vivek): Set the joint pos only, no need to do the arm planning
#             if joint_pos is not None:
#                 arm_path = self.motion_planner.plan_arm_motion(joint_pos)
#                 if arm_path:
#                     self.motion_planner.dry_run_arm_plan(arm_path=arm_path)
#                 else:
#                     arm_reward = self.arm_mp_reward
#             else:
#                 arm_reward = self.arm_mp_reward

#         apply = np.zeros(self.env.action_space.shape[0])
#         raw_state, reward, done, info = self.env.step(apply)

#         state_list = []
#         for k in raw_state.keys():
#             if k != "occupancy_grid" and k in self.env.observation_space.spaces.keys():
#                 state_list.append(
#                     raw_state[k].reshape(
#                         -1,
#                     )
#                 )
#         self._state = np.concatenate(state_list).astype(np.float32)

#         reward += base_reward + arm_reward
#         return self._state, reward, done, info


# from gibson2.envs.igibson_env import iGibsonEnv
import gibson2
from gibson2.envs.igibson_env_v2 import iGibsonEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
import numpy as np
import pybullet as p
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box, Discrete
from mushroom_rl.core import Core
import math
from gibson2.utils.utils import rotate_vector_3d
from gibson2.external.pybullet_tools.utils import set_joint_positions
from gibson2.external.pybullet_tools.utils import joints_from_names
import copy
import torch 
from cep.utils import numpy2torch, torch2numpy
from cep.liegroups.torch import SO3, SE3
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
import os 

def CosSin2theta(q: np.array, robot):  # TODO: 07.12 for base continuous joint

    '''
        param: q -> np.array(11, 1)
        return: q_ref -> np.array(10, 1)
    '''

    q_ref = np.zeros((robot.nv, 1))  # (10, 1)
    continuous_theta_cos = math.acos(q[2])  # Transform cos(theta) to theta
    continuous_theta_sin = math.asin(q[3])

    q_ref[2] = continuous_theta_cos
    for i in range(2):  # Prismatic joint Y and X
        q_ref[i] = q[i]
    for j in range(3, 10):  # 7 arm revolute joints
        q_ref[j] = q[j+1]

    q_ref = q_ref.reshape(robot.nv, )
    return q_ref

def theta2CosSin(q, robot):  # TODO: 07.12 for base continuous joint

    '''
        param: q -> np.array(10, 1)
        return: q_ref -> np.array(11, 1)
    '''

    q_ref = np.zeros((robot.nq, 1))  # (11, 1), Define q which will be used in pinochio
    # Transform theta to cos(theta) and sin(theta)
    continuous_theta_cos = math.cos(q[2])  # Transform cos(theta) to theta
    continuous_theta_sin = math.sin(q[2])

    q_ref[2] = continuous_theta_cos
    q_ref[3] = continuous_theta_sin
    for i in range(2):  # prismatic Y and X joint
        q_ref[i] = q[i]
    for j in range(3, 10):  # 7 arm revolute joints
        q_ref[j+1] = q[j]

    q_ref = q_ref.reshape(robot.nq, )
    return q_ref

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
            np.array([-10, -10, 0.5, -np.pi, -np.pi/2, -np.pi]),
            np.array([10, 10, 1.8, np.pi, np.pi/2, np.pi]),
            (6,),
        )  # (x, y, orn, robo)
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
        self.max_dist_subgoal = 2
        
        self.load_tiago(os.path.join(gibson2.assets_path, "models", "tiago"), "tiago_single_mp.urdf")
        self.num_iter = 100

    def getPosVelJoints(self, robotId, joint_indexes):  # Function to get the position/velocity of all joints from pybullet

        jointStates = p.getJointStates(robotId,
                                    joint_indexes)  # State of all joints (position, velocity, reaction forces, appliedJointMotortoruqe)
        # print('np.shape(jointStates):', np.shape(jointStates))
        # print('len(jointStates): ', len(jointStates))
        # print('jointStates[0]: ', jointStates[0])
        # print('jointStates[0][0]: ', jointStates[0][0])
        # baseState = p.getBasePositionAndOrientation(robotId)  # Position and orientation of the free flying base (Position, orientation)
        # baseVel = p.getBaseVelocity(robotId)  # Velocity of the free flying base  (linear velocity, angular velocity)

        # Reshaping data into q and qdot
        joint_pos = np.vstack((np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).transpose()))
        # q = np.vstack((np.array([baseState[0]]).transpose(), np.array([baseState[1]]).transpose(),
        #                np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).transpose()))
        # print('q: ', q) # ([:3] -> base position,
        #                 # [3:7] -> base orientation in Quatenion,
        #                 # [7:9] -> wheel right and left,
        #                 # [9:16] -> position of 7 joints,
        #                 # [16:] -> position of gripper right finger and left finger
        # print('np.shape(q): ', np.shape(q)) # (18, 1), baseState[1] is orientation in Quatenion

        return joint_pos

    def calculate_vew(self, state, R):  # TODO: Velocity control
        '''
        params:
            state: Tensor -> contains end-effector rotation and position s[0], spatial velocity s[1]
            R: Tensor (4, 4) -> Homogenous transformation matrix of end-effector
        return:
            vtl: Tensor (1, 6). dx, contains (w, v), then dq = J_pinv * dx, and q = q + dq * dt
        '''

        x = state[0]  # Tensor(4, 4), end-effector rotation and position SE(3)
        # v = state[1]  # Tensor (1, 6), end-effector spatial velocity V_b

        R_inv = torch.inverse(R)
        Htl = torch.matmul(R_inv, x)  # R_inv * X
        Xe = SE3.from_matrix(Htl, normalize=True)  # <cep.liegroups.torch.se3.SE3Matrix>, SE(3)
        xtl = Xe.log()  # Tensor(1, 6), (omega, V)
        vtl = -xtl

        A = SE3.from_matrix(R)
        Adj_lw = A.adjoint()
        ve_w = torch.matmul(Adj_lw, vtl)

        return ve_w
    

    def calculate_mu(self, state, R):  # TODO: Acceleration control
        '''
        params:
            state: Tensor -> contains end-effector rotation and position s[0], spatial velocity s[1]
            R: Tensor (4, 4) -> Homogenous transformation matrix of end-effector
        return:
            mu: Tensor (1, 6). ddx, contains (dw, dv),
                then ddq = J_pinv * ddx, and dq = dq + ddq * dt
                                            q = q + dq * dt
        '''

        x = state[0]  # Tensor(4, 4), end-effector rotation and position SE(3)
        v = state[1]  # Tensor (1, 6), end-effector spatial velocity V_b
        # index = [3, 4, 5, 0, 1, 2]
        # v = v[index]

        R_inv = torch.inverse(R)
        Htl = torch.matmul(R_inv, x)  # R_inv * X
        Xe = SE3.from_matrix(Htl, normalize=True)  # <cep.liegroups.torch.se3.SE3Matrix>, SE(3)
        xtl = Xe.log()  # Tensor(1, 6), (omega, V)
        vtl = -xtl

        A = SE3.from_matrix(R)
        Adj_lw = A.adjoint()
        ve_w = torch.matmul(Adj_lw, vtl)

        # TODO: Acceleration control
        scale = 20
        mu = scale * ve_w - 1.2 * scale * v

        return mu

    def load_tiago(self, root_path, robot_name):
        self.robot = RobotWrapper.BuildFromURDF(os.path.join(root_path, robot_name), [root_path])
        self.joint_indexes = [0, 1, 2, 34, 35, 36, 37, 38, 39, 40]
        self.robotId = self.env.robots[0].robot_ids[0]
        self.EE_idx = self.robot.model.getFrameId('arm_7_link')

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.
        
        Input
            :param roll: The roll (rotation around x-axis) angle in radians.
            :param pitch: The pitch (rotation around y-axis) angle in radians.
            :param yaw: The yaw (rotation around z-axis) angle in radians.
        
        Output
            :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return np.array([qx, qy, qz, qw])

    def se3ToTransfrom(self, SE3):
    # Transform a SE3 to a  (4, 4) transformation matrix.
        r = numpy2torch(SE3.rotation)
        t = numpy2torch(SE3.translation)
        x1 = torch.cat((r, t.reshape(3, 1)), 1)
        homo = torch.tensor([[0, 0, 0, 1]])
        Tf = torch.cat((x1.float(), homo.float()), 0)
    
        return Tf

    def do_mp(self, pos_desired, rpy_desired):
        # import pdb; pdb.set_trace()
        # qua = self.get_quaternion_from_euler(rpy_desired[0], rpy_desired[1], rpy_desired[2])
        # qua = pin.Quaternion(qua[0], qua[1], qua[2], qua[3])
        qua = pin.Quaternion(0., 0., 0., 1.)
        # qua = pin.Quaternion(0.4619398, 0.1913417, 0.4619398, 0.7325378)

        x_desired = pin.SE3(qua, pos_desired)  # TODO: set desired position and orientation
        # x_desired.rotation = np.eye(3)
        q = pin.neutral(self.robot.model)
        dq = np.zeros(self.robot.nv)

        # q_home = [-0.1, -0.1, 0., 0., 0., 0., 0., 0., 0., 0.]
        # for i, q_i in enumerate(q_home):
        #     q_ii = q_i + np.random.rand(1) * 0.3  # Reset initial joint positions
        #     p.resetJointState(self.robotId, self.joint_indexes[i], q_ii)


        R = self.se3ToTransfrom(x_desired)

        for ii in range(self.num_iter):
            # check_joint_states = self.getPosVelJoints(self.robotId, self.joint_indexes)
            pin.computeJointJacobians(self.robot.model, self.robot.data, q)
            pin.forwardKinematics(self.robot.model, self.robot.data, q)
            pin.framesForwardKinematics(self.robot.model, self.robot.data, q)
            cur_x = self.robot.data.oMf[self.EE_idx]  # current position and orientation of EE
            # joint_states = self.getPosVelJoints(self.robotId, self.joint_indexes)


            cur_dx_vec = pin.getFrameVelocity(self.robot.model, self.robot.data, self.EE_idx, pin.ReferenceFrame.WORLD).vector
            cur_x = self.se3ToTransfrom(cur_x)  # To 4x4 Transformation matrix
            cur_dx = numpy2torch(cur_dx_vec)
            state = [cur_x, cur_dx]

            v_ew = self.calculate_vew(state=state, R=R)  # TODO: Velocity control
            mu = self.calculate_mu(state=state, R=R)  # TODO: Acceleration control
            
            pin.computeJointJacobians(self.robot.model, self.robot.data, q)
            pin.updateFramePlacements(self.robot.model, self.robot.data)


            J = pin.getFrameJacobian(self.robot.model, self.robot.data, self.EE_idx, pin.ReferenceFrame.WORLD)
            J = numpy2torch(J)

                        # TODO: Velocity control
            damp = 1e-6  # 0.1, 0.01, 0.001
            Idt = numpy2torch(np.eye(6))
            JJ_vel = torch.matmul(J.T,
                                  torch.inverse(torch.matmul(J, J.T) + (damp ** 2 * Idt)))  # TODO: Add damped pseudoinverse
            dq = torch.matmul(JJ_vel, v_ew)
            # dq = torch.matmul(torch.pinverse(J), v_ew)
            dq = torch2numpy(dq)
            # # Euler discretization
            # dt = .01
            # q = q + dq * dt
            # q = pin.integrate(robot.model, q, dq * dt)

            # TODO: Acceleration control
            JJ_acc = torch.matmul(J.T,
                                  torch.inverse(torch.matmul(J, J.T) + (damp ** 2 * Idt)))  # TODO: Add damped pseudoinverse
            ddq = torch.matmul(JJ_acc, mu)
            ddq = torch2numpy(ddq)

            # TODO: Euler discretization
            q = CosSin2theta(q, self.robot)  # TODO: Transform (cos, sin) to theta, 07.12

            dt = 0.01
            dq = dq + ddq * dt
            q = q + dq * dt

            for jj in range(len(self.joint_indexes)):  # TODO: PYBULLET set joint positions
                p.resetJointState(self.robotId, self.joint_indexes[jj], q[jj])

            q = theta2CosSin(q, self.robot)

            # joint_pos = self.getPosVelJoints(self.robotId, self.joint_indexes)
            p.stepSimulation()

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
        (x_ee, y_ee, z_ee, r, p, y)
        """
        # import pdb; pdb.set_trace()
        # action_c = copy.deepcopy(action[:3])
        # robot_pos = self.env.robots[0].get_position()
        # _, _, yaw = self.env.robots[0].get_rpy()
        # print(robot_pos, yaw)
        # local_action = np.array([action_c[0], action_c[1], robot_pos[2], 1])
        # rotation_mtrx = np.array(
            # [[np.cos(yaw), -np.sin(yaw), 0, robot_pos[0]],
            # [np.sin(yaw), np.cos(yaw), 0, robot_pos[1]],
            # [0, 0, 1, robot_pos[2]],
            # [0, 0, 0, 1]]
        # )
        # robot_to_world = np.matmul(rotation_mtrx, local_action)[:3]
        # robot_to_world[2] = action[2]  # change the z output to orn, and as orn is not in the goal it does not matter

        self.do_mp(action[:3], action[3:])
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

        return self._state, reward, done, info