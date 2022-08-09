import numpy as np 
import torch 
import math 
import pybullet as p
from cep.utils import numpy2torch, torch2numpy
from cep.liegroups.torch import SO3, SE3


def calculate_mu(state, R):  # TODO: Acceleration control
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


def calculate_vew(state, R):  # TODO: Velocity control
    '''
    params:
        state: Tensor -> contains end-effector rotation and position s[0], spatial velocity s[1]
        R: Tensor (4, 4) -> Homogenous transformation matrix of end-effector
    return:
        vtl: Tensor (1, 6). dx, contains (w, v), then dq = J_pinv * dx, and q = q + dq * dt
    '''
    # import pdb; pdb.set_trace()
    x = state[0]  # Tensor(4, 4), end-effector rotation and position SE(3)
    # v = state[1]  # Tensor (1, 6), end-effector spatial velocity V_b

    R_inv = torch.inverse(R)
    Htl = torch.matmul(R_inv, x)  # R_inv * X
    Xe = SE3.from_matrix(Htl, normalize=True)  # <cep.liegroups.torch.se3.SE3Matrix>, SE(3)
    xtl = Xe.log()  # Tensor(1, 6), (omega, V) | Spatial Vector conneting the EE in the goal frame 
    vtl = -xtl # The direction changed

    A = SE3.from_matrix(R)
    Adj_lw = A.adjoint()
    ve_w = torch.matmul(Adj_lw, vtl)

    return ve_w

def se3ToTransfrom(SE3):
    # Transform a SE3 to a  (4, 4) transformation matrix.

    r = numpy2torch(SE3.rotation)
    t = numpy2torch(SE3.translation)
    x1 = torch.cat((r, t.reshape(3, 1)), 1)
    homo = torch.tensor([[0, 0, 0, 1]])
    Tf = torch.cat((x1.float(), homo.float()), 0)

    return Tf

def getPosVelJoints(robotId, joint_indexes):  # Function to get the position/velocity of all joints from pybullet

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

def CosSin2theta(q: np.array, robot):  # TODO: 07.12 for base continuous joint

    '''
        param: q -> np.array(11, 1)
        return: q_ref -> np.array(10, 1)
    '''
    # import pdb; pdb.set_trace()
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
    # import pdb; pdb.set_trace()
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
