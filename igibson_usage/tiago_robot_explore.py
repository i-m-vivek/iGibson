import pybullet as p 
import os 
import numpy as np 
import pybullet_data
import time 

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setGravity(0,0,-10)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])

flags = p.URDF_USE_MATERIAL_COLORS_FROM_MTL
flags = flags | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT

tiago = p.loadURDF("/home/vkmittal14/WORKSPACE/TUD/tiago_urdfs/tiago_old.urdf",startPos, startOrientation, flags=flags)
# fetch = p.loadURDF("/home/vkmittal14/WORKSPACE/TUD/vk_igibson/iGibson/gibson2/data/assets/models/fetch/fetch.urdf", startPos, startOrientation)
for i in range(50000): 
    p.stepSimulation()
    # time.sleep(1/240)
p.disconnect()