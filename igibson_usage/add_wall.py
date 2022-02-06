import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
planeId = p.loadURDF("plane.urdf")

startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("wall.urdf", startPos, startOrientation)


for i in range(10000):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
