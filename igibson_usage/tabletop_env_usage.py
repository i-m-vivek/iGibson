import pybullet as p
from gibson2.envs.igibson_env import iGibsonEnv
from time import time
import gibson2
import os
from gibson2.render.profiler import Profiler
import logging


config_filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "tiago_tabletop.yaml"
)
env = iGibsonEnv(config_file=config_filename, mode="gui")
state_hist = []
reward_hist = []
done_hist = []
info_hist = []
env.reset()
for i in range(1000):
    with Profiler("Environment action step"):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        state_hist.append(state)
        reward_hist.append(reward)
        done_hist.append(done)
        info_hist.append(info)
        if done:
            logging.info("Episode finished after {} timesteps".format(i + 1))
            break
env.close()
