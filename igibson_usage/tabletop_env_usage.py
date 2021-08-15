import pybullet as p
from gibson2.envs.igibson_env import iGibsonEnv
from time import time
import gibson2
import os
from gibson2.render.profiler import Profiler
import logging
import pickle as pkl

config_filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "tiago_tabletop.yaml"
)
env = iGibsonEnv(config_file=config_filename, mode="gui")
state_hist = []
reward_hist = []
done_hist = []
info_hist = []
env.reset()
for i in range(10000):
    # with Profiler("Environment action step"):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print(state)
    # print(reward)
    state_hist.append(state)
    reward_hist.append(reward)
    done_hist.append(done)
    info_hist.append(info)
    if done:
        logging.info("Episode finished after {} timesteps".format(i + 1))
        break
# data_dict= {
    # "state": state_hist, 
    # "reward": reward_hist, 
    # "done_hist": done_hist,
    # "info_hist": info_hist
# }
# 
# with open("pkl_files/tabletop_task.pkl", "wb") as f:
    # pkl.dump(data_dict, f)
env.close()
