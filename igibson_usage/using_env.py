from gibson2.envs.igibson_env import iGibsonEnv
from time import time
import gibson2
import os
from gibson2.render.profiler import Profiler
import logging
import pickle
import numpy as np 
import pybullet as p
# gibson2/examples/configs/turtlebot_interactive_nav.yaml
config_filename = os.path.join("new_configs", 'tiago_basearm_random_reaching_relmogen.yaml')
# config_filename = "gibson2/examples/configs/turtlebot_interactive_nav.yaml"
env = iGibsonEnv(config_file=config_filename, mode="gui", automatic_reset=True)
state_hist = []
reward_hist = []
done_hist = []
info_hist = []
# env.load()
env.reset()
# print("\n\n\n\n\n STARTING THINGS UP \n\n\n\n\n ")
# for i in range(1000):
i = 1
import ipdb; ipdb.set_trace()
while True:
    if i%50 == 0:
        env.reset()
    i+=1
    # with Profiler("Environment action step"):
    action = env.action_space.sample()
    action = np.array(action)
    action_mask = np.zeros(14)
    # action_mask[2:] = 1
    action = action*action_mask
    action = action.tolist()
    # print(env.scene.table)
    # env.check_collision(env.scene.table)
    state, reward, done, info = env.step(action)
        # print(p.getContactPoints(env.scene.table, env.robots[0].robot_ids[0]))
        # print(reward)
        # print(state["task_obs"])
        # state_hist.append(state)
        # reward_hist.append(reward)
        # done_hist.append(done)
        # info_hist.append(info)
        # print(reward, info)
        # if done:
        #     logging.info("Episode finished after {} timesteps".format(i + 1))
        #     break

# save_dict = {}
# save_dict["state_hist"] = state_hist
# save_dict["reward_hist"] = reward_hist
# save_dict["done_hist"] = done_hist
# save_dict["info_hist"] = info_hist

# name = "sample_states"

# with open(f"igibson_usage/pkl_files/{name}.pkl", "wb") as f:
#     pickle.dump(save_dict, f)
# print(type(state))
# print(state.keys())
env.close()
