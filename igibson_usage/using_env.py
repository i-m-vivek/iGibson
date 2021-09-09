from gibson2.envs.igibson_env import iGibsonEnv
from time import time
import gibson2
import os
from gibson2.render.profiler import Profiler
import logging
import pickle
import numpy as np 

# config_filename = os.path.join(gibson2.example_config_path, 'tiago_point_nav_stadium.yaml')
config_filename = "igibson_usage/configs/exp_config.yaml"
env = iGibsonEnv(config_file=config_filename, mode="gui")
state_hist = []
reward_hist = []
done_hist = []
info_hist = []
env.reset()
# print("\n\n\n\n\n STARTING THINGS UP \n\n\n\n\n ")
for i in range(100):
    with Profiler("Environment action step"):
        action = env.action_space.sample()
        # action = np.array(action)
        # action_mask = np.zeros(14)
        # action_mask[:2] = 1
        # action = action*action_mask
        # action = action.tolist()
        state, reward, done, info = env.step(action)
        # print(reward)
        print(state)
        state_hist.append(state)
        reward_hist.append(reward)
        done_hist.append(done)
        info_hist.append(info)
        # print(reward, info)
        if done:
            logging.info("Episode finished after {} timesteps".format(i + 1))
            break

save_dict = {}
save_dict["state_hist"] = state_hist
# save_dict["reward_hist"] = reward_hist
# save_dict["done_hist"] = done_hist
# save_dict["info_hist"] = info_hist

name = "sample_states"

with open(f"igibson_usage/pkl_files/{name}.pkl", "wb") as f:
    pickle.dump(save_dict, f)
# print(type(state))
# print(state.keys())
env.close()
