from gibson2.envs.igibson_env import iGibsonEnv
from time import time
import gibson2
import os
from gibson2.render.profiler import Profiler
import logging
import numpy as np 

def main():
    # config_filename = os.path.join(gibson2.example_config_path, 'turtlebot_demo.yaml')
    config_filename = "/home/vkmittal14/WORKSPACE/TUD/vk_igibson/iGibson/igibson_usage/configs/exp_config.yaml"
    env = iGibsonEnv(config_file=config_filename, mode='gui')
    for j in range(10):
        env.reset()
        for i in range(100):
            with Profiler('Environment action step'):
                action = env.action_space.sample()
                action = np.zeros_like(action)
                # action[0] = 0
                # action[1] = 0
                state, reward, done, info = env.step(action)
                if done:
                    logging.info(
                        "Episode finished after {} timesteps".format(i + 1))
                    break
    env.close()


if __name__ == "__main__":
    main()
