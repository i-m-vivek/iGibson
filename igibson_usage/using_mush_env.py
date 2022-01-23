import numpy as np
from gibson2.envs.mushrl_mp_igibson_env import iGibsonMPEnv
import pdb
env = iGibsonMPEnv(
    config_file="new_configs/tiago_base_reaching_relmogen.yaml",
    mode="gui",
)
print(env.info.observation_space.shape)
while True:
    action = np.array([2, 5, 0.75])
    state, reward, done, _ = env.step(action)
