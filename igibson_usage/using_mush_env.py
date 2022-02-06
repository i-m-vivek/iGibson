import numpy as np
from gibson2.envs.mushrl_mp_igibson_env_basearm import iGibsonMPEnv
import pdb
env = iGibsonMPEnv(
    config_file="new_configs/tiago_basearm_random_reaching_relmogen.yaml",
    mode="gui",
)
print(env.info.observation_space.shape)
action = np.array([0.75, 0.75, 0.75, 0, 0.4, 0.4, 0.4])
while True:
    pdb.set_trace()
    state, reward, done, _ = env.step(action)