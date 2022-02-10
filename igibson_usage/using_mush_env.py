import numpy as np
from gibson2.envs.mushrl_mp_igibson_env_basearm import iGibsonMPEnv
import pdb
env = iGibsonMPEnv(
    config_file="new_configs/tiago_basearm_relmogen.yaml",
    mode="gui",
)
print(env.info.observation_space.shape)
action = np.array([2.75, 1.75, 0.75, 0, 0.4, 0.4, 0.46])
env.env.robots[0].set_orientation([0, 0, np.sin(-3*np.pi / 8), np.cos(-3*np.pi / 8)])
# env.env.robots[0].set_orientation([0, 0, 0, 1])
while True:
    pdb.set_trace()
    state, reward, done, _ = env.step(action)