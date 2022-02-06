from gibson2.envs.mp_igibson_env import iGibsonMPEnv
import numpy as np 
import pdb
env = iGibsonMPEnv(
    config_file="configs/tiago_motion_planning.yaml",
    mode = "headless",
)

while True:
    pdb.set_trace()
    action = np.array([2, 5, 0.75])
    state, reward, done, _ = env.do_mp(action, 0)