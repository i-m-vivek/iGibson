from mushroom_rl.environments.gym_env import Gym
import pdb 
pdb.set_trace()
horizon = 200
gamma = 0.99
mdp = Gym("Pendulum-v0", horizon, gamma)
