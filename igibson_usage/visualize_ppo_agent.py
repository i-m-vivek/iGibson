import os
from time import time
from collections import deque
import random
import numpy as np
import sys
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

import hrl4in
from hrl4in.utils.logging import logger
from hrl4in.rl.ppo import PPO, Policy, RolloutStorage
from hrl4in.utils.utils import *
from hrl4in.utils.args import *

import gibson2
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.envs.parallel_env import ParallelNavEnv

def main():
    parser = argparse.ArgumentParser()
    add_ppo_args(parser)
    add_env_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.pth_gpu_id))
    config_file = os.path.join(
        "/home/vkmittal14/WORKSPACE/TUD/vk_igibson/iGibson/igibson_usage/new_configs", args.config_file
    )
    def load_env(env_mode, device_idx):
        return iGibsonEnv(
            config_file=config_file,
            mode="gui",
            action_timestep=args.action_timestep,
            physics_timestep=args.physics_timestep,
            automatic_reset=True,
            device_idx=device_idx,
        )
    sim_gpu_id = [int(gpu_id) for gpu_id in args.sim_gpu_id.split(",")]
    env_id_to_which_gpu = np.linspace(
        0,
        len(sim_gpu_id),
        num=args.num_train_processes + args.num_eval_processes,
        dtype=np.int,
        endpoint=False,
    )

    eval_envs = [
        lambda device_idx=sim_gpu_id[env_id_to_which_gpu[env_id]]: load_env(
            "gui", device_idx
        )
        for env_id in range(args.num_train_processes)
    ]
    eval_envs += [lambda: load_env(args.env_mode, sim_gpu_id[env_id_to_which_gpu[-1]])]
    eval_envs = ParallelNavEnv(eval_envs, blocking=False)

    cnn_layers_params = [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]
    action_dim = eval_envs.action_space.shape[0]
    action_mask = np.ones(action_dim)

    if args.use_base_only and (eval_envs._envs[0].config["robot"] == "Tiago_Single"):
        action_mask[2:] = 0
    if args.use_arm_only and (eval_envs._envs[0].config["robot"] == "Tiago_Single"):
        action_mask[:2] = 0 
    actor_critic = Policy(
        observation_space=eval_envs.observation_space,
        action_space=eval_envs.action_space,
        hidden_size=args.hidden_size,
        cnn_layers_params=cnn_layers_params,
        initial_stddev=args.action_init_std_dev,
        min_stddev=args.action_min_std_dev,
        stddev_anneal_schedule=args.action_std_dev_anneal_schedule,
        stddev_transform=torch.nn.functional.softplus,
    )
    actor_critic.to(device)

    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_clipped_value_loss=True,
    )
    ckpt_path = "/home/vkmittal14/WORKSPACE/TUD/vk_igibson/trained_model_wts/hrl4in_new_exp/ckpt.12000.pth"
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(ckpt["state_dict"])

    observations = eval_envs.reset()
    batch = batch_obs(observations)
    for sensor in batch:
        batch[sensor] = batch[sensor].to(device)
    recurrent_hidden_states = torch.zeros(eval_envs._num_envs, args.hidden_size, device=device)
    masks = torch.zeros(eval_envs._num_envs, 1, device=device)
    for _ in range(50000):
        import pdb; pdb.set_trace()
        with torch.no_grad():
            _, actions, _, recurrent_hidden_states = actor_critic.act(
                batch,
                recurrent_hidden_states,
                masks,
                deterministic=True,
                update=0,
            )
        actions_np = actions.cpu().numpy()
        actions_np = actions_np*action_mask
        print(actions_np)
        outputs = eval_envs.step(actions_np)

        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        batch = batch_obs(observations)
        for sensor in batch:
            batch[sensor] = batch[sensor].to(device)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=device,
        )

if __name__ == "__main__":
    main()
