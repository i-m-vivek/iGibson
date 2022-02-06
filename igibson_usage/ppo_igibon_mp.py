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
from gibson2.envs.mp_igibson_env import iGibsonMPEnv


def main():
    parser = argparse.ArgumentParser()
    add_ppo_args(parser)
    add_env_args(parser)
    add_common_args(parser)
    args = parser.parse_args()

    (
        ckpt_folder,
        ckpt_path,
        start_epoch,
        start_env_step,
        summary_folder,
        log_file,
    ) = set_up_experiment_folder(
        args.experiment_folder, args.checkpoint_index, args.use_checkpoint
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:{}".format(args.pth_gpu_id))
    logger.add_filehandler(log_file)

    if not args.eval_only:
        writer = SummaryWriter(log_dir=summary_folder)
    else:
        writer = None

    for p in sorted(list(vars(args))):
        logger.info("{}: {}".format(p, getattr(args, p)))

    config_file = os.path.join(
        os.path.dirname(gibson2.__file__), "../igibson_usage/configs", args.config_file
    )

    assert os.path.isfile(config_file), "config file does not exist: {}".format(
        config_file
    )

    for (k, v) in parse_config(config_file).items():
        logger.info("{}: {}".format(k, v))

    train_env = iGibsonMPEnv(
        config_file=config_file,
        mode="headless",
        action_timestep=args.action_timestep,
        physics_timestep=args.physics_timestep,
        automatic_reset=True,
        device_idx=0,
    )
    eval_env = iGibsonMPEnv(
        config_file=config_file,
        mode="headless",
        action_timestep=args.action_timestep,
        physics_timestep=args.physics_timestep,
        automatic_reset=True,
        device_idx=0,
    )
    cnn_layers_params = [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]
    action_dim = train_env.action_space.shape[0]
    action_mask = np.ones(action_dim)

    if args.use_base_only and (train_env._envs[0].config["robot"] == "Tiago_Single"):
        action_mask[2:] = 0
    if args.use_arm_only and (train_env._envs[0].config["robot"] == "Tiago_Single"):
        action_mask[:2] = 0 

    actor_critic = Policy(
        observation_space=train_env.observation_space,
        action_space=3,
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
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(ckpt["state_dict"])
        logger.info("loaded checkpoing: {}".format(ckpt_path))

    logger.info(
        "agent number of parameters: {}".format(
            sum(param.numel() for param in agent.parameters())
        )
    )
    observations = train_env.reset()

    batch = batch_obs(observations)

    rollouts = RolloutStorage(
        args.num_steps,
        1,
        train_env.observation_space,
        3,
        args.hidden_size,
    )

    episode_rewards = torch.zeros(1, 1)
    episode_success_rates = torch.zeros(1, 1)
    episode_lengths = torch.zeros(1, 1)
    episode_collision_steps = torch.zeros(1, 1)
    episode_counts = torch.zeros(1, 1)
    current_episode_reward = torch.zeros(1, 1)

    t_start = time()
    env_time = 0
    pth_time = 0
    count_steps = start_env_step

    for update in range(start_epoch, args.num_updates):
        update_lr(
            agent.optimizer,
            args.lr,
            update,
            args.num_updates,
            args.use_linear_lr_decay,
            0,
        )

        agent.clip_param = args.clip_param * (1 - update / args.num_updates)

        for step in range(args.num_steps):
            t_sample_action = time()
            with torch.no_grad():
                step_observation = {
                    k: v[step] for k, v in rollouts.observations.items()
                }

                # values: [num_processes, 1]
                # actions: [num_processes, 1]
                # actions_log_probs: [num_processes, 1]
                # recurrent_hidden_states: [num_processes, hidden_size]
                (
                    values,
                    actions,
                    actions_log_probs,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    step_observation,
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                    update=update,
                )

            pth_time += time() - t_sample_action
            t_step_env = time()

            actions_np = actions.cpu().numpy()
            outputs = train_env.step(actions_np)
            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
            env_time += time() - t_step_env

            t_update_stats = time()
            batch = batch_obs(observations)
            rewards = torch.tensor(rewards, dtype=torch.float)
            rewards = rewards.unsqueeze(1)
            masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones], dtype=torch.float
            )
            success_masks = torch.tensor(
                [
                    [1.0] if done and "success" in info and info["success"] else [0.0]
                    for done, info in zip(dones, infos)
                ],
                dtype=torch.float,
            )
            lengths = torch.tensor(
                [
                    [float(info["episode_length"])]
                    if done and "episode_length" in info
                    else [0.0]
                    for done, info in zip(dones, infos)
                ],
                dtype=torch.float,
            )
            collision_steps = torch.tensor(
                [
                    [float(info["collision_step"])]
                    if done and "collision_step" in info
                    else [0.0]
                    for done, info in zip(dones, infos)
                ],
                dtype=torch.float,
            )
            current_episode_reward += rewards
            episode_rewards += (1 - masks) * current_episode_reward
            episode_success_rates += success_masks
            episode_lengths += lengths
            episode_collision_steps += collision_steps
            episode_counts += 1 - masks
            current_episode_reward *= masks

            rollouts.insert(
                batch,
                recurrent_hidden_states,
                actions,
                actions_log_probs,
                values,
                rewards,
                masks,
            )
            count_steps += 1
            pth_time += time() - t_update_stats

        t_update_model = time()
        with torch.no_grad():
            last_observation = {k: v[-1] for k, v in rollouts.observations.items()}
            next_value = actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        value_loss, action_loss, dist_entropy = agent.update(rollouts, update=update)
        rollouts.after_update()
        pth_time += time() - t_update_model