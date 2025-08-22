# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# This file is used to play a checkpoint of an RL agent from RSL-RL. It will create a onnx file as well as a pt file. THe modification to the code is to allow to test the performance of grasping.


"""Script to play a checkpoint if an RL agent from RSL-RL."""
"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--episodes_per_env", type=int, default=100, help="Number of episodes to run per environment."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import time
import torch
import numpy as np
import pandas as pd

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
import ur5_isaaclab.tasks  # noqa: F401


def main():
    """Play with RSL-RL agent: report first time z > 0.1m each episode and save CSV."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # locate checkpoint
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[ERROR] No pretrained checkpoint.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create and wrap env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load trained model
    ppo = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo.load(resume_path)
    policy = ppo.get_inference_policy(device=env.unwrapped.device)

    # export artifacts
    try:
        policy_nn = ppo.alg.policy
    except AttributeError:
        policy_nn = ppo.alg.actor_critic
    export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo.obs_normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=ppo.obs_normalizer, path=export_dir, filename="policy.onnx")

    # prepare logging across multiple envs
    num_envs = args_cli.num_envs or 1
    TARGET_EPISODES = args_cli.episodes_per_env
    lift_events = []
    step_counts = np.zeros(num_envs, dtype=int)
    threshold_crossed = [False] * num_envs
    episodes_done = np.zeros(num_envs, dtype=int)

    # initial observations
    obs, _ = env.get_observations()
    dt = env.unwrapped.step_dt

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, _ = env.step(actions)

        step_counts += 1

        # threshold check
        arr = obs.detach().cpu().numpy() if torch.is_tensor(obs) else np.array(obs)
        for i in range(num_envs):
            if not threshold_crossed[i] and arr[i][26] > 0.1:
                t_i = step_counts[i] * dt
                print(f"Env {i}, Episode {episodes_done[i]+1}: z={arr[i][26]:.3f}m at {t_i:.3f}s → good lift")
                lift_events.append({
                    "env":    i,
                    "episode":episodes_done[i] + 1,
                    "z":      float(arr[i][26]),
                    "time_s": t_i,
                })
                threshold_crossed[i] = True

        # handle done
        for i, done in enumerate(dones):
            if done:
                episodes_done[i]    += 1
                step_counts[i]       = 0
                threshold_crossed[i] = False

        # stop when done
        if np.all(episodes_done >= TARGET_EPISODES):
            break

        # real-time pacing
        sleep = dt - (time.time() - start_time)
        if args_cli.real_time and sleep > 0:
            time.sleep(sleep)

    env.close()

    # save CSV
    df = pd.DataFrame(lift_events)
    csv_path = os.path.join(log_dir, "lift_events.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved lift events to {csv_path}")

if __name__ == "__main__":
    main()
    simulation_app.close()
