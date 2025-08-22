# Observations for the lift task with a moving object.
from __future__ import annotations
import isaaclab.sim as sim_utils

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
import carb
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import math
def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def continuous_move(env, valid_env_ids, *, asset_cfg, speed=None, speed_range=None, threshold_steps=40):
    # lazy init
    if not hasattr(env, "_move_counter"):
        env._move_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        if speed_range is not None:
            low, high = speed_range
            speeds = torch.empty(env.num_envs, device=env.device).uniform_(low, high)
            env._move_speed = speeds
        else:
            env._move_speed = torch.full((env.num_envs,), speed, device=env.device)
        env._move_thresh = threshold_steps

    idx = valid_env_ids
    obj = env.scene[asset_cfg.name]
    root = obj.data.root_state_w.clone()
    lin = root[:, 7:10]
    ang = root[:, 10:13]

    lin[idx, 1] = env._move_speed[idx]
    obj.write_root_velocity_to_sim(torch.cat([lin, ang], dim=1))

    carb.log_info(f"[continuous_move] envs={idx.tolist()}, "
                  f"counter={int(env._move_counter[idx][0])}, "
                  f"speed={env._move_speed[idx][0]:.3f}")

    env._move_counter[idx] += 1
    flip = env._move_counter >= env._move_thresh
    if flip.any():
        # Invert speed for flipped environments
        env._move_speed[flip] = -env._move_speed[flip]
        env._move_counter[flip] = 0
