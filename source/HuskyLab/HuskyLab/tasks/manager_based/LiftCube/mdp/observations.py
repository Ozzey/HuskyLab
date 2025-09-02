# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def scene_graph_embedding_from_pickle(env: "ManagerBasedRLEnv", scene_file: str, embed_dim: int = 32) -> torch.Tensor:
    """Load precomputed per-scene graph embeddings from a pickle file or a directory of pickles.

    Returns a torch.Tensor with shape (num_envs, embed_dim). If embeddings are missing, returns zeros.
   
    Supported inputs for `scene_file`:
    - a directory containing files named `scene_{i}.pkl` (or any .pkl files) which include a
      `graph_embedding` field (or attribute) with shape (embed_dim,) or (1, embed_dim).
    - a single pickle file containing a list/iterable of scene objects / dicts that include
      `graph_embedding` entries.

    This loader is intentionally conservative and falls back to zeros on any error.
    """
    import os
    import pickle
    import numpy as _np

    # determine num_envs from a scene asset if possible
    num_envs = None
    try:
        # try robot first
        robot = env.scene["robot"]
        num_envs = int(robot.data.root_pos_w.shape[0])
    except Exception:
        try:
            # fallback to env attribute
            num_envs = int(getattr(env, "num_envs", getattr(env, "unwrapped", getattr(env, "env", 1)).num_envs))
        except Exception:
            num_envs = 1

    embs = _np.zeros((num_envs, embed_dim), dtype=_np.float32)

    def _extract_graph_embedding(obj):
        # obj might be dict-like, an object with attribute, or a bare array
        if obj is None:
            return None
        if isinstance(obj, _np.ndarray):
            arr = obj
        elif isinstance(obj, dict) and "graph_embedding" in obj:
            arr = _np.asarray(obj["graph_embedding"])
        elif hasattr(obj, "graph_embedding"):
            arr = _np.asarray(getattr(obj, "graph_embedding"))
        else:
            # try common keys
            for key in ("embedding", "emb", "graph_emb"):
                if isinstance(obj, dict) and key in obj:
                    return _np.asarray(obj[key])
            return None

        if arr.ndim == 1 and arr.shape[0] == embed_dim:
            return arr
        if arr.ndim == 2 and arr.shape[-1] == embed_dim:
            return arr.reshape(-1)[:embed_dim]
        return None

    # Load from directory of pickles
    try:
        if os.path.isdir(scene_file):
            files = sorted([f for f in os.listdir(scene_file) if f.endswith(".pkl")])
            # prefer files named scene_0, scene_1 ... otherwise use order
            for i in range(min(num_envs, len(files))):
                p = os.path.join(scene_file, files[i])
                try:
                    with open(p, "rb") as fh:
                        data = pickle.load(fh)
                    e = _extract_graph_embedding(data)
                    if e is not None:
                        embs[i, : e.shape[0]] = e[:embed_dim]
                except Exception:
                    # leave zeros for this env
                    continue
            return torch.from_numpy(embs)

        # Load from a single pickle file (could be a list of scenes)
        if os.path.isfile(scene_file):
            with open(scene_file, "rb") as fh:
                data = pickle.load(fh)
            # If it's an array of embeddings
            if isinstance(data, (list, tuple, _np.ndarray)):
                arr = _np.asarray(data)
                if arr.ndim == 2 and arr.shape[1] == embed_dim:
                    n = min(num_envs, arr.shape[0])
                    embs[:n] = arr[:n]
                    return torch.from_numpy(embs)
                # try to extract graph_embedding per element
                for i in range(min(num_envs, len(arr))):
                    e = _extract_graph_embedding(arr[i])
                    if e is not None:
                        embs[i, : e.shape[0]] = e[:embed_dim]
                return torch.from_numpy(embs)

            # single scene object
            e = _extract_graph_embedding(data)
            if e is not None:
                # broadcast the single embedding to all envs
                for i in range(num_envs):
                    embs[i, : e.shape[0]] = e[:embed_dim]
                return torch.from_numpy(embs)

    except Exception:
        # any problem -> return zeros
        return torch.from_numpy(embs)

    return torch.from_numpy(embs)
