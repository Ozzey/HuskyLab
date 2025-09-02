1. Put `torch.save(loss, f'/workspace/HuskyLab/losses/loss{self.epoch_num}.pt')` in line 177 `/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/rl_games/algos_torch/a2c_continuous.py`

2. Assets put in `/workspace/HuskyLab/source/HuskyLab/HuskyLab/Assets`

3. Running training `root@rentserv:/workspace/HuskyLab# ../isaaclab/isaaclab.sh -p scripts/rl_games/train.py --task=Isaac-LiftCube-RGB-v0 --headless --enable_cameras --num_envs=64`

4. in `/workspace/HuskyLab/source/HuskyLab/HuskyLab/tasks/manager_based/LiftCube/agents/rl_games_camera_ppo_cfg.yaml` changing the
    * `max_epochs` = 60
    * `minibatch_size` = 384
    * `num_envs` = 64 while running, 
    * `minibatch_size`, `num_envs` both should be decreased by x, which is divisible by 4
    *  `minibatch_size` = 24576
    * `num_envs` =  4096, i divided both by `64`.
     