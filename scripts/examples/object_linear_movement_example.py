# This is a simple examplle on how can we apply a constant velocity to a prim in the scene.
# Note that for prims that we desired to interact with the world, we need to add a rigid body and collider, and if needed we can add mass.
# For more complex examples, please refer to training scripts.

# To launch from package source:  python scripts/examples/object_linear_movement_example.py --num_envs=1

# Launch the simulation app with the command line arguments
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Linear Movement Example")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
# import scene without the UR5 robot
from scenes.scene_02 import scene_config


def run_simulator(sim, scene):
    sim_dt = sim.get_physics_dt()
    speed_y = 0.3
    step_count = 0
    threshold_steps = 250

    cube = scene["cube"]

    while simulation_app.is_running():
        # Read the current state to preserve the Z component of velocity (gravity)
        root_state = cube.data.root_state_w.clone()      # [num_envs, 13]
        lin_vel = root_state[:, 7:10]                    # [vx, vy, vz]
        ang_vel = root_state[:, 10:13]                   # [wx, wy, wz]

        # Assign the desired vel
        lin_vel[:, 1] = speed_y

        # write the velocity to the state of the cube
        cube.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=1))

        # update the counter in order to change direction
        step_count += 1
        if step_count >= threshold_steps:
            speed_y = -speed_y
            step_count = 0

        # simulation step
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        cube.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([-0.8, 2.0, 1.3], [-0.2, 0.3, 0.70])

    scene_cfg = scene_config(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
