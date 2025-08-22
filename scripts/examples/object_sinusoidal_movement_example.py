# This code is an example on how we can apply a sinusoidal movement to a prim in the scene.
# Note that the implementation during training is a little diferent from this one, but the idea is similar.

# To launch from package source:  python scripts/examples/object_sinusoidal_movement_example.py --num_envs=1

# RUn the simulation app first
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sinusoidal Movement Example")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from scenes.scene_02 import scene_config


def run_simulator(sim, scene):
    sim_dt = sim.get_physics_dt()

    # parameters for sinusoidal movement in the X axis
    amplitude = 0.25            # meters
    frequency = 0.4            # Hz
    omega = 2 * math.pi * frequency

    # constant speed in Y
    speed_y = 0.3              # m/s

    cube = scene["cube"]
    t = 0.0

    while simulation_app.is_running():
        t += sim_dt

        # Read the current state of the prim
        root_state = cube.data.root_state_w.clone()     # [num_envs, 13]
        lin_vel = root_state[:, 7:10]                   # [vx, vy, vz]
        ang_vel = root_state[:, 10:13]                  # [wx, wy, wz]

        # calculate the sinusoidal movement for each time step
        vx = amplitude * omega * math.cos(omega * t)
        lin_vel[:, 0] = vx
        lin_vel[:, 1] = speed_y

        # write the velocity to the state of the cube
        cube.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=1))

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
