# This code is an example on how you can control the gripper in simulation

# To launch from package source:  python scripts/examples/gripper_action_example.py --num_envs=1
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR5")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from scenes.scene_01 import scene_config  # import scene with the robot, table, cameras, etc
from controllers.controller import control_gripper  # import gripper controller


def run_simulator(sim, scene):
    sim_dt = sim.get_physics_dt()
    ur5 = scene["ur5"]

    joint_pos = ur5.data.default_joint_pos.clone()
    joint_vel = ur5.data.default_joint_vel.clone()
    ur5.write_joint_state_to_sim(joint_pos, joint_vel)
    ur5.reset()

    gripper_open = True
    toggle_interval = 10.0
    last_toggle_time = time.time()

    while simulation_app.is_running():
        current_time = time.time()
        if current_time - last_toggle_time > toggle_interval:
            gripper_open = not gripper_open
            if gripper_open:
                print("opening the gripper")
            else:
                print("closing the gripper")
            control_gripper(ur5, open=gripper_open)
            last_toggle_time = current_time
        ur5.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        ur5.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])  # This is the default camera view when simulation starts (where the camera is), (where is pointing at)

    scene_cfg = scene_config(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
