# This is a basic example on how to use the Differential IK controller with the UR5
# This code imports the scene configuration (robot, object,ground, light)


# To launch from package source:   python scripts/examples/diff_ik_example.py --num_envs=1

# First the simulation needs to be set up, this is done using the AppLauncher from isaaclab.app
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="UR5 with Differential IK Controller")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms


# import the scene configuration declared in the scene_01.py file inside the scenes directory
from scenes.scene_01 import scene_config


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()

    # create a differential IK controller object
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Robot Joints used by the controller
    robot_entity_cfg = SceneEntityCfg(
        "ur5",
        joint_names=[
            "shoulder_pan_joint", 
            "shoulder_lift_joint", 
            "elbow_joint", 
            "wrist_1_joint", 
            "wrist_2_joint", 
            "wrist_3_joint"
        ],
        body_names=["wrist_3_link"]
    )
    robot_entity_cfg.resolve(scene)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1

    # Set a goal for the end-effector
    ee_goal = torch.tensor([[0.5, -0.2, 0.5, 1.0, 0.0, 0.0, 0.0]], device=sim.device)
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=sim.device)
    ik_commands[:] = ee_goal

    ur5 = scene["ur5"]
    joint_pos = ur5.data.default_joint_pos.clone()
    joint_vel = ur5.data.default_joint_vel.clone()
    ur5.write_joint_state_to_sim(joint_pos, joint_vel)
    ur5.reset()
    diff_ik_controller.reset()
    diff_ik_controller.set_command(ik_commands)

    while simulation_app.is_running():
        jacobian = ur5.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = ur5.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = ur5.data.root_state_w[:, 0:7]
        joint_pos = ur5.data.joint_pos[:, robot_entity_cfg.joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7],
            ee_pose_w[:, :3], ee_pose_w[:, 3:7]
        )
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # print("=== Differential IK Controller ===")
        # print("Input:")
        # print(f"  ee_pos_b: {ee_pos_b.cpu().numpy()}")
        # print(f"  ee_quat_b: {ee_quat_b.cpu().numpy()}")
        # print(f"  joint_pos: {joint_pos.cpu().numpy()}")
        # print("Output:")
        # joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        # print(f"  joint_pos_des: {joint_pos_des.cpu().numpy()}")

        ur5.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        ur5.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    scene_cfg = scene_config(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
