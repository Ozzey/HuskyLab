
# This code is an exaple of a rutine on how to control the UR5 robot using the Differential IK controller
# To launch from package source:  python scripts/examples/grab_cube.py --num_envs=1

# Launch the simulation app with the command line arguments
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR5 with Differential IK Controller")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")  # you can change the number of envrioments that run in parallel
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import os, math, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from ur5_isaaclab.robots.ur5_cfg import UR5_CFG
from scenes.scene_01 import scene_config 
from controllers.controller import control_gripper

# --- Función modificada de control_gripper ---
def control_gripper(ur5, open=True):
    left_joint_name = "robotiq_85_left_knuckle_joint"
    right_joint_name = "robotiq_85_right_knuckle_joint"

    # Obtener índices desde la lista de nombres
    left_index = ur5.data.joint_names.index(left_joint_name)
    right_index = ur5.data.joint_names.index(right_joint_name)

    if open:
        target_left = 0.0
        target_right = 0.0
    else:
        # Se usan valores positivos para ambos dedos
        target_left = math.radians(41.0)
        target_right = math.radians(41.0)

    # Crear tensor con los valores y los índices
    gripper_target = torch.tensor([[target_left, target_right]], dtype=torch.float32, device=ur5.device)
    joint_ids = torch.tensor([left_index, right_index], device=ur5.device)

    # Depuración: imprimir valores establecidos
    #print(f"Setting gripper targets -> Left: {target_left:.4f} rad, Right: {target_right:.4f} rad")

    ur5.set_joint_position_target(gripper_target, joint_ids=joint_ids)

def run_simulator(sim, scene):
    sim_dt = sim.get_physics_dt()
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    
    phase = "approach"  # Puede ser "approach" o "move"
    gripper_closed = False
    wait_delay_before_closing = 4.0  # Tiempo (en segundos) antes de cerrar la garra
    wait_delay_after_closing = 4.0   # Tiempo de espera después de cerrar la garra
    approach_time = None
    post_grasp_target = torch.tensor([[0.0, 0.5, 0.5]], device=sim.device).repeat(scene.num_envs, 1)
    
    robot_entity_cfg = SceneEntityCfg(
        "ur5",
        joint_names=[
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ],
        body_names=["wrist_3_link"]
    )
    robot_entity_cfg.resolve(scene)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1

    ur5 = scene["ur5"]
    joint_pos = ur5.data.default_joint_pos.clone()
    joint_vel = ur5.data.default_joint_vel.clone()
    ur5.write_joint_state_to_sim(joint_pos, joint_vel)
    ur5.reset()
    diff_ik_controller.reset()
    
    printed_initial = False

    while simulation_app.is_running():
        cube = scene["cube"]
        cube_world_pos = cube.data.body_state_w.cpu().numpy()
        cube_transform_sensor = scene["cube_transform"]
        cube_rel_pos = cube_transform_sensor.data.target_pos_source
        if cube_rel_pos.dim() == 3:
            cube_rel_pos = cube_rel_pos.squeeze(1)
        pregrasp_offset = 0.27
        cube_rel_pos[:, 2] = 0.01 + pregrasp_offset

        current_target = cube_rel_pos if phase == "approach" else post_grasp_target

        robot_base_pos = ur5.data.root_state_w[:, :3].cpu().numpy()
        ee_pose_e = ur5.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_world_pos = ee_pose_e[:, :3].cpu().numpy()
        if not printed_initial:
            print("Robot base:", robot_base_pos)
            print("Cube:", cube_world_pos)
            print("Gripper:", ee_world_pos)
            printed_initial = True

        target_orient = torch.tensor([0.707, -0.707, 0, 0], device=sim.device).repeat(scene.num_envs, 1)
        ik_command = torch.cat([current_target, target_orient], dim=1)
        diff_ik_controller.set_command(ik_command)

        jacobian = ur5.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        joint_pos = ur5.data.joint_pos[:, robot_entity_cfg.joint_ids]

        root_pose_e = ur5.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_e[:, :3], root_pose_e[:, 3:7],
            ee_pose_e[:, :3], ee_pose_e[:, 3:7]
        )
        
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        ur5.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        error = torch.norm(ee_pos_b - current_target, dim=1)
        threshold = 0.05

        if phase == "approach" and torch.all(error < threshold):
            if approach_time is None:
                approach_time = time.time()
            else:
                elapsed = time.time() - approach_time
                if elapsed < wait_delay_before_closing:

                    pass
                elif elapsed < (wait_delay_before_closing + wait_delay_after_closing):
                    if not gripper_closed:
                        control_gripper(ur5, open=False)
                        gripper_closed = True
                        print("Gripper closed. Waiting...")
                    else:
                        control_gripper(ur5, open=False)
                else:
                    phase = "move"
                    print("Switching to move phase.")
        else:
            approach_time = None

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        ur5.update(sim_dt)

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