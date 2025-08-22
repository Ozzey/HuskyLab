# THis the gripper controller for the Robotiq 2F-85

import math
import torch


def control_gripper(ur5, open=True):
    # The two joints of the gripper that have a revolute joints and a drive to control them, in the real gripper we only control 1 DoF that controls both fingers, but for simmulation purposes, we control two in order to have the same force applied by the two sides.
    left_joint_name = "robotiq_85_left_knuckle_joint"
    right_joint_name = "robotiq_85_right_knuckle_joint"

    left_index = ur5.data.joint_names.index(left_joint_name)
    right_index = ur5.data.joint_names.index(right_joint_name)

    if open:
        target_left = 0.0
        target_right = 0.0
    else:
        target_left = math.radians(41.0)
        target_right = math.radians(41.0)

    gripper_target = torch.tensor([[target_left, target_right]], dtype=torch.float32, device=ur5.device)
    joint_ids = torch.tensor([left_index, right_index], device=ur5.device)
    # We are only applying position control, but in the future we could implement force control.
    print(f"Setting gripper targets -> Left: {target_left:.4f} rad, Right: {target_right:.4f} rad")

    ur5.set_joint_position_target(gripper_target, joint_ids=joint_ids)
