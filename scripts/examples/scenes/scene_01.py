# This code is an example of scene configuration for a UR5 robot in Isaac Lab.

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


# import the UR5 configuration
from ur5_isaaclab.robots.ur5_cfg import UR5_CFG

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import MultiUsdFileCfg

# define the path to the package root
from ur5_isaaclab import package_root

# You can select a variety of objects, is all are selected, the scene will randomly choose one of them
usd_paths = [
    # f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
    f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    # os.path.join(package_root, "assets", "035_power_drill.usd"),
    # os.path.join(package_root, "assets", "tuna_can.usd"),
    # os.path.join(package_root, "assets", "banana.usd"),
    # os.path.join(package_root, "assets", "mug.usd"),
    # os.path.join(package_root, "assets", "marker.usd"),
]


@configclass
class scene_config(InteractiveSceneCfg):
    # Define a ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )
    # Define a light source
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )
    # We import a table where the robot will operate
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=UsdFileCfg(
            usd_path=os.path.join(package_root, "assets", "table.usd")
        )
    )
    # import the UR5 confguration from the ur5_cfg.py file
    ur5: ArticulationCfg = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/ur5")
    # Set the initial position of the UR5 robot on the scene
    ur5.init_state.pos = (-0.7, 0.0, 0.63)

    # Add the object to the sceene with certain properties
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0, 0.64], rot=[1, 0, 0, 0]),
        spawn=MultiUsdFileCfg(
                usd_path=usd_paths,
                random_choice=True,
                scale=(0.6, 0.6, 0.6),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
        ),
    )
    # define some frame transformers to visualize the frames of the robot and the object
    # some frame transformations are only for visualization, they can be disabled on the debug_vis parameter

    def __post_init__(self):
        super().__post_init__()
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.07, 0.07, 0.07)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.cube_transform = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/ur5/base",
            target_frames=[FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object"
            )],
            debug_vis=True,
            visualizer_cfg=marker_cfg
        )

        self.ee_transform = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/ur5/base",
            target_frames=[FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/ur5/wrist_3_link",
                offset=OffsetCfg(pos=[0.0, 0.23, 0])  # offset for the gripper frame
            )],
            debug_vis=True,
            visualizer_cfg=marker_cfg
        )
