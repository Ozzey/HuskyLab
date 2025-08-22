import math
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import  os
# Pre-defined configs
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from ur5_isaaclab.robots.ur5_cfg import UR5_CFG
from ur5_isaaclab.tasks.manager_based.lift_dynamic import mdp
from ur5_isaaclab.tasks.manager_based.lift_dynamic.lift_env_cfg import LiftEnvCfg
from ur5_isaaclab import package_root
from isaaclab.sim import MultiUsdFileCfg
import isaaclab.sim as sim_utils # for loading multiple USD files


# you can select multiple objects, if you uncomment all of them, one at random will be seleccted at each
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
class UR5CubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):

        super().__post_init__()

        self.scene.robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "robotiq_85_left_knuckle_joint",
                "robotiq_85_right_knuckle_joint"
            ],
            open_command_expr={
                "robotiq_85_left_knuckle_joint": 0.0,
                "robotiq_85_right_knuckle_joint": 0.0
            },
            close_command_expr={
                "robotiq_85_left_knuckle_joint": math.radians(41.0),
                "robotiq_85_right_knuckle_joint": math.radians(41.0)
            },
        )

        self.commands.object_pose.body_name = "gripper_link"

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            debug_vis=False,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0.0, 0.0],                  # initial position of the object, later can be change by the environment
                rot=[0.70711, -0.70711, 0.0, 0.0],    # change orientation to have the object upright
                lin_vel=[0.0, 0.0, 0.0],    
                ang_vel=[0.0, 0.0, 0.0],     
            ),
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
                    kinematic_enabled=False,
                ),
            ),
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/world",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0, 0.0]),
                ),
            ],
        )

        object_marker_cfg = FRAME_MARKER_CFG.copy()
        object_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        object_marker_cfg.prim_path = "/Visuals/ObjectMarker"
        self.scene.object_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Object",  
            debug_vis=True,
            visualizer_cfg=object_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    name="object_frame",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )

        world_marker_cfg = FRAME_MARKER_CFG.copy()
        world_marker_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        world_marker_cfg.prim_path = "/Visuals/WorldMarker"
        self.scene.world_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/world",
            debug_vis=False,
            visualizer_cfg=world_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/world",
                    name="world_frame",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )


@configclass
class UR5CubeLiftEnvCfg_PLAY(UR5CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

