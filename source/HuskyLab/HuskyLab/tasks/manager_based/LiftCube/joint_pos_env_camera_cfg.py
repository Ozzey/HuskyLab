import os
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass


from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg

import HuskyLab.tasks.manager_based.LiftCube.mdp as mdp

from .joint_pos_env_cfg import UR5CubeLiftEnvCfg
from .lift_env_cfg import ObjectTableSceneCfg
from HuskyLab import package_root
##
# Scene definition
##


@configclass
class LiftCubeRGBCameraSceneCfg(ObjectTableSceneCfg):


    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Camera",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(2.0, 0.0, 0.1),  # Moved closer to the workspace
    #         rot=(0.5, 0.5, 0.5, 0.5),  # Oriented to view cube and origin
    #         convention="opengl"
    #     ),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=11.2,
    #         focus_distance=2.2,
    #         horizontal_aperture=6.4,
    #         clipping_range=(0.1, 1.0e5)
    #     ),
    #     width=640,
    #     height=480,
    #     update_period=0.1
    # )

    room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/room",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.0, -3.0, -1.5]),
        spawn=UsdFileCfg(usd_path=os.path.join(package_root, "Assets", "room.usd"),
        )
    )

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/gripper_cam",
                update_period=0.01,
                height=64,
                width=64,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, horizontal_aperture=45.0, clipping_range=(0.1, 20)
                ),
                offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.1,0.0), rot=( 0.5, 0.5, -0.5, 0.5 ), convention="opengl"),
            )




# 1,0,0,0 -> 180,0,0 


@configclass
class RGBObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = RGBCameraPolicyCfg()


@configclass
class ResNet18ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen ResNet18."""

        image = ObsTerm(
            func=mdp.image_features,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "model_name": "resnet18"},
        )

    policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()




@configclass
class LiftCubeRGBCameraEnvCfg(UR5CubeLiftEnvCfg):
    """Configuration for the Cube environment with RGB camera."""

    scene: LiftCubeRGBCameraSceneCfg = LiftCubeRGBCameraSceneCfg(num_envs=512, env_spacing=20)
    observations: RGBObservationsCfg = RGBObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # remove ground as it obstructs the camera
        self.scene.ground = None
        # viewer settings
        self.viewer.eye = (7.0, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 2.5)


@configclass
class LiftCubeResNet18CameraEnvCfg(LiftCubeRGBCameraEnvCfg):
    """Configuration for the Cube environment with ResNet18 features as observations."""

    observations: ResNet18ObservationCfg = ResNet18ObservationCfg()