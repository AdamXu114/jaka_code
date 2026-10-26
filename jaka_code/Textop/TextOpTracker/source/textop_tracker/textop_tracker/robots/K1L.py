import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from textop_tracker.assets import ASSET_DIR

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

K1L_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(#UrdfFileCfg(
       # fix_base=False,
        #replace_cylinders_with_capsules=True,
        usd_path=f"{ASSET_DIR}/K1L_usd/K1L_FULL/K1L_FULL.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        # joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        #     gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        # ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.98),  # 0.95
        # pos=(0.0, 0.0, 1.20),
        joint_pos={
            "left_hip_pitch_joint": 0.0,  # 0.25,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.0,  # 0.5,
            "left_ankle_pitch_joint": 0.0,  # 0.25,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,  # 0.25,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.0,  # 0.5,
            "right_ankle_pitch_joint": 0.0,  # 0.25,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "neck_yaw_joint": 0.0,
            "neck_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": -1.3,
            "left_shoulder_yaw_joint": -1.57,
            "left_elbow_joint": 1.57,
            "left_wrist_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -1.3,
            "right_shoulder_yaw_joint": -1.57,
            "right_elbow_joint": 1.57,
            "right_wrist_joint": 0.0,

        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,

    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 360.0,
                ".*_hip_roll_joint": 320.0,
                ".*_hip_yaw_joint": 128.0,
                ".*_knee_joint": 360.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 9.0,
                ".*_hip_roll_joint": 10.1,
                ".*_hip_yaw_joint": 10.1,
                ".*_knee_joint": 9.0,
            },
            stiffness={
                ".*_hip_pitch_joint": 500.0,
                ".*_hip_roll_joint": 500.0,
                ".*_hip_yaw_joint": 500.0,
                ".*_knee_joint": 500.0,
            },
            damping={
                ".*_hip_pitch_joint": 15.0,
                ".*_hip_roll_joint": 30.0,
                ".*_hip_yaw_joint": 15.0,
                ".*_knee_joint": 15.0,
            },
            armature=0.02,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=96.0,
            velocity_limit_sim=14.7,
            stiffness={
                ".*_ankle_pitch_joint": 100.0,
                ".*_ankle_roll_joint": 50.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 0.5,
            },
            armature=0.01,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                              ".*_elbow_joint", ".*_wrist_joint"],
            effort_limit_sim=96.0,
            velocity_limit_sim=3.14,
            stiffness={
                ".*_shoulder_pitch_joint": 2079.0,  # 500.0,
                ".*_shoulder_roll_joint": 2079.0,  # 500.0,
                ".*_shoulder_yaw_joint": 1444.0,  # 500.0,
                ".*_elbow_joint": 1444.0,  # 300.0,
                ".*_wrist_joint": 314.0,  # 300.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 208.0,  # 30.0,
                ".*_shoulder_roll_joint": 208.0,  # 30.0,
                ".*_shoulder_yaw_joint": 144.0,  # 20.0,
                ".*_elbow_joint": 144.0,  # 10.0,
                ".*_wrist_joint": 31.4,  # 10.0,
            },
            armature={
                ".*_shoulder_pitch_joint": 0.33,
                ".*_shoulder_roll_joint": 0.33,
                ".*_shoulder_yaw_joint": 0.23,
                ".*_elbow_joint": 0.23,
                ".*_wrist_joint": 0.05,
            },
        ),
        "others": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "neck_yaw_joint", "neck_pitch_joint"],
            effort_limit_sim=96.0,
            velocity_limit_sim=3.14,
            stiffness={
                "waist_yaw_joint": 500.0,
                "neck_yaw_joint": 314.0,  # 300.0,
                "neck_pitch_joint": 314.0,  # 300.0,
            },
            damping={
                "waist_yaw_joint": 20.0,
                "neck_yaw_joint": 31.4,  # 10.0,
                "neck_pitch_joint": 31.4,  # 10.0,
            },
            armature={
                "waist_yaw_joint": 0.2,
                "neck_yaw_joint": 0.05,
                "neck_pitch_joint": 0.05,
            },
        ),
    },
)


Khan_mini_CFG = ArticulationCfg(
    # spawn=sim_utils.UsdFileCfg(
    #     usd_path=f"{ASSET_DIR}/Khan_mini_simplified/Khan_mini_simplified.usd",
    #     activate_contact_sensors=True,
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=False,
    #         retain_accelerations=False,
    #         linear_damping=0.0,
    #         angular_damping=0.0,
    #         max_linear_velocity=1000.0,
    #         max_angular_velocity=1000.0,
    #         max_depenetration_velocity=1.0,
    #     ),
    #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #         enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    #     ),
    # ),
spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/Khan_mini_simplified/khan.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65), #0.95
        joint_pos={
            "Left_hip_pitch_joint": 0.0,
            "Left_hip_roll_joint": 0.0,
            "Left_hip_yaw_joint": 0.0,
            "Left_knee_joint": 0.0,
            "Left_ankle_pitch_joint": 0.0,
            "Left_ankle_roll_joint": 0.0,
            "Left_shoulder_pitch_joint": 0.0,
            "Left_shoulder_roll_joint": -1.57,
            "Left_shoulder_yaw_joint": 0.0,
            "Left_elbow_joint": 1.57,
            "Left_wrist_roll_joint": 0.0,
            "Left_wrist_yaw_joint": 0.3,

            "Right_hip_pitch_joint": 0.0,
            "Right_hip_roll_joint": 0.0,
            "Right_hip_yaw_joint": 0.0,
            "Right_knee_joint": 0.0,
            "Right_ankle_pitch_joint": 0.0,
            "Right_ankle_roll_joint": 0.0,
            "Right_shoulder_pitch_joint": 0.0,
            "Right_shoulder_roll_joint": -1.57,
            "Right_shoulder_yaw_joint": 0.0,
            "Right_elbow_joint": 1.57,
            "Right_wrist_roll_joint": 0.0,
            "Right_wrist_yaw_joint": 0.3,

            "waist_yaw_joint": 0.0,
            "Neck_yaw_joint": 0.0,
            "Neck_pitch_joint": 0.0,

        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,

    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[ #0.62 21.91
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_yaw_joint": 120.0,
                ".*_knee_joint": 120.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 11.52,
                ".*_hip_roll_joint": 11.52,
                ".*_hip_yaw_joint": 11.52,
                ".*_knee_joint": 11.52,
            },
            stiffness={
                 ".*_hip_pitch_joint": 187.0,
                ".*_hip_roll_joint": 187.0,
                ".*_hip_yaw_joint": 187.0,
                ".*_knee_joint": 187.0,
            },
            damping={
                ".*_hip_pitch_joint": 18.7,
                ".*_hip_roll_joint": 18.7,
                ".*_hip_yaw_joint": 18.7,
                ".*_knee_joint": 18.7,
            },
            armature=0.03, #JX9-120 model
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=96.0,
            velocity_limit_sim=14.6,
            # stiffness={
            #     ".*_ankle_pitch_joint": 28.50124619574858,  
            #     ".*_ankle_roll_joint": 28.50124619574858,#50.0,
            # },
            # damping={
            #     ".*_ankle_pitch_joint": 1.814445686584846, #2.0,
            #     ".*_ankle_roll_joint": 1.814445686584846, #0.5,
            # },
            # armature=2*0.003609725,#0.025,
            stiffness={
                ".*_ankle_pitch_joint": 100.0,
                ".*_ankle_roll_joint": 50.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 0.5,
            },
            armature=0.016,
            
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 96.0,
                ".*_shoulder_roll_joint": 96.0,
                ".*_shoulder_yaw_joint": 36.0,
                ".*_elbow_joint": 36.0,
                ".*_wrist_.*_joint": 8.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 11.5,
                ".*_shoulder_roll_joint": 11.5,
                ".*_shoulder_yaw_joint": 15.7,
                ".*_elbow_joint": 15.7,
                ".*_wrist_.*_joint": 12.5,
            },
            # stiffness={
            #     ".*_shoulder_pitch_joint": 158.0,
            #     ".*_shoulder_roll_joint": 158.0,
            #     ".*_shoulder_yaw_joint": 40.8,
            #     ".*_elbow_joint": 40.8,
            #     ".*_wrist_.*_joint": 13.4,
            # },
            # damping={
            #     ".*_shoulder_pitch_joint": 15.8,
            #     ".*_shoulder_roll_joint": 15.8,
            #     ".*_shoulder_yaw_joint": 4.0,
            #     ".*_elbow_joint": 4.0,
            #     ".*_wrist_.*_joint": 1.3,
            # },
            # armature={
            #     ".*_shoulder_pitch_joint": 0.025,
            #     ".*_shoulder_roll_joint": 0.025,
            #     ".*_shoulder_yaw_joint": 0.01, #0.003
            #     ".*_elbow_joint": 0.01,
            #     ".*_wrist_.*_joint": 0.01, #0.0005,
            # },
            stiffness={
                ".*_shoulder_pitch_joint": 102.0,
                ".*_shoulder_roll_joint": 102.0,
                ".*_shoulder_yaw_joint": 40.8,
                ".*_elbow_joint": 40.8,
                ".*_wrist_.*_joint": 6.7,
            },
            damping={
                ".*_shoulder_pitch_joint": 10.2,
                ".*_shoulder_roll_joint": 10.2,
                ".*_shoulder_yaw_joint": 4.0,
                ".*_elbow_joint": 4.0,
                ".*_wrist_.*_joint": 0.67,
            },
            armature={
                ".*_shoulder_pitch_joint": 0.016,
                ".*_shoulder_roll_joint": 0.016,
                ".*_shoulder_yaw_joint": 0.01, #0.003
                ".*_elbow_joint": 0.01,
                ".*_wrist_.*_joint": 0.01, #0.0005,
            },
        ),
        "others": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "Neck_yaw_joint", "Neck_pitch_joint"],
            effort_limit_sim={
                "waist_yaw_joint": 120.0,
                "Neck_yaw_joint": 8.0,
                "Neck_pitch_joint": 8.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 120.0,
                "Neck_yaw_joint": 8.0,
                "Neck_pitch_joint": 8.0,
            },
            # stiffness={
            #     "waist_yaw_joint": 187.0,
            #     "Neck_yaw_joint": 13.4,
            #     "Neck_pitch_joint": 13.4,
            # },
            # damping={
            #     "waist_yaw_joint": 18.7,
            #     "Neck_yaw_joint": 1.34,
            #     "Neck_pitch_joint": 1.34,
            # },
            # armature={
            #     "waist_yaw_joint": 0.03,
            #     "Neck_yaw_joint": 0.01,
            #     "Neck_pitch_joint": 0.01,
            # },
            stiffness={
                "waist_yaw_joint": 187.0,
                "Neck_yaw_joint": 6.7,
                "Neck_pitch_joint": 6.7,
            },
            damping={
                "waist_yaw_joint": 18.7,
                "Neck_yaw_joint": 0.67,
                "Neck_pitch_joint": 0.67,
            },
            armature={
                "waist_yaw_joint": 0.03,
                "Neck_yaw_joint": 0.01,
                "Neck_pitch_joint": 0.01,
            },
        ),
    },
)

K1_ACTION_SCALE = {}
for a in K1L_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            K1_ACTION_SCALE[n] = 0.25 #* e[n] / s[n]
