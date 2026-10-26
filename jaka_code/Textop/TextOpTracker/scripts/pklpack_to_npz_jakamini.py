"""This script unpacks a large pkl file containing multiple motions and converts each to individual motion.npz files.

.. code-block:: bash

    # Usage
    python pklpack_to_npz.py --input_file /path/to/aaa.pkl \
    --output_dir ./artifacts/unpacked_motions --input_fps 30 --output_fps 50
"""

from tqdm import tqdm

import os
import argparse
import numpy as np
import joblib
from pathlib import Path
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Unpack pkl file with multiple motions and convert each to npz file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion pkl file.")
parser.add_argument("--output_dir", type=str, required=True, help="The directory to save individual motion npz files.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument("--max_motions", type=int, default=None, help="Maximum number of motions to process (for testing).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from textop_tracker.robots.K1L import K1L_CYLINDER_CFG,Khan_mini_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = Khan_mini_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class PKLMotionLoader:
    def __init__(
        self,
        motion_data: dict,
        input_fps: int,
        output_fps: int,
        device: torch.device,
    ):
        self.motion_data = motion_data
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the motion data dict."""
        # assert self.motion_data["fps"] == self.input_fps, f"Input FPS Mismatch!, {self.motion_data['fps']} != {self.input_fps}"
        self.real_fps_input=self.motion_data["fps"]
        self.motion_base_poss_input = torch.from_numpy(self.motion_data["root_pos"][2:]).to(torch.float32).to(self.device)
        self.motion_base_poss_input[...,2]+=0.025#0.03#0.035
        self.motion_base_rots_input = torch.from_numpy(self.motion_data["root_rot"][2:]).to(torch.float32).to(self.device)
        # motion_data['root_rot']: XYZW

        #self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]] # convert xyzw to wxyz

        num_dof = self.motion_data["dof_pos"].shape[1]
        self.motion_dof_poss_input = torch.from_numpy(self.motion_data["dof_pos"][2:]).to(torch.float32).to(self.device)
        #print(self.motion_dof_poss_input[:3])
        
        # self.motion_dof_poss_input = torch.cat(
        #             # [
        #             #     self.motion_dof_poss_input[:, :18],
        #             #     torch.zeros_like(self.motion_dof_poss_input[:, :1]),
        #             #     self.motion_dof_poss_input[:, 18:23],
        #             #     torch.zeros_like(self.motion_dof_poss_input[:, :3]),
        #             # ],
        #             # dim=1,
        #              [
        #                 self.motion_dof_poss_input[:, :18],
        #                 torch.ones_like(self.motion_dof_poss_input[:, :1])*0.3,
        #                 self.motion_dof_poss_input[:, 18:23],
        #                 torch.ones_like(self.motion_dof_poss_input[:, :1])*0.3,
        #                 torch.zeros_like(self.motion_dof_poss_input[:, :2]),
        #             ],
        #             dim=1,
        #         )

        # if num_dof == 29:
        #     # Everything good
        #     self.motion_dof_poss_input = torch.cat(
        #         [
        #             self.motion_dof_poss_input[:, :19],
        #             torch.zeros_like(self.motion_dof_poss_input[:, :3]),
        #             self.motion_dof_poss_input[:, 22:26],
        #             torch.zeros_like(self.motion_dof_poss_input[:, :3]),
        #         ],
        #         dim=1,
        #     )
        #     pass
        # elif num_dof == 23:
        #     # 29 = 15 + ( 4 + __3__)*2
        #     #    = 19 + 3 + 4 + 3
        #     self.motion_dof_poss_input = torch.cat(
        #         [
        #             self.motion_dof_poss_input[:, :19],
        #             torch.zeros_like(self.motion_dof_poss_input[:, :3]),
        #             self.motion_dof_poss_input[:, 19:23],
        #             torch.zeros_like(self.motion_dof_poss_input[:, :3]),
        #         ],
        #         dim=1,
        #     )
        #     assert self.motion_dof_poss_input.shape[1] == 29
        #
        # else:
        #     raise NotImplementedError(f"Unsupported DOF count: {num_dof}")

        self.input_frames = self.motion_dof_poss_input.shape[0]
        self.duration = (self.input_frames) * self.input_dt
        print(f"Motion loaded, duration: {self.duration} sec, frames: {self.input_frames},real_fps:{self.real_fps_input}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i].item())
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        bool,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str], motion_data: dict, motion_name: str,idx):
    """Runs the simulation loop for a single motion."""
    # Load motion
    motion = PKLMotionLoader(
        motion_data=motion_data,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------
    time_steps = 0
    height_offset = None
    apply_offset = False
    z_left = 0.0
    z_right = 0.0
    z_left_list=[]
    z_right_list=[]

    # 此时 robot 已经被写成了 motion 第 0 帧
    ankle_names = ["Left_ankle_roll_Link", "Right_ankle_roll_Link"]
    ankle_indices = robot.find_bodies(ankle_names, preserve_order=True)[0]
    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        # if height_offset is not None:
        #     root_states[:, 2] += height_offset
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        # sim.render()  # We don't want physic (sim.step())
        # scene.update(sim.get_physics_dt())
        #
        # pos_lookat = root_states[0, :3].cpu().numpy()
        # sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)
        # if time_steps == 1:
        #     body_pos_w = robot.data.body_pos_w[0, ankle_indices]  # (2, 3)
        #     z_left = body_pos_w[0, 2].item()
        #     z_right = body_pos_w[1, 2].item()
        #     if abs(z_left - z_right) < 0.1:
        #         z_min = min(z_left, z_right)
        #         height_offset = 0.08 - z_min
        #         apply_offset = True
        #     else:
        #         height_offset = 0.0
        #         apply_offset = False

        body_pos_w = robot.data.body_pos_w[0, ankle_indices]  # (2, 3)
        z_left = body_pos_w[0, 2].item()
        z_right = body_pos_w[1, 2].item()
        z_left_list.append(z_left)
        z_right_list.append(z_right)

        if not file_saved :#and time_steps >= 2:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)
                
            #### not for self record pico #####
            z_left_min=np.min(z_left_list)
            z_right_min=np.min(z_right_list)
            log["body_pos_w"][...,2]+=0.035-min(z_left_min,z_right_min)
            
            
            
            # Create safe filename from motion name
            # safe_motion_name = "".join(c for c in str(motion_name) if c.isalnum() or c in ("-", "_")).rstrip()
            # safe_motion_name = safe_motion_name.replace(" ", "_")
            # if len(safe_motion_name) > 100:  # Limit filename length
            #     safe_motion_name = safe_motion_name[:100]

            safe_motion_name=motion_name+"_low"#f"Uaenet_{int(idx):04d}"

            save_path = f"{args_cli.output_dir}/{safe_motion_name}/motion.npz"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, **log)
            print(f"Saved motion to: {save_path}")
            break


def main():
    """Main function."""
    # Create output directory
    import sys
    # import numpy as np
    # try:
    #     import numpy._core
    # except ImportError:
    #     # 只有当真的没有 _core 时才打补丁
    #     sys.modules['numpy._core'] = np.core
    #     from numpy import core
    #     sys.modules['numpy._core.multiarray'] = core.multiarray
    os.makedirs(args_cli.output_dir, exist_ok=True)

    # Load the pkl file
    # import pickle
    # print(f"Loading pkl file: {args_cli.input_file}")
    motions_data = joblib.load(args_cli.input_file)
    # with open(args_cli.input_file, "rb") as f:
    #     motions_data = pickle.load(f)
    print(f"Loaded {len(motions_data)} motions")

    # Limit number of motions if specified
    if args_cli.max_motions:
        motions_data = dict(list(motions_data.items())[: args_cli.max_motions])
        print(f"Limited to {len(motions_data)} motions for processing")

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Process each motion
    joint_names = [
        "Left_hip_pitch_joint", #left leg (6 dof)
        "Left_hip_roll_joint",
        "Left_hip_yaw_joint",
        "Left_knee_joint",
        "Left_ankle_pitch_joint",
        "Left_ankle_roll_joint",

        "Right_hip_pitch_joint", #right leg (6dof)
        "Right_hip_roll_joint",
        "Right_hip_yaw_joint",
        "Right_knee_joint",
        "Right_ankle_pitch_joint",
        "Right_ankle_roll_joint",

        "waist_yaw_joint",
        
        "Left_shoulder_pitch_joint", #left arm (6 dof)
        "Left_shoulder_roll_joint",
        "Left_shoulder_yaw_joint",
        "Left_elbow_joint",
        "Left_wrist_roll_joint",
        "Left_wrist_yaw_joint",

        "Right_shoulder_pitch_joint", #right arm (6 dof)
        "Right_shoulder_roll_joint",
        "Right_shoulder_yaw_joint",
        "Right_elbow_joint",
        "Right_wrist_roll_joint",
        "Right_wrist_yaw_joint",

       

        "Neck_yaw_joint",  #neck (2 dof)
        "Neck_pitch_joint",
    ]

    processed_count = 0

    motions_list = list(motions_data.items())
    for idx, (motion_name, motion_data) in enumerate(tqdm(motions_list, desc="Processing motions", unit="motion")):
        print(f"\nProcessing motion {idx + 1}/{len(motions_list)}: {motion_name}")
        # try:
        # if not "Uaenet" in str(motion_name):# "yanjie" in str(motion_name) or "0811" in str(motion_name) or "00_" in str(motion_name): # or "Uaenet" in str(motion_name) or "handball" in str(motion_name):
        #     continue

        if True:
            # Run the simulator for this motion
            run_simulator(
                sim,
                scene,
                joint_names,
                motion_data,
                motion_name,
                idx
            )
            processed_count += 1
        # except Exception as e:
        #     print(f"Error processing motion {motion_name}: {e}")
        #     continue
        # if idx>=2000:
        #     break
    print(f"\nCompleted processing {processed_count} motions")


if __name__ == "__main__":
    # run the main function
    main()
    os._exit(0)  # type: ignore
