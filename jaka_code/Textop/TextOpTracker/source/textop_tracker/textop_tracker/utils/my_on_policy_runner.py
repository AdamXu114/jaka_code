import os
import pickle
from typing import Optional

import torch
import numpy as np
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from textop_tracker.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

# class MyOnPolicyRunner(OnPolicyRunner):
#     def save(self, path: str, infos=None):
#         """Save the model and training information."""
#         super().save(path, infos)
#         if self.logger_type in ["wandb"]:
#             policy_path = path.split("model")[0]
#             filename = policy_path.split("/")[-2] + ".onnx"
#             export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
#             attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
#             wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device="cpu",
        registry_name: Optional[str] = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        
        unwrapped_env = self.env.unwrapped
        if True:# hasattr(unwrapped_env, "command_manager") and "motion" in unwrapped_env.command_manager.active_terms:
            num_motions = unwrapped_env.command_manager.get_term("motion").num_motion
            self.action_rate_for_all_motion = torch.zeros((num_motions, 1), dtype=torch.float32, device=self.device)
            self.action_rate_threshold = 2000.0
            self.filtered_transition_count = 0
            self.filtered_motion_count = torch.zeros(num_motions, dtype=torch.long, device=self.device)
        else:
            self.action_rate_for_all_motion = None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        original_step = self.env.step

        def step_wrapper(actions):
            unwrapped_env = self.env.unwrapped
            ##new
            clamped_actions = torch.clamp(actions, -10.0, 10.0)
            # Also clamp stored transition.actions so PPO update uses bounded actions
            self.alg.transition.actions.clamp_(-10.0, 10.0)
            ##new
            
            bad_mask = None
            if self.action_rate_for_all_motion is not None and hasattr(unwrapped_env, "action_manager") and hasattr(unwrapped_env.action_manager, "action"):
                # The actions passed here are a_t. The previously applied action is unwrapped_env.action_manager.action (a_{t-1}).
                prev_actions = unwrapped_env.action_manager.action
               # action_rate = torch.sum(torch.square(actions - prev_actions), dim=1)
                action_rate = torch.sum(torch.square(clamped_actions - prev_actions), dim=1)


                motion_ids = unwrapped_env.command_manager.get_term("motion").motion_idx

                current_max = self.action_rate_for_all_motion[motion_ids, 0]
                self.action_rate_for_all_motion[motion_ids, 0] = torch.maximum(current_max, action_rate)

                # Detect extreme action_rate
                bad_mask = action_rate > self.action_rate_threshold

            # Execute the original env step
            #obs, rewards, dones, infos = original_step(actions)
            # Execute the original env step with CLAMPED actions
            obs, rewards, dones, infos = original_step(clamped_actions)


            # Filter toxic transitions: neutralize rewards/dones/values to prevent gradient explosion
            if bad_mask is not None and bad_mask.any():
                self.filtered_transition_count += bad_mask.sum().item()
                bad_motion_ids = motion_ids[bad_mask]
                self.filtered_motion_count.scatter_add_(
                    0, bad_motion_ids, torch.ones_like(bad_motion_ids)
                )
                rewards[bad_mask] = 0.0
                dones[bad_mask] = 1
                # Zero out value and log_prob so this transition contributes ~0 to PPO loss
                self.alg.transition.values[bad_mask] = 0.0
                # self.alg.transition.actions_log_prob[bad_mask] = 0.0

            return obs, rewards, dones, infos

        self.env.step = step_wrapper
        super().learn(num_learning_iterations, init_at_random_ep_len)
        self.env.step = original_step

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width, pad)
        
        if self.current_learning_iteration % 100 == 0:
            if hasattr(self, "action_rate_for_all_motion") and self.action_rate_for_all_motion is not None:
                if self.log_dir is not None:
                    # Build filtered motion info for this 100-iteration window
                    filtered_motion_info = []
                    nonzero_ids = torch.nonzero(self.filtered_motion_count, as_tuple=False).squeeze(-1)
                    if nonzero_ids.numel() > 0:
                        unwrapped_env = self.env.unwrapped
                        motion_files = unwrapped_env.command_manager.get_term("motion").cfg.motion_files
                        for mid in nonzero_ids.cpu().numpy():
                            filtered_motion_info.append({
                                "motion_id": int(mid),
                                "motion_name": os.path.basename(os.path.dirname(motion_files[int(mid)])),
                                "count": int(self.filtered_motion_count[mid].item()),
                            })

                    save_data = {
                        "action_rate_for_all_motion": self.action_rate_for_all_motion.cpu().numpy(),
                        "filtered_transition_count": self.filtered_transition_count,
                        "filtered_motion_info": filtered_motion_info,
                    }
                    save_path = os.path.join(self.log_dir, f"model_{self.current_learning_iteration}-action_rate_for_all_motion.pkl")
                    pickle.dump(save_data, open(save_path, "wb"))

                    # Reset counters for the next 100-iteration window
                    self.filtered_transition_count = 0
                    self.filtered_motion_count.zero_()

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        unwrapped_env = self.env.unwrapped
 
        policy_path = path.split("model")[0]
        onnx_filename = "latest.onnx"
        export_motion_policy_as_onnx(
            unwrapped_env, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=onnx_filename
        )
        if self.logger_type in ["wandb"]:
            attach_onnx_metadata(unwrapped_env, wandb.run.name, path=policy_path, filename=onnx_filename)
            wandb.save(policy_path + onnx_filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

        # For DEBUG:
        if unwrapped_env.command_manager.get_term("motion").cfg.enable_adaptive_sampling:
            fail_count = unwrapped_env.command_manager.get_term("motion").failed_motion_count.cpu().numpy()
            success_count = unwrapped_env.command_manager.get_term("motion").success_motion_count.cpu().numpy()
            total_count = fail_count + success_count
            p_fail = fail_count / (total_count + 1e-8)
            p_fail_sample_v2 = (p_fail**unwrapped_env.command_manager.get_term("motion").cfg.adaptive_beta)
            p_fail_sample_v2 = p_fail_sample_v2 / (p_fail_sample_v2.sum() + 1e-8)
            sampling_probabilities_v2 = (
                p_fail_sample_v2 * (1 - unwrapped_env.command_manager.get_term("motion").cfg.adaptive_uniform_ratio) +
                unwrapped_env.command_manager.get_term("motion").cfg.adaptive_uniform_ratio /
                float(unwrapped_env.command_manager.get_term("motion").num_motion)
            )
            adpsam_count = {
                "failed_motion_count": fail_count,
                "success_motion_count": success_count,
                "p_fail": p_fail,
                "p_fail_sample_v2": p_fail_sample_v2,
                "sampling_probabilities_v2": sampling_probabilities_v2,
            }
            # (adpsam_count, step=self.current_learning_iteration)
            pickle.dump(adpsam_count, open(path[:-len(".pt")] + "-adpsam_count.pkl", "wb"))


# import os
# import pickle
# from typing import Optional

# import torch
# import numpy as np
# from rsl_rl.env import VecEnv
# from rsl_rl.runners.on_policy_runner import OnPolicyRunner

# from isaaclab_rl.rsl_rl import export_policy_as_onnx

# import wandb
# from textop_tracker.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

# # class MyOnPolicyRunner(OnPolicyRunner):
# #     def save(self, path: str, infos=None):
# #         """Save the model and training information."""
# #         super().save(path, infos)
# #         if self.logger_type in ["wandb"]:
# #             policy_path = path.split("model")[0]
# #             filename = policy_path.split("/")[-2] + ".onnx"
# #             export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
# #             attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
# #             wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


# class MotionOnPolicyRunner(OnPolicyRunner):
#     def __init__(
#         self,
#         env: VecEnv,
#         train_cfg: dict,
#         log_dir: str | None = None,
#         device="cpu",
#         registry_name: Optional[str] = None
#     ):
#         super().__init__(env, train_cfg, log_dir, device)
#         self.registry_name = registry_name
        
#         unwrapped_env = self.env.unwrapped
#         num_motions = unwrapped_env.command_manager.get_term("motion").num_motion
#         self.action_rate_for_all_motion = torch.zeros((num_motions, 1), dtype=torch.float32, device=self.device)
#         # if hasattr(unwrapped_env, "command_manager") and "motion" in unwrapped_env.command_manager.active_terms:
#         #     num_motions = unwrapped_env.command_manager.get_term("motion").num_motion
#         #     self.action_rate_for_all_motion = torch.zeros((num_motions, 1), dtype=torch.float32, device=self.device)
#         # else:
#         #     self.action_rate_for_all_motion = None

#     def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
#         original_step = self.env.step

#         def step_wrapper(actions):
#             unwrapped_env = self.env.unwrapped
#             if self.action_rate_for_all_motion is not None and hasattr(unwrapped_env, "action_manager") and hasattr(unwrapped_env.action_manager, "action"):
#                 # The actions passed here are a_t. The previously applied action is unwrapped_env.action_manager.action (a_{t-1}).
#                 prev_actions = unwrapped_env.action_manager.action
#                 action_rate = torch.sum(torch.square(actions - prev_actions), dim=1)

#                 motion_ids = unwrapped_env.command_manager.get_term("motion").motion_idx

#                 current_max = self.action_rate_for_all_motion[motion_ids, 0]
#                 self.action_rate_for_all_motion[motion_ids, 0] = torch.maximum(current_max, action_rate)

#             return original_step(actions)

#         self.env.step = step_wrapper
#         super().learn(num_learning_iterations, init_at_random_ep_len)
#         self.env.step = original_step

#     def log(self, locs: dict, width: int = 80, pad: int = 35):
#         super().log(locs, width, pad)
        
#         if self.current_learning_iteration % 100 == 0:
#             if hasattr(self, "action_rate_for_all_motion") and self.action_rate_for_all_motion is not None:
#                 save_path = os.path.join(self.log_dir, f"model_{self.current_learning_iteration}-action_rate_for_all_motion.pkl")
#                 pickle.dump(self.action_rate_for_all_motion.cpu().numpy(), open(save_path, "wb"))

#     def save(self, path: str, infos=None):
#         """Save the model and training information."""
#         super().save(path, infos)
#         unwrapped_env = self.env.unwrapped

#         policy_path = path.split("model")[0]
#         onnx_filename = "latest.onnx"
#         export_motion_policy_as_onnx(
#             unwrapped_env, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=onnx_filename
#         )
#         if self.logger_type in ["wandb"]:
#             attach_onnx_metadata(unwrapped_env, wandb.run.name, path=policy_path, filename=onnx_filename)
#             wandb.save(policy_path + onnx_filename, base_path=os.path.dirname(policy_path))

#             # link the artifact registry to this run
#             if self.registry_name is not None:
#                 wandb.run.use_artifact(self.registry_name)
#                 self.registry_name = None

#         # For DEBUG:
#         if unwrapped_env.command_manager.get_term("motion").cfg.enable_adaptive_sampling:
#             fail_count = unwrapped_env.command_manager.get_term("motion").failed_motion_count.cpu().numpy()
#             success_count = unwrapped_env.command_manager.get_term("motion").success_motion_count.cpu().numpy()
#             total_count = fail_count + success_count
#             p_fail = fail_count / (total_count + 1e-8)
#             p_fail_sample_v2 = (p_fail**unwrapped_env.command_manager.get_term("motion").cfg.adaptive_beta)
#             p_fail_sample_v2 = p_fail_sample_v2 / (p_fail_sample_v2.sum() + 1e-8)
#             sampling_probabilities_v2 = (
#                 p_fail_sample_v2 * (1 - unwrapped_env.command_manager.get_term("motion").cfg.adaptive_uniform_ratio) +
#                 unwrapped_env.command_manager.get_term("motion").cfg.adaptive_uniform_ratio /
#                 float(unwrapped_env.command_manager.get_term("motion").num_motion)
#             )
#             adpsam_count = {
#                 "failed_motion_count": fail_count,
#                 "success_motion_count": success_count,
#                 "p_fail": p_fail,
#                 "p_fail_sample_v2": p_fail_sample_v2,
#                 "sampling_probabilities_v2": sampling_probabilities_v2,
#             }
#             # (adpsam_count, step=self.current_learning_iteration)
#             pickle.dump(adpsam_count, open(path[:-len(".pt")] + "-adpsam_count.pkl", "wb"))

# import os
# import pickle
# from typing import Optional

# import numpy as np
# from rsl_rl.env import VecEnv
# from rsl_rl.runners.on_policy_runner import OnPolicyRunner

# from isaaclab_rl.rsl_rl import export_policy_as_onnx

# import wandb
# from textop_tracker.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

# # class MyOnPolicyRunner(OnPolicyRunner):
# #     def save(self, path: str, infos=None):
# #         """Save the model and training information."""
# #         super().save(path, infos)
# #         if self.logger_type in ["wandb"]:
# #             policy_path = path.split("model")[0]
# #             filename = policy_path.split("/")[-2] + ".onnx"
# #             export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
# #             attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
# #             wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


# class MotionOnPolicyRunner(OnPolicyRunner):
#     def __init__(
#         self,
#         env: VecEnv,
#         train_cfg: dict,
#         log_dir: str | None = None,
#         device="cpu",
#         registry_name: Optional[str] = None
#     ):
#         super().__init__(env, train_cfg, log_dir, device)
#         self.registry_name = registry_name

#     def save(self, path: str, infos=None):
#         """Save the model and training information."""
#         super().save(path, infos)
#         unwrapped_env = self.env.unwrapped

#         policy_path = path.split("model")[0]
#         onnx_filename = "latest.onnx"
#         export_motion_policy_as_onnx(
#             unwrapped_env, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=onnx_filename
#         )
#         if self.logger_type in ["wandb"]:
#             attach_onnx_metadata(unwrapped_env, wandb.run.name, path=policy_path, filename=onnx_filename)
#             wandb.save(policy_path + onnx_filename, base_path=os.path.dirname(policy_path))

#             # link the artifact registry to this run
#             if self.registry_name is not None:
#                 wandb.run.use_artifact(self.registry_name)
#                 self.registry_name = None

#         # For DEBUG:
#         if unwrapped_env.command_manager.get_term("motion").cfg.enable_adaptive_sampling:
#             fail_count = unwrapped_env.command_manager.get_term("motion").failed_motion_count.cpu().numpy()
#             success_count = unwrapped_env.command_manager.get_term("motion").success_motion_count.cpu().numpy()
#             total_count = fail_count + success_count
#             p_fail = fail_count / (total_count + 1e-8)
#             p_fail_sample_v2 = (p_fail**unwrapped_env.command_manager.get_term("motion").cfg.adaptive_beta)
#             p_fail_sample_v2 = p_fail_sample_v2 / (p_fail_sample_v2.sum() + 1e-8)
#             sampling_probabilities_v2 = (
#                 p_fail_sample_v2 * (1 - unwrapped_env.command_manager.get_term("motion").cfg.adaptive_uniform_ratio) +
#                 unwrapped_env.command_manager.get_term("motion").cfg.adaptive_uniform_ratio /
#                 float(unwrapped_env.command_manager.get_term("motion").num_motion)
#             )
#             adpsam_count = {
#                 "failed_motion_count": fail_count,
#                 "success_motion_count": success_count,
#                 "p_fail": p_fail,
#                 "p_fail_sample_v2": p_fail_sample_v2,
#                 "sampling_probabilities_v2": sampling_probabilities_v2,
#             }
#             # (adpsam_count, step=self.current_learning_iteration)
#             pickle.dump(adpsam_count, open(path[:-len(".pt")] + "-adpsam_count.pkl", "wb"))
