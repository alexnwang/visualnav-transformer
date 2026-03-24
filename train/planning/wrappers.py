from functools import partial
import os
from pprint import pprint
import torch
import copy
from torchvision import transforms
from dreamsim import dreamsim
import numpy as np
from PIL import Image, ImageDraw
from planning.plotting_fns import save_action_obs_sequence_viz
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.training.nymeria_training_utils import get_action_smpl_torch
from planning.utils import _compute_pose_and_loss, _compute_part_distance_matrices
from planning.sampling import peva_sample, waypoint_sample

def build_skeleton_top_seq(curr_obs_img, pred_deltas, first_pose, xsens_offsets,
                           cam_model, T_C_pelvis, image_size, T):
    """
    Build a (T, 3, H, W) sequence of curr_obs with skeleton overlays per timestep.
    If cam_model/T_C_pelvis are None, returns blank (zero) frames.

    Args:
        curr_obs_img: (3, H, W) current observation image
        pred_deltas:  (1, T, action_dim) action deltas (batch-of-1)
        first_pose:   (1, 1, action_dim) starting pose (batch-of-1)
        xsens_offsets: (15, 3) skeleton joint offsets
        cam_model:    camera calibration, or None to skip
        T_C_pelvis:   SE3 pelvis→camera transform, or None to skip
        image_size:   int, spatial size of images
        T:            number of timesteps
    Returns:
        top_seq: (T, 3, H, W) tensor
    """
    device = curr_obs_img.device
    if cam_model is not None and T_C_pelvis is not None:
        from planning.vis_utils import pose_to_image_coords, draw_image_coords as draw_skel
        from torchvision import transforms as T_transforms

        pred_actions = get_action_smpl_torch(
            first_pose, pred_deltas, XSensConstants.upper_body_num_parts
        )  # (1, T, 48)

        skel_tensors = []
        for t in range(pred_actions.shape[1]):
            img_pil = Image.fromarray(
                (255.0 * curr_obs_img.permute(1, 2, 0)).to(torch.uint8).numpy()
            )
            draw = ImageDraw.Draw(img_pil)
            image_coords = pose_to_image_coords(
                pred_actions[:, t], cam_model, xsens_offsets, T_C_pelvis, image_size=image_size
            )  # (1, 15, 2)
            draw_skel(draw, image_coords)
            skel_tensors.append(T_transforms.ToTensor()(img_pil))
        return torch.stack(skel_tensors).to(device)  # (T, 3, H, W)
    else:
        return torch.zeros(T, *curr_obs_img.shape).to(device)


def save_mu_step_results(mu_rollout_state, state_0, state_g, cam_model, T_C_pelvis, save_path):
    """
    Save per-iteration mu evaluation artifacts:
      - action_obs_seq.png: 2-row visualization
          top: [policy input goal | curr_obs+skel t=1 | ... | curr_obs+skel t=T | blank]
               policy input goal = curr_obs+predicted waypoints (waypoint CEM) or plain curr_obs (peva)
               skeleton overlays are blank if cam_model/T_C_pelvis not provided
          bot: [curr_obs | generated_obs t=1 | ... | generated_obs t=T | goal_obs]

    Args:
        mu_rollout_state: output of wm.rollout(state_0, mu) — generated_obs, deltas, goal_images
        state_0: transformed initial observation dict (trans_obs_0)
        state_g: encoded goal state dict (z_obs_g)
        cam_model: camera calibration (from NymeriaDataProvider), or None to skip skeleton vis
        T_C_pelvis: SE3 transform pelvis→camera at curr_time, or None to skip skeleton vis
        save_path: directory to write artifacts into
    """
    os.makedirs(save_path, exist_ok=True)

    generated_obs = mu_rollout_state["generated_obs"]  # (1, T, 3, H, W)
    curr_obs_img  = state_0["images"][0, -1]            # (3, H, W)
    goal_obs      = state_g["images"][0]                # (3, H, W)
    goal_images   = mu_rollout_state.get("goal_images") # (1, W, 3, H, W) or None

    # Skeleton overlays from predicted deltas (SMPL actions in both CEM variants)
    T = generated_obs.shape[1]
    pred_deltas   = mu_rollout_state["deltas"]   # (1, T, 48)
    first_pose    = state_g["first_pose"]        # (1, 1, 48)
    xsens_offsets = state_g["xsens_offsets"][0]  # (15, 3)
    image_size    = state_0["images"].shape[-1]
    top_seq = build_skeleton_top_seq(
        curr_obs_img, pred_deltas, first_pose, xsens_offsets,
        cam_model, T_C_pelvis, image_size, T
    )

    # top-left: curr_obs + predicted waypoints (waypoint CEM) or plain curr_obs (peva)
    policy_input_goal = goal_images[0, 0] if goal_images is not None else curr_obs_img
    save_action_obs_sequence_viz(
        save_path=os.path.join(save_path, "action_obs_seq.png"),
        goal_image=policy_input_goal,
        curr_obs=curr_obs_img,
        goal_obs=goal_obs,
        top_seq=top_seq,
        bot_seq=generated_obs[0],           # (T, 3, H, W)
    )


class Preprocessor:
    def __init__(self, transform=torch.nn.Identity()):
        self.transform = transform

    def transform_obs(self, obs):
        res = {}
        for key in obs:
            if key == "images":
                res[key] = self.transform(obs[key])
            elif key == "goal_image":
                res[key] = self.transform(obs[key])
            else:
                res[key] = obs[key]
        return res

class WMWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def encode_obs(self, obs):
        return copy.deepcopy(obs)

    def rollout(self, *args, **kwargs):
        raise NotImplementedError

class PevaWM(WMWrapper):
    def __init__(self, peva_model, peva_diffusion, peva_vae, peva_stats,
                 image_size, peva_context_size):
        super().__init__()
        self.peva_model = peva_model
        self.peva_diffusion = peva_diffusion
        self.peva_vae = peva_vae
        self.peva_stats = peva_stats
        
        self.image_size = image_size
        self.latent_size = image_size // 8
        self.peva_context_size = peva_context_size
        
        self.sample_fn = partial(peva_sample, 
                                 peva_model=self.peva_model, peva_diffusion=self.peva_diffusion, peva_vae=self.peva_vae, peva_stats=self.peva_stats,
                                 image_size=self.image_size, peva_context_size=self.peva_context_size, peva_latent_size=self.latent_size)
    
    def rollout(self, state_0, act):
        """
        state_0: B, peva_context_size, 3, H, W
        act: B, T, action_dim
        """
        device = act.device 
        curr_obs = state_0['images'] # B, peva_context_size, 3, H, W
        
        generated_frames, deltas = self.sample_fn(curr_obs=curr_obs, deltas=act, device=device)
        
        return {"generated_obs": generated_frames.to(torch.float32),
                "deltas": deltas.to(torch.float32),
                "goal_images": None}

class WaypointWM(WMWrapper):
    def __init__(self,peva_model, peva_diffusion, peva_vae, peva_stats,
                 policy_model, policy_diffusion,
                 image_size, peva_context_size, 
                 policy_context_size, policy_pred_horizon, policy_action_dim,
                 skip_last_peva=False):
        super().__init__()
        self.peva_model = peva_model
        self.peva_diffusion = peva_diffusion
        self.peva_vae = peva_vae
        self.peva_stats = peva_stats
        
        self.policy_model = policy_model
        self.policy_diffusion = policy_diffusion
        
        self.image_size = image_size
        self.latent_size = image_size // 8
        self.peva_context_size = peva_context_size
        self.policy_context_size = policy_context_size
        
        self.policy_pred_horizon = policy_pred_horizon
        self.policy_action_dim = policy_action_dim
        
        self.skip_last_peva = skip_last_peva
        
        self.sample_fn = partial(
            waypoint_sample,
            policy_model=policy_model, policy_diffusion=policy_diffusion,
            peva_model=peva_model, peva_diffusion=peva_diffusion, peva_vae=peva_vae, peva_stats=peva_stats,
            policy_pred_horizon=policy_pred_horizon, policy_action_dim=policy_action_dim,
            image_size=image_size, 
            policy_context_size=policy_context_size, peva_context_size=peva_context_size, peva_latent_size=self.latent_size,
            skip_last_peva=skip_last_peva
        )
    
    def rollout(self, state_0, act):
        device = act.device
        curr_obs = state_0['images'] # B, peva_context_size, 3, H, W
        goal_obs = state_0['goal_image'] # B, 3, H, W
        context_poses = state_0['context_poses'] # B, peva_context_size, 48
        
        waypoints = act * self.image_size
        
        (
            generated_frames, # B, W, policy_pred_horizon, 3, H, W
            policy_sampled_deltas, # B, W, policy_pred_horizon, policy_action_dim
            waypoint_annotated_goals # B, W, 3, H, W
        ) = waypoint_sample(self.policy_model, self.policy_diffusion,
                    self.peva_model, self.peva_diffusion, self.peva_vae, self.peva_stats,
                    waypoints, context_poses, curr_obs, goal_obs,
                    self.policy_pred_horizon, self.policy_action_dim,
                    self.image_size, 
                    self.policy_context_size, self.peva_context_size, self.latent_size,
                    device,
                    skip_last_peva=self.skip_last_peva)
        
        return {"generated_obs": generated_frames.to(torch.float32).flatten(1,2),
                "deltas": policy_sampled_deltas.to(torch.float32).flatten(1,2),
                "goal_images": waypoint_annotated_goals.to(torch.float32)}

class ObjectiveDreamSIM:
    def __init__(self, pred_horizon, device,
                 return_metric=False,):
        """
        Args:
            pred_horizon (int): the prediction horizon
            device (str): the device to use
            return_metric (bool): whether to return the metric rather than DreamSIM distance. This is a cheat method.
        """
        self.pred_horizon = pred_horizon
        self.device = device
        self.model, self.preprocess = dreamsim(pretrained=True, device=device, cache_dir="/scratch/anw2067/cache")
        self.save_dir = None
        self.return_metric = return_metric

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
    
    def __call__(self, cem_step, rollout_state, state_0, goal_state, save_path=None, topk=0,
                 cam_model=None, T_C_pelvis=None):
        first_pose = goal_state["first_pose"] # B, 1, policy_action_dim
        deltas = rollout_state["deltas"] # B, W*policy_pred_horizon, policy_action_dim
        gt_deltas = goal_state["deltas"] # B, T, policy_action_dim
        
        curr_obs = state_0["images"] # B, peva_context_size, 3, H, W
        gt_goal_image = state_0["goal_image"] # B, 3, H, W
        generated_obs = rollout_state["generated_obs"] # B, W*policy_pred_horizon, 3, H, W
        rollout_waypoint_annotated_obs = rollout_state["goal_images"] # B, W, 3, H, W or None 
        
        xsens_offsets = goal_state["xsens_offsets"][0]
        skel = XsensSkeleton(xsens_offsets)
        
        pred_actions = get_action_smpl_torch(first_pose, deltas, XSensConstants.upper_body_num_parts) # B, T, 48
        gt_actions = get_action_smpl_torch(first_pose, gt_deltas, XSensConstants.upper_body_num_parts) # B, T, 48
        
        (xyz_dist_matrix, ang_dist_matrix, leaf_xyz, leaf_ang) = _compute_part_distance_matrices(pred_actions[:, -1], gt_actions[:, -1], skel)
        _, _, leaf_xyz_init, leaf_ang_init = _compute_part_distance_matrices(first_pose[:, -1], gt_actions[:, -1], skel)
        
        goal_obs = goal_state["images"] # B, 3, H, W
        
        B = generated_obs.shape[0]
        res = torch.empty(B, device=self.device)
        for i in range(B):
            pred_image_pil = transforms.ToPILImage()(generated_obs[i, -1])
            goal_image_pil = transforms.ToPILImage()(goal_obs[i])
            
            rollout_state = self.preprocess(pred_image_pil).to(self.device)
            goal_state = self.preprocess(goal_image_pil).to(self.device)

            sim = self.model(rollout_state, goal_state)
            res[i] = sim
        if save_path is not None and topk > 0:
            elite_idx = torch.argsort(res)[:topk]
            T = generated_obs.shape[1]
            for rank, i in enumerate(elite_idx):
                i = i.item()
                image_size = curr_obs.shape[-1]
                top_seq_i = build_skeleton_top_seq(
                    curr_obs[i, -1], deltas[i:i+1], first_pose[i:i+1], xsens_offsets,
                    cam_model, T_C_pelvis, image_size, T
                )
                policy_input_goal_i = rollout_waypoint_annotated_obs[i, 0] if rollout_waypoint_annotated_obs is not None else curr_obs[i, -1]
                save_action_obs_sequence_viz(
                    save_path=f"{save_path}/elite{rank}-leaf_xyz{np.round(leaf_xyz[i].item(), decimals=3)}.png",
                    goal_image=policy_input_goal_i,
                    curr_obs=curr_obs[i, -1],
                    goal_obs=goal_obs[i],
                    top_seq=top_seq_i,
                    bot_seq=generated_obs[i],
                )
            
        print(f"ObjectiveFn: {res.mean().item()}")
        if self.return_metric:
            return leaf_xyz, {"loss": res.mean().item(),
                              "objective-xyz_distance": leaf_xyz.mean().item(),
                              "objective-start_xyz_distance": leaf_xyz_init.mean().item()}
        else:
            return res, {"loss": res.mean().item(),
                         "xyz_distance": leaf_xyz.mean().item(),
                         "start_xyz_distance": leaf_xyz_init.mean().item(),
                         "angular_distance": leaf_ang.mean().item()}
    
class EvaluatorPeva(PevaWM):
    def __init__(self,
                 *args,
                 num_eval_samples=1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_eval_samples = num_eval_samples
    
    def eval_actions(self, cem_step, actions_mu, state_0, state_g):
        """
        actions_mu: B, T, action_dim
        gt_dict: dict of gt_actions, skel
        """
        B = actions_mu.shape[0]

        goal_image_coords = state_g['goal_image_coords'] # B, 23, 2
        xsens_offsets = state_g["xsens_offsets"][0]
        skel = XsensSkeleton(xsens_offsets)

        num_joints_visible = (goal_image_coords[:, XSensConstants.leaf_indices] != -1).all(dim=-1).float().sum(dim=-1)

        deltas_gt = state_g["deltas"]
        first_pose = state_g["first_pose"] # B, 1, 48

        if self.num_eval_samples > 1:
            actions_mu = actions_mu[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)
            deltas_gt = deltas_gt[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)
            first_pose = first_pose[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)

        pred_actions = get_action_smpl_torch(first_pose, actions_mu, XSensConstants.upper_body_num_parts) # B, T, 48
        gt_actions = get_action_smpl_torch(first_pose, deltas_gt, XSensConstants.upper_body_num_parts) # B, T, 48

        (xyz_dist_matrix, _,
         leaf_xyz, _) = _compute_part_distance_matrices(pred_actions[:, -1], gt_actions[:, -1], skel)
        init_xyz_dist_matrix, _, leaf_xyz_init, _ = _compute_part_distance_matrices(first_pose[:, -1], gt_actions[:, -1], skel)

        intermediate_xyz = xyz_dist_matrix[:, XSensConstants.intermediate_indices].mean(dim=-1)
        intermediate_xyz_init = init_xyz_dist_matrix[:, XSensConstants.intermediate_indices].mean(dim=-1)

        all_xyz = xyz_dist_matrix.mean(dim=-1)
        all_xyz_init = init_xyz_dist_matrix.mean(dim=-1)

        res = {
            "leaf_xyz": leaf_xyz.mean().item(),
            "leaf_xyz_min": leaf_xyz.min().item(),
            "leaf_xyz_init": leaf_xyz_init.mean().item(),
            "intermediate_xyz": intermediate_xyz.mean().item(),
            "intermediate_xyz_min": intermediate_xyz.min().item(),
            "intermediate_xyz_init": intermediate_xyz_init.mean().item(),
            "all_xyz": all_xyz.mean().item(),
            "all_xyz_min": all_xyz.min().item(),
            "all_xyz_init": all_xyz_init.mean().item(),
        }
        other_vals = {
            "num_joints_visible": num_joints_visible.mean().item()
        }
        pprint({**res})
        return res, other_vals

    def eval_mu_step(self, cem_step, mu, mu_rollout_state, state_0, state_g,
                     cam_model=None, T_C_pelvis=None, save_path=None):
        """Evaluate mu after a CEM iteration: compute metrics and save artifacts."""
        metrics, other_vals = self.eval_actions(cem_step, mu, state_0, state_g)
        if save_path is not None:
            save_mu_step_results(mu_rollout_state, state_0, state_g,
                                 cam_model, T_C_pelvis, save_path)
        return metrics, other_vals


class EvaluatorWaypoint(WaypointWM):
    def __init__(self,
                 *args,
                 num_eval_samples=1,
                 skip_last_peva=True,
                 **kwargs):
        kwargs['skip_last_peva'] = skip_last_peva
        super().__init__(*args, **kwargs)
        
        self.num_eval_samples = num_eval_samples
        
        if not self.skip_last_peva:
            print("WARNING: Evaluator is not skipping last PEVA rollout")
    
    def eval_actions(self, cem_step, actions_mu, state_0, state_g):
        """
        actions_mu: B, T, action_dim
        gt_dict: dict of gt_actions, skel
        """
        B = actions_mu.shape[0]
        device = actions_mu.device
        leaf_indexer = torch.tensor([x in ["Pelvis", "Head", "R_Hand", "L_Hand"] for x in XSensConstants.part_names[:XSensConstants.upper_body_num_parts]], device=device) # 1, num_parts
        
        waypoints = actions_mu * self.image_size # B=1, W, 8
        context_poses = state_0['context_poses'] # B, peva_context_size, 48
        curr_obs = state_0['images'] # B, peva_context_size, 3, H, W
        goal_obs = state_0['goal_image'] # B, 3, H, W
        
        gt_goal_image = state_0['goal_image'] # B, 3, H, W

        deltas_gt = state_g["deltas"]
        first_pose = state_g["first_pose"] # B, 1, 48
        goal_image_coords = state_g['goal_image_coords'] # B, 23, 2
        gt_waypoints = goal_image_coords[:, :XSensConstants.upper_body_num_parts][:, leaf_indexer].flatten(1,2)[:, None] # B, 4, 2
        xsens_offsets = state_g["xsens_offsets"][0]
        skel = XsensSkeleton(xsens_offsets)
        
        num_joints_visible = (goal_image_coords[:, XSensConstants.leaf_indices] != -1).all(dim=-1).float().sum(dim=-1)
        
        if self.num_eval_samples > 1:
            waypoints = waypoints[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)
            context_poses = context_poses[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)
            curr_obs = curr_obs[:, None].repeat(1, self.num_eval_samples, 1, 1, 1, 1).flatten(0, 1)
            goal_obs = goal_obs[:, None].repeat(1, self.num_eval_samples, 1, 1, 1).flatten(0, 1)
            first_pose = first_pose[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)
            deltas_gt = deltas_gt[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)
            goal_image_coords = goal_image_coords[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)
            gt_waypoints = gt_waypoints[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)
        
        (
            generated_frames, # B, W, policy_pred_horizon, 3, H, W
            policy_sampled_deltas, # B, W, policy_pred_horizon, policy_action_dim
            waypoint_annotated_goals # B, W, 3, H, W
        ) = self.sample_fn(waypoints=waypoints, context_poses=context_poses, curr_obs=curr_obs, goal_obs=goal_obs, device=device)
        pred_deltas = policy_sampled_deltas.flatten(1,2)
        generated_frames = generated_frames.flatten(1,2)
        
        gt_waypoint_deltas = self.sample_fn(waypoints=gt_waypoints, context_poses=context_poses, curr_obs=curr_obs, goal_obs=goal_obs, device=device)[1].flatten(1,2)
        no_waypoint_deltas= self.sample_fn(waypoints=torch.ones_like(waypoints)*-1, context_poses=context_poses, curr_obs=curr_obs, goal_obs=goal_obs, device=device)[1].flatten(1,2)
        
        pred_actions = get_action_smpl_torch(first_pose, pred_deltas, XSensConstants.upper_body_num_parts) # B, T, 48
        gt_actions = get_action_smpl_torch(first_pose, deltas_gt, XSensConstants.upper_body_num_parts) # B, T, 48
        gt_waypoint_actions = get_action_smpl_torch(first_pose, gt_waypoint_deltas, XSensConstants.upper_body_num_parts) # B, T, 48
        no_waypoint_actions = get_action_smpl_torch(first_pose, no_waypoint_deltas, XSensConstants.upper_body_num_parts) # B, T, 48
        
        (xyz_dist_matrix, ang_dist_matrix, # B, num_parts
         leaf_xyz, leaf_ang) = _compute_part_distance_matrices(pred_actions[:, -1], gt_actions[:, -1], skel)
        gt_waypoint_xyz_dist_matrix, _, leaf_xyz_gt_waypoints, leaf_ang_gt_waypoints = _compute_part_distance_matrices(gt_waypoint_actions[:, -1], gt_actions[:, -1], skel)
        no_waypoint_xyz_dist_matrix, _, leaf_xyz_no_waypoints, leaf_ang_no_waypoints = _compute_part_distance_matrices(no_waypoint_actions[:, -1], gt_actions[:, -1], skel)
        init_xyz_dist_matrix, _, leaf_xyz_init, leaf_ang_init = _compute_part_distance_matrices(first_pose[:, -1], gt_actions[:, -1], skel)
        
        intermediate_xyz = xyz_dist_matrix[:, XSensConstants.intermediate_indices].mean(dim=-1)
        intermediate_xyz_init = init_xyz_dist_matrix[:, XSensConstants.intermediate_indices].mean(dim=-1)
        all_xyz = xyz_dist_matrix.mean(dim=-1)
        all_xyz_init = init_xyz_dist_matrix.mean(dim=-1)
        
        gt_intermediate_xyz = gt_waypoint_xyz_dist_matrix[:, XSensConstants.intermediate_indices].mean(dim=-1)
        gt_all_xyz = gt_waypoint_xyz_dist_matrix.mean(dim=-1)
        no_intermediate_xyz = no_waypoint_xyz_dist_matrix[:, XSensConstants.intermediate_indices].mean(dim=-1)
        no_all_xyz = no_waypoint_xyz_dist_matrix.mean(dim=-1)
        
        visible_indexer = (goal_image_coords != -1).all(dim=-1)[:, :XSensConstants.upper_body_num_parts] # B, num_parts
        indexer = torch.logical_and(leaf_indexer[None], visible_indexer) # B, num_parts
        visible_leaf_xyz_distance = torch.where(indexer, xyz_dist_matrix, torch.nan).nanmean(dim=-1) # B
        visible_leaf_angular_distance = torch.where(indexer, ang_dist_matrix, torch.nan).nanmean(dim=-1) # B
        
        wpts_ = waypoints.unflatten(-1, (4, 2)) # B, W, 4, 2
        avg_num_waypoints_visible = torch.logical_and(wpts_ < self.image_size, wpts_ > 0.).all(dim=-1).float().sum(dim=-1) # B
        
        res = {
            "leaf_xyz": leaf_xyz.mean().item(),
            "leaf_xyz_min": leaf_xyz.min().item(),
            "leaf_xyz_init": leaf_xyz_init.mean().item(),
            "intermediate_xyz": intermediate_xyz.mean().item(),
            "intermediate_xyz_min": intermediate_xyz.min().item(),
            "intermediate_xyz_init": intermediate_xyz_init.mean().item(),
            "all_xyz": all_xyz.mean().item(),
            "all_xyz_min": all_xyz.min().item(),
            "all_xyz_init": all_xyz_init.mean().item(),
            # others
            "gtwp.intermediate_xyz": gt_intermediate_xyz.mean().item(),
            "gtwp.intermediate_xyz_min": gt_intermediate_xyz.min().item(),
            "gtwp.all_xyz": gt_all_xyz.mean().item(),
            "gtwp.all_xyz_min": gt_all_xyz.min().item(),
            "nowp.intermediate_xyz": no_intermediate_xyz.mean().item(),
            "nowp.intermediate_xyz_min": no_intermediate_xyz.min().item(),
            "nowp.all_xyz": no_all_xyz.mean().item(),
            "nowp.all_xyz_min": no_all_xyz.min().item(),
        }
        aux_counts = {
            "avg_waypoints_visible": avg_num_waypoints_visible.mean().item(),
            "num_joints_visible": num_joints_visible.mean().item(),
        }
        pprint({**res, **aux_counts})
        return res, aux_counts

    def eval_mu_step(self, cem_step, mu, mu_rollout_state, state_0, state_g,
                     cam_model=None, T_C_pelvis=None, save_path=None):
        """Evaluate mu after a CEM iteration: compute metrics and save artifacts."""
        metrics, other_vals = self.eval_actions(cem_step, mu, state_0, state_g)
        if save_path is not None:
            save_mu_step_results(mu_rollout_state, state_0, state_g,
                                 cam_model, T_C_pelvis, save_path)
        return metrics, other_vals
