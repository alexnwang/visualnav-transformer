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
from planning.utils import _compute_pose_and_loss, _compute_part_distance_matrices, draw_waypoints
from planning.sampling import peva_sample, policy_sample, waypoint_sample

def build_skeleton_top_seq(curr_obs_img, pred_deltas, first_pose, xsens_offsets,
                           fisheye_params, R_C_pelvis, t_C_pelvis, image_size, T,
                           overlay='skeleton', smpl_alpha=0.7, show_text=True):
    """
    Build a (T, 3, H, W) sequence of curr_obs with skeleton overlays per timestep.
    If fisheye_params/R_C_pelvis/t_C_pelvis are None, returns blank (zero) frames.

    Args:
        curr_obs_img:   (3, H, W) current observation image
        pred_deltas:    (1, T, action_dim) action deltas (batch-of-1)
        first_pose:     (1, 1, action_dim) starting pose (batch-of-1)
        xsens_offsets:  (15, 3) skeleton joint offsets
        fisheye_params: (15,) Fisheye624 intrinsics, or None to skip
        R_C_pelvis:     (3, 3) rotation pelvis→camera, or None to skip
        t_C_pelvis:     (3,)   translation pelvis→camera, or None to skip
        image_size:     int, spatial size of images
        T:              number of timesteps
        overlay:        'skeleton', 'skin', or 'both'
        smpl_alpha:     opacity of the SMPL mesh overlay (only used with skin)
    Returns:
        top_seq: (T, 3, H, W) tensor
    """
    device = curr_obs_img.device
    if fisheye_params is not None and R_C_pelvis is not None and t_C_pelvis is not None:
        from torchvision import transforms as T_transforms

        pred_actions = get_action_smpl_torch(
            first_pose, pred_deltas, XSensConstants.upper_body_num_parts
        )  # (1, T, 48)

        draw_skin = overlay in ('skin', 'both')
        draw_skel = overlay in ('skeleton', 'both')

        skel_tensors = []
        for t in range(pred_actions.shape[1]):
            if draw_skin:
                from planning.vis_utils import render_smpl_on_image
                img_pil = render_smpl_on_image(
                    curr_obs_img, pred_actions[:, t],
                    R_C_pelvis, t_C_pelvis, fisheye_params,
                    image_size, alpha=smpl_alpha,
                    xsens_offsets=xsens_offsets,
                    draw_skeleton=draw_skel,
                    show_text=show_text,
                )
            else:
                from planning.vis_utils import pose_to_image_coords_v2, draw_image_coords as draw_skel_fn
                img_pil = Image.fromarray(
                    (255.0 * curr_obs_img.permute(1, 2, 0)).to(torch.uint8).cpu().numpy()
                )
                draw = ImageDraw.Draw(img_pil)
                image_coords = pose_to_image_coords_v2(
                    pred_actions[:, t], R_C_pelvis, t_C_pelvis, fisheye_params,
                    xsens_offsets, image_size=image_size
                )  # (1, 15, 2)
                draw_skel_fn(draw, image_coords, show_text=show_text)
            skel_tensors.append(T_transforms.ToTensor()(img_pil))
        return torch.stack(skel_tensors).to(device)  # (T, 3, H, W)
    else:
        return torch.zeros(T, *curr_obs_img.shape).to(device)


def save_mu_step_results(mu_rollout_state, state_0, state_g,
                         fisheye_params, R_C_pelvis, t_C_pelvis, save_path,
                         render_skin=True):
    """
    Save per-iteration mu evaluation artifacts:
      - action_obs_seq.png: 2-row visualization
          top: [policy input goal | curr_obs+skel t=1 | ... | curr_obs+skel t=T | blank]
               policy input goal = curr_obs+predicted waypoints (waypoint CEM) or plain curr_obs (peva)
               skeleton overlays are blank if camera params not provided
          bot: [curr_obs | generated_obs t=1 | ... | generated_obs t=T | goal_obs]

    Args:
        mu_rollout_state: output of wm.rollout(state_0, mu) — generated_obs, deltas, goal_images
        state_0: transformed initial observation dict (trans_obs_0)
        state_g: encoded goal state dict (z_obs_g)
        fisheye_params: (15,) Fisheye624 intrinsics, or None to skip skeleton vis
        R_C_pelvis: (3, 3) rotation pelvis→camera, or None to skip skeleton vis
        t_C_pelvis: (3,) translation pelvis→camera, or None to skip skeleton vis
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

    if render_skin:
        top_seq = build_skeleton_top_seq(
            curr_obs_img, pred_deltas, first_pose, xsens_offsets,
            fisheye_params, R_C_pelvis, t_C_pelvis, image_size, T,
            overlay='both', smpl_alpha=0.9, show_text=False,
        )
    top_seq_noskin = build_skeleton_top_seq(
        curr_obs_img, pred_deltas, first_pose, xsens_offsets,
        fisheye_params, R_C_pelvis, t_C_pelvis, image_size, T,
        show_text=False,
    )

    # top-left: curr_obs + predicted waypoints (waypoint CEM) or plain curr_obs (peva)
    policy_input_goal = goal_images[0, 0] if goal_images is not None else curr_obs_img
    if render_skin:
        save_action_obs_sequence_viz(
            save_path=os.path.join(save_path, "action_obs_seq.png"),
            goal_image=policy_input_goal,
            curr_obs=curr_obs_img,
            goal_obs=goal_obs,
            top_seq=top_seq,
            bot_seq=generated_obs[0],           # (T, 3, H, W)
        )
    save_action_obs_sequence_viz(
        save_path=os.path.join(save_path, "action_obs_seq_noskin.png"),
        goal_image=policy_input_goal,
        curr_obs=curr_obs_img,
        goal_obs=goal_obs,
        top_seq=top_seq_noskin,
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
    def __init__(self, peva_model, peva_diffusion, peva_vae, peva_stats,
                 policy_model, policy_diffusion,
                 image_size, peva_context_size,
                 policy_context_size, policy_pred_horizon, policy_action_dim,
                 skip_last_peva=False,
                 waypoint_mode="waypoint"):
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
        self.waypoint_mode = waypoint_mode

        self.sample_fn = partial(
            waypoint_sample,
            policy_model=policy_model, policy_diffusion=policy_diffusion,
            peva_model=peva_model, peva_diffusion=peva_diffusion, peva_vae=peva_vae, peva_stats=peva_stats,
            policy_pred_horizon=policy_pred_horizon, policy_action_dim=policy_action_dim,
            image_size=image_size,
            policy_context_size=policy_context_size, peva_context_size=peva_context_size, peva_latent_size=self.latent_size,
            skip_last_peva=skip_last_peva,
            waypoint_mode=waypoint_mode,
        )

    def _scale_waypoints(self, act):
        """Convert normalized action coords to pixel/metric space.
        For waypoint_point3d, only x,y dims are scaled to pixels; depth is left as-is."""
        if self.waypoint_mode == "waypoint_point3d":
            waypoints = act.reshape(*act.shape[:-1], 4, 3).clone()
            waypoints[..., :2] *= self.image_size
            return waypoints.flatten(-2, -1)
        else:
            return act * self.image_size
    
    def set_skip_last_peva(self, skip_last_peva):
        self.skip_last_peva = skip_last_peva

    def rollout(self, state_0, act):
        device = act.device
        curr_obs = state_0['images'] # B, peva_context_size, 3, H, W
        goal_obs = state_0['goal_image'] # B, 3, H, W
        context_poses = state_0['context_poses'] # B, peva_context_size, 48

        waypoints = self._scale_waypoints(act)

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
                    skip_last_peva=self.skip_last_peva,
                    waypoint_mode=self.waypoint_mode)

        return {"generated_obs": generated_frames.to(torch.float32).flatten(1,2),
                "deltas": policy_sampled_deltas.to(torch.float32).flatten(1,2),
                "goal_images": waypoint_annotated_goals.to(torch.float32)}

    def policy_only_rollout(self, state_0, act):
        """Run only the policy (no PEVA), returning deltas and goal images.

        For W>1, poses are propagated but images are not updated between steps
        (since PEVA is skipped).
        """
        imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        device = act.device
        curr_obs = state_0['images']  # B, ctx, 3, H, W
        context_poses = state_0['context_poses']  # B, ctx, 48
        waypoints = self._scale_waypoints(act)

        B, W = waypoints.shape[:2]
        delta_accum = torch.zeros(B, W, self.policy_pred_horizon, self.policy_action_dim, device=device)
        goal_obs_accum = torch.zeros(B, W, 3, self.image_size, self.image_size, device=device)

        curr_poses = context_poses.clone()
        for w in range(W):
            policy_obs = imagenet_norm(curr_obs[:, -self.policy_context_size:].flatten(0, 1)).unflatten(0, (B, self.policy_context_size))
            if self.waypoint_mode == "waypoint_point3d":
                xy_waypoints = waypoints[:, w].reshape(-1, 4, 3)[:, :, :2]
                goal_obs_accum[:, w] = draw_waypoints(curr_obs[:, -1], xy_waypoints)
                goal_img = imagenet_norm(curr_obs[:, -1])
                goal_coords = waypoints[:, w]
            else:
                goal_obs_accum[:, w] = draw_waypoints(curr_obs[:, -1], waypoints[:, w])
                goal_img = imagenet_norm(goal_obs_accum[:, w])
                goal_coords = None

            deltas = policy_sample(self.policy_model, self.policy_diffusion,
                        policy_obs, goal_img,
                        curr_poses[:, -self.policy_context_size:],
                        self.policy_pred_horizon, self.policy_action_dim, device,
                        goal_coordinates=goal_coords)
            delta_accum[:, w] = deltas

            new_poses = get_action_smpl_torch(curr_poses[:, -1:], deltas, XSensConstants.upper_body_num_parts)
            new_poses[:, :, :6] = torch.zeros_like(new_poses[:, :, :6])
            curr_poses = torch.cat([curr_poses[:, new_poses.shape[1]:], new_poses], dim=1)

        return {"deltas": delta_accum.to(torch.float32).flatten(1,2),
                "goal_images": goal_obs_accum.to(torch.float32)}

    def wm_only_rollout(self, state_0, precomputed_deltas, goal_images=None):
        """Run only PEVA world model using pre-computed deltas from the policy."""
        device = precomputed_deltas.device
        curr_obs = state_0['images']

        generated_frames, deltas = peva_sample(
            self.peva_model, self.peva_diffusion, self.peva_vae, self.peva_stats,
            curr_obs, precomputed_deltas,
            self.peva_context_size, self.latent_size,
            self.image_size, device)

        return {"generated_obs": generated_frames.to(torch.float32),
                "deltas": deltas.to(torch.float32),
                "goal_images": goal_images}

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
                 fisheye_params=None, R_C_pelvis=None, t_C_pelvis=None):
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
        _, ang_dist_matrix_init, leaf_xyz_init, leaf_ang_init = _compute_part_distance_matrices(first_pose[:, -1], gt_actions[:, -1], skel)
        
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
                    fisheye_params, R_C_pelvis, t_C_pelvis, image_size, T
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
            return leaf_xyz, {"dreamsim": res.mean().item(),
                              "objective-xyz_distance": leaf_xyz.mean().item(),
                              "objective-start_xyz_distance": leaf_xyz_init.mean().item()}
        else:
            return res, {"dreamsim": res.mean().item(),
                         "xyz_distance": xyz_dist_matrix.mean().item(),
                         "start_xyz_distance": leaf_xyz_init.mean().item(),
                         "angular_distance": ang_dist_matrix.mean().item(),
                         "start_angular_distance": ang_dist_matrix_init.mean().item()}
    
class EvaluatorPeva(PevaWM):
    def __init__(self,
                 *args,
                 num_eval_samples=1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_eval_samples = num_eval_samples
    
    def eval_task(self, state_0, state_g, mu, objective_fn):
        """Compute per-task constant metrics (init baselines, visibility). Returns (task_metrics, {}, {})."""
        goal_image_coords = state_g['goal_image_coords']  # B, 23, 2
        xsens_offsets = state_g["xsens_offsets"][0]
        skel = XsensSkeleton(xsens_offsets)
        deltas_gt = state_g["deltas"]
        first_pose = state_g["first_pose"]  # B, 1, 48

        gt_actions = get_action_smpl_torch(first_pose, deltas_gt, XSensConstants.upper_body_num_parts)
        init_xyz_dist_matrix, _, leaf_xyz_init, _ = _compute_part_distance_matrices(first_pose[:, -1], gt_actions[:, -1], skel)
        intermediate_xyz_init = init_xyz_dist_matrix[:, XSensConstants.intermediate_indices].mean(dim=-1)
        all_xyz_init = init_xyz_dist_matrix.mean(dim=-1)
        num_joints_visible = (goal_image_coords[:, XSensConstants.leaf_indices] != -1).all(dim=-1).float().sum(dim=-1)

        with torch.no_grad():
            mu_state = self.rollout(state_0=state_0, act=mu)
        mu_loss, _ = objective_fn(-1, mu_state, state_0, state_g, save_path=None, topk=0)

        return {
            "leaf_xyz_init": leaf_xyz_init.mean().item(),
            "intermediate_xyz_init": intermediate_xyz_init.mean().item(),
            "all_xyz_init": all_xyz_init.mean().item(),
            "num_joints_visible": num_joints_visible.mean().item(),
            "dreamsim_init": mu_loss[0].item(),
        }, {}, {}

    def eval_actions(self, cem_step, actions_mu, state_0, state_g):
        xsens_offsets = state_g["xsens_offsets"][0]
        skel = XsensSkeleton(xsens_offsets)
        deltas_gt = state_g["deltas"]
        first_pose = state_g["first_pose"]  # B, 1, 48

        if self.num_eval_samples > 1:
            actions_mu = actions_mu[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)
            deltas_gt = deltas_gt[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)
            first_pose = first_pose[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)

        pred_actions = get_action_smpl_torch(first_pose, actions_mu, XSensConstants.upper_body_num_parts)
        gt_actions = get_action_smpl_torch(first_pose, deltas_gt, XSensConstants.upper_body_num_parts)

        (xyz_dist_matrix, _, leaf_xyz, _) = _compute_part_distance_matrices(pred_actions[:, -1], gt_actions[:, -1], skel)
        intermediate_xyz = xyz_dist_matrix[:, XSensConstants.intermediate_indices].mean(dim=-1)
        all_xyz = xyz_dist_matrix.mean(dim=-1)

        metrics = {
            "leaf_xyz": leaf_xyz.mean().item(),
            "leaf_xyz_min": leaf_xyz.min().item(),
            "intermediate_xyz": intermediate_xyz.mean().item(),
            "intermediate_xyz_min": intermediate_xyz.min().item(),
            "all_xyz": all_xyz.mean().item(),
            "all_xyz_min": all_xyz.min().item(),
        }
        pprint(metrics)
        return metrics, {}

    def eval_mu_step(self, cem_step, mu, mu_rollout_state, state_0, state_g,
                     fisheye_params=None, R_C_pelvis=None, t_C_pelvis=None, save_path=None,
                     render_skin=True):
        """Evaluate mu after a CEM iteration: compute metrics and save artifacts."""
        metrics, other_vals = self.eval_actions(cem_step, mu, state_0, state_g)
        if save_path is not None:
            save_mu_step_results(mu_rollout_state, state_0, state_g,
                                 fisheye_params, R_C_pelvis, t_C_pelvis, save_path,
                                 render_skin=render_skin)
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
    
    def eval_task(self, state_0, state_g, mu, objective_fn):
        """Compute per-task constant metrics (init baselines, gtwp, nowp, visibility).
        Returns (task_metrics, gtwp_metrics, nowp_metrics)."""
        device = state_0['images'].device
        leaf_indexer = torch.tensor([x in ["Pelvis", "Head", "R_Hand", "L_Hand"] for x in XSensConstants.part_names[:XSensConstants.upper_body_num_parts]], device=device)

        context_poses = state_0['context_poses']
        curr_obs = state_0['images']
        goal_obs = state_0['goal_image']

        goal_image_coords = state_g['goal_image_coords']  # B, 23, 2
        xsens_offsets = state_g["xsens_offsets"][0]
        skel = XsensSkeleton(xsens_offsets)
        deltas_gt = state_g["deltas"]
        first_pose = state_g["first_pose"]  # B, 1, 48

        B = curr_obs.shape[0]
        num_joints_visible = (goal_image_coords[:, XSensConstants.leaf_indices] != -1).all(dim=-1).float().sum(dim=-1)

        gt_actions = get_action_smpl_torch(first_pose, deltas_gt, XSensConstants.upper_body_num_parts)
        init_xyz_dist_matrix, _, leaf_xyz_init, _ = _compute_part_distance_matrices(first_pose[:, -1], gt_actions[:, -1], skel)
        intermediate_xyz_init = init_xyz_dist_matrix[:, XSensConstants.intermediate_indices].mean(dim=-1)
        all_xyz_init = init_xyz_dist_matrix.mean(dim=-1)

        with torch.no_grad():
            self.set_skip_last_peva(False)
            mu_state = self.rollout(state_0=state_0, act=mu)
            self.set_skip_last_peva(True)
        mu_loss, _ = objective_fn(-1, mu_state, state_0, state_g, save_path=None, topk=0)

        task_metrics = {
            "leaf_xyz_init": leaf_xyz_init.mean().item(),
            "intermediate_xyz_init": intermediate_xyz_init.mean().item(),
            "all_xyz_init": all_xyz_init.mean().item(),
            "num_joints_visible": num_joints_visible.mean().item(),
            "dreamsim_init": mu_loss[0].item(),
        }

        N = max(self.num_eval_samples, 1)
        rep_context_poses = context_poses[:, None].repeat(1, N, 1, 1).flatten(0, 1) if N > 1 else context_poses
        rep_curr_obs = curr_obs[:, None].repeat(1, N, 1, 1, 1, 1).flatten(0, 1) if N > 1 else curr_obs
        rep_goal_obs = goal_obs[:, None].repeat(1, N, 1, 1, 1).flatten(0, 1) if N > 1 else goal_obs
        rep_first_pose = first_pose[:, None].repeat(1, N, 1, 1).flatten(0, 1) if N > 1 else first_pose
        rep_gt_actions = gt_actions[:, None].repeat(1, N, 1, 1).flatten(0, 1) if N > 1 else gt_actions

        if self.waypoint_mode == "waypoint_point3d":
            # gt 3D waypoints not yet available; nowp uses all-(-1) goal_coordinates
            no_waypoints = torch.ones(B * N, 1, 12, device=device) * -1
            no_waypoint_deltas = self.sample_fn(waypoints=no_waypoints, context_poses=rep_context_poses, curr_obs=rep_curr_obs, goal_obs=rep_goal_obs, device=device)[1].flatten(1, 2)
            no_waypoint_actions = get_action_smpl_torch(rep_first_pose, no_waypoint_deltas, XSensConstants.upper_body_num_parts)
            no_waypoint_xyz_dist_matrix, _, leaf_xyz_no_waypoints, _ = _compute_part_distance_matrices(no_waypoint_actions[:, -1], rep_gt_actions[:, -1], skel)
            no_intermediate_xyz = no_waypoint_xyz_dist_matrix[:, XSensConstants.intermediate_indices].mean(dim=-1)
            no_all_xyz = no_waypoint_xyz_dist_matrix.mean(dim=-1)
            nowp_metrics = {
                "leaf_xyz": leaf_xyz_no_waypoints.mean().item(),
                "leaf_xyz_min": leaf_xyz_no_waypoints.min().item(),
                "intermediate_xyz": no_intermediate_xyz.mean().item(),
                "intermediate_xyz_min": no_intermediate_xyz.min().item(),
                "all_xyz": no_all_xyz.mean().item(),
                "all_xyz_min": no_all_xyz.min().item(),
            }
            return task_metrics, {}, nowp_metrics
        else:
            gt_waypoints = goal_image_coords[:, :XSensConstants.upper_body_num_parts][:, leaf_indexer].flatten(1, 2)[:, None]

            gtwp_waypoints = gt_waypoints
            if N > 1:
                gtwp_waypoints = gtwp_waypoints[:, None].repeat(1, N, 1, 1).flatten(0, 1)
            gt_waypoint_deltas = self.sample_fn(waypoints=gtwp_waypoints, context_poses=rep_context_poses, curr_obs=rep_curr_obs, goal_obs=rep_goal_obs, device=device)[1].flatten(1, 2)
            gt_waypoint_actions = get_action_smpl_torch(rep_first_pose, gt_waypoint_deltas, XSensConstants.upper_body_num_parts)
            gt_waypoint_xyz_dist_matrix, _, leaf_xyz_gt_waypoints, _ = _compute_part_distance_matrices(gt_waypoint_actions[:, -1], rep_gt_actions[:, -1], skel)
            gt_intermediate_xyz = gt_waypoint_xyz_dist_matrix[:, XSensConstants.intermediate_indices].mean(dim=-1)
            gt_all_xyz = gt_waypoint_xyz_dist_matrix.mean(dim=-1)

            nowp_waypoints = torch.ones_like(gt_waypoints) * -1
            if N > 1:
                nowp_waypoints = nowp_waypoints[:, None].repeat(1, N, 1, 1).flatten(0, 1)
            no_waypoint_deltas = self.sample_fn(waypoints=nowp_waypoints, context_poses=rep_context_poses, curr_obs=rep_curr_obs, goal_obs=rep_goal_obs, device=device)[1].flatten(1, 2)
            no_waypoint_actions = get_action_smpl_torch(rep_first_pose, no_waypoint_deltas, XSensConstants.upper_body_num_parts)
            no_waypoint_xyz_dist_matrix, _, leaf_xyz_no_waypoints, _ = _compute_part_distance_matrices(no_waypoint_actions[:, -1], rep_gt_actions[:, -1], skel)
            no_intermediate_xyz = no_waypoint_xyz_dist_matrix[:, XSensConstants.intermediate_indices].mean(dim=-1)
            no_all_xyz = no_waypoint_xyz_dist_matrix.mean(dim=-1)

            gtwp_metrics = {
                "leaf_xyz": leaf_xyz_gt_waypoints.mean().item(),
                "leaf_xyz_min": leaf_xyz_gt_waypoints.min().item(),
                "intermediate_xyz": gt_intermediate_xyz.mean().item(),
                "intermediate_xyz_min": gt_intermediate_xyz.min().item(),
                "all_xyz": gt_all_xyz.mean().item(),
                "all_xyz_min": gt_all_xyz.min().item(),
            }
            nowp_metrics = {
                "leaf_xyz": leaf_xyz_no_waypoints.mean().item(),
                "leaf_xyz_min": leaf_xyz_no_waypoints.min().item(),
                "intermediate_xyz": no_intermediate_xyz.mean().item(),
                "intermediate_xyz_min": no_intermediate_xyz.min().item(),
                "all_xyz": no_all_xyz.mean().item(),
                "all_xyz_min": no_all_xyz.min().item(),
            }
            return task_metrics, gtwp_metrics, nowp_metrics

    def eval_actions(self, cem_step, actions_mu, state_0, state_g):
        device = actions_mu.device
        xsens_offsets = state_g["xsens_offsets"][0]
        skel = XsensSkeleton(xsens_offsets)

        waypoints = self._scale_waypoints(actions_mu)
        context_poses = state_0['context_poses']
        curr_obs = state_0['images']
        goal_obs = state_0['goal_image']
        deltas_gt = state_g["deltas"]
        first_pose = state_g["first_pose"]  # B, 1, 48

        if self.num_eval_samples > 1:
            waypoints = waypoints[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)
            context_poses = context_poses[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)
            curr_obs = curr_obs[:, None].repeat(1, self.num_eval_samples, 1, 1, 1, 1).flatten(0, 1)
            goal_obs = goal_obs[:, None].repeat(1, self.num_eval_samples, 1, 1, 1).flatten(0, 1)
            first_pose = first_pose[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)
            deltas_gt = deltas_gt[:, None].repeat(1, self.num_eval_samples, 1, 1).flatten(0, 1)

        (_, policy_sampled_deltas, _) = self.sample_fn(waypoints=waypoints, context_poses=context_poses, curr_obs=curr_obs, goal_obs=goal_obs, device=device)
        pred_deltas = policy_sampled_deltas.flatten(1, 2)

        pred_actions = get_action_smpl_torch(first_pose, pred_deltas, XSensConstants.upper_body_num_parts)
        gt_actions = get_action_smpl_torch(first_pose, deltas_gt, XSensConstants.upper_body_num_parts)

        (xyz_dist_matrix, _, leaf_xyz, _) = _compute_part_distance_matrices(pred_actions[:, -1], gt_actions[:, -1], skel)
        intermediate_xyz = xyz_dist_matrix[:, XSensConstants.intermediate_indices].mean(dim=-1)
        all_xyz = xyz_dist_matrix.mean(dim=-1)

        if self.waypoint_mode == "waypoint_point3d":
            wpts_xy = waypoints.unflatten(-1, (4, 3))[..., :2]  # B, W, 4, 2
        else:
            wpts_xy = waypoints.unflatten(-1, (4, 2))  # B, W, 4, 2
        avg_num_waypoints_visible = torch.logical_and(wpts_xy < self.image_size, wpts_xy > 0.).all(dim=-1).float().sum(dim=-1)

        metrics = {
            "leaf_xyz": leaf_xyz.mean().item(),
            "leaf_xyz_min": leaf_xyz.min().item(),
            "intermediate_xyz": intermediate_xyz.mean().item(),
            "intermediate_xyz_min": intermediate_xyz.min().item(),
            "all_xyz": all_xyz.mean().item(),
            "all_xyz_min": all_xyz.min().item(),
        }
        other_vals = {
            "avg_waypoints_visible": avg_num_waypoints_visible.mean().item(),
        }
        pprint({**metrics, **other_vals})
        return metrics, other_vals

    def eval_mu_step(self, cem_step, mu, mu_rollout_state, state_0, state_g,
                     fisheye_params=None, R_C_pelvis=None, t_C_pelvis=None, save_path=None,
                     render_skin=True):
        """Evaluate mu after a CEM iteration: compute metrics and save artifacts."""
        metrics, other_vals = self.eval_actions(cem_step, mu, state_0, state_g)
        if save_path is not None:
            save_mu_step_results(mu_rollout_state, state_0, state_g,
                                 fisheye_params, R_C_pelvis, t_C_pelvis, save_path,
                                 render_skin=render_skin)
        return metrics, other_vals
