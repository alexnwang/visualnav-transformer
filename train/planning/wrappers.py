from functools import partial
import os
from pprint import pprint
import torch
import copy
from torchvision import transforms
from dreamsim import dreamsim
from torchvision.utils import np
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.training.nymeria_training_utils import get_action_smpl_torch
from planning.utils import _compute_pose_and_loss, _compute_part_distance_matrices
from planning.sampling import waypoint_sample
import matplotlib.pyplot as plt

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

class WaypointWM(torch.nn.Module):
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
        
    def encode_obs(self, obs):
        return copy.deepcopy(obs)
    
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
        
        return {"generated_obs": generated_frames.flatten(1,2).to(torch.float32),
                "deltas": policy_sampled_deltas.flatten(1,2).to(torch.float32),
                "goal_images": waypoint_annotated_goals.to(torch.float32)}

class ObjectiveDreamSIM:
    def __init__(self, device):
        self.device = device
        self.model, self.preprocess = dreamsim(pretrained=True, device=device, cache_dir="/scratch/anw2067/cache")
        self.save_dir = None

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
    
    def __call__(self, cem_step, rollout_state, state_0, goal_state, save_path=None, topk=0):
        first_pose = goal_state["first_pose"]
        deltas = rollout_state["deltas"]
        gt_deltas = goal_state["deltas"]
        xsens_offsets = goal_state["xsens_offsets"][0]
        skel = XsensSkeleton(xsens_offsets)
        
        pred_actions = get_action_smpl_torch(first_pose, deltas, XSensConstants.upper_body_num_parts) # B, T, 48
        gt_actions = get_action_smpl_torch(first_pose, gt_deltas, XSensConstants.upper_body_num_parts) # B, T, 48
        
        (xyz_dist_matrix, ang_dist_matrix, leaf_xyz, leaf_ang) = _compute_part_distance_matrices(pred_actions[:, -1], gt_actions[:, -1], skel)
        _, _, leaf_xyz_init, leaf_ang_init = _compute_part_distance_matrices(first_pose[:, -1], gt_actions[:, -1], skel)
        
        pred_image = rollout_state["generated_obs"]
        goal_image = goal_state["images"]
        B = pred_image.shape[0]
        res = torch.empty(B, device=self.device)
        for i in range(B):
            pred_image_pil = transforms.ToPILImage()(pred_image[i, -1])
            goal_image_pil = transforms.ToPILImage()(goal_image[i])
            
            rollout_state = self.preprocess(pred_image_pil).to(self.device)
            goal_state = self.preprocess(goal_image_pil).to(self.device)

            sim = self.model(rollout_state, goal_state)
            res[i] = sim
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            self.save_plot(res.detach().cpu().numpy(),
                           leaf_xyz.detach().cpu().numpy(),
                           leaf_xyz_init[0].detach().cpu().numpy(),
                           "DreamSIM", "Leaf XYZ Distance", 
                           f"{save_path}/step{cem_step}-dreamSIM_xyz.png", k=topk)
            self.save_plot(res.detach().cpu().numpy(),
                           leaf_ang.detach().cpu().numpy(),
                           leaf_ang_init[0].detach().cpu().numpy(),
                           "DreamSIM", "Leaf Angular Distance", 
                           f"{save_path}/step{cem_step}-dreamSIM_ang.png", k=topk)
        print(f"ObjectiveFn: {res.mean().item()}")
        return res
    
    def save_plot(self, x, y, line_y, x_label, y_label, filename, k=0):
        plt.figure()
        if k > 0:
            argsort = np.argsort(x)
            plt.scatter(x[argsort[:k]], y[argsort[:k]], color='b')
            plt.scatter(x[argsort[k:]], y[argsort[k:]], color='grey')
        else:
            plt.scatter(x, y)
        plt.axhline(line_y, color='r', linestyle='--')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(filename.format(k=k))
        plt.close()
    

class Evaluator(WaypointWM):
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
    
    def eval_actions(self, actions_mu, state_0, state_g):
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

        deltas_gt = state_g["deltas"]
        first_pose = state_g["first_pose"] # B, 1, 48
        goal_image_coords = state_g['goal_image_coords'] # B, 23, 2
        gt_waypoints = goal_image_coords[:, :XSensConstants.upper_body_num_parts][:, leaf_indexer].flatten(1,2)[:, None] # B, 4, 2
        xsens_offsets = state_g["xsens_offsets"][0]
        skel = XsensSkeleton(xsens_offsets)
        
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
        
        _, deltas_gt_waypoints, _ = self.sample_fn(waypoints=gt_waypoints, context_poses=context_poses, curr_obs=curr_obs, goal_obs=goal_obs, device=device)
        gt_waypoint_deltas = deltas_gt_waypoints.flatten(1,2)
        
        pred_actions = get_action_smpl_torch(first_pose, pred_deltas, XSensConstants.upper_body_num_parts) # B, T, 48
        gt_actions = get_action_smpl_torch(first_pose, deltas_gt, XSensConstants.upper_body_num_parts) # B, T, 48
        gt_waypoint_actions = get_action_smpl_torch(first_pose, gt_waypoint_deltas, XSensConstants.upper_body_num_parts) # B, T, 48
        
        (xyz_dist_matrix, ang_dist_matrix, # B, num_parts
         leaf_xyz, leaf_ang) = _compute_part_distance_matrices(pred_actions[:, -1], gt_actions[:, -1], skel)
        _, _, leaf_xyz_gt_waypoints, leaf_ang_gt_waypoints = _compute_part_distance_matrices(gt_waypoint_actions[:, -1], gt_actions[:, -1], skel)
        _, _, leaf_xyz_init, leaf_ang_init = _compute_part_distance_matrices(first_pose[:, -1], gt_actions[:, -1], skel)
        
        visible_indexer = (goal_image_coords != -1).all(dim=-1)[:, :XSensConstants.upper_body_num_parts] # B, num_parts
        indexer = torch.logical_and(leaf_indexer[None], visible_indexer) # B, num_parts
        visible_leaf_xyz_distance = torch.where(indexer, xyz_dist_matrix, torch.nan).nanmean(dim=-1) # B
        visible_leaf_angular_distance = torch.where(indexer, ang_dist_matrix, torch.nan).nanmean(dim=-1) # B
        
        wpts_ = waypoints.unflatten(-1, (4, 2)) # B, W, 4, 2
        avg_num_waypoints_visible = torch.logical_and(wpts_ < self.image_size, wpts_ > 0.).all(dim=-1).float().sum(dim=-1) # B
        
        res = {
            "xyz_distance": leaf_xyz.mean().item(),
            # "angular_distance": leaf_ang.mean().item(),
            "min-xyz_distance": leaf_xyz.min().item(),
            # "min-angular_distance": leaf_ang.min().item(),
            "start_xyz_distance": leaf_xyz_init.mean().item(),
            # "start_angular_distance": leaf_ang_init.mean().item(),
            "visible_xyz_distance": visible_leaf_xyz_distance.mean().item(),
            # "visible_angular_distance": visible_leaf_angular_distance.mean().item(),
            "min-visible_xyz_distance": visible_leaf_xyz_distance.min().item(),
            # "min-visible_angular_distance": visible_leaf_angular_distance.min().item(),
            "gt_waypoint-xyz_distance": leaf_xyz_gt_waypoints.mean().item(),
            # "gt_waypoint-angular_distance": leaf_ang_gt_waypoints.mean().item(),
            "min-gt_waypoint-xyz_distance": leaf_xyz_gt_waypoints.min().item(),
            # "min-gt_waypoint-angular_distance": leaf_ang_gt_waypoints.min().item(),
            # other stuff
            "avg_waypoints_visible": avg_num_waypoints_visible.mean().item(),
        }
        pprint(res)
        return res
