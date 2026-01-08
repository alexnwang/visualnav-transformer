import torch
import copy
from torchvision import transforms
from dreamsim import dreamsim
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.training.nymeria_training_utils import get_action_smpl_torch
from planning.utils import _compute_pose_and_loss
from planning.sampling import waypoint_sample

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
        
    def encode_obs(self, obs):
        return copy.deepcopy(obs)
    
    def rollout(self, obs_0, act):
        device = act.device
        curr_obs = obs_0['images'] # B, peva_context_size, 3, H, W
        goal_obs = obs_0['goal_image'] # B, 3, H, W
        context_poses = obs_0['context_poses'] # B, peva_context_size, 48
        
        generated_frames, delta_accum = waypoint_sample(self.policy_model, self.policy_diffusion,
                    self.peva_model, self.peva_diffusion, self.peva_vae, self.peva_stats,
                    act, context_poses, curr_obs, goal_obs,
                    self.policy_pred_horizon, self.policy_action_dim,
                    self.image_size, 
                    self.policy_context_size, self.peva_context_size, self.latent_size,
                    device,
                    skip_last_peva=self.skip_last_peva)
        
        # all_frames = torch.cat([obs_0['images'], generated_frames], dim=1)
        # all_frames = all_frames * 0.5 + 0.5
        # for i in range(all_frames.shape[0]):
        #     image = torch.cat([img for img in all_frames[i]], dim=-1)
        #     save_image(image, f"rollout_{i}.png")
        return {"images": generated_frames.to(torch.float32), "deltas": delta_accum.to(torch.float32)}, None

class ObjectiveDreamSIM:
    def __init__(self, device):
        self.device = device
        self.model, self.preprocess = dreamsim(pretrained=True, device=device, cache_dir="/scratch/anw2067/cache")
        
    def __call__(self, rollout_state, goal_state):
        
        pred_image = rollout_state["images"]
        goal_image = goal_state["images"]
        B = pred_image.shape[0]
        res = []
        for i in range(B):
            pred_image_pil = transforms.ToPILImage()(pred_image[i, -1])
            goal_image_pil = transforms.ToPILImage()(goal_image[i])
            
            rollout_state = self.preprocess(pred_image_pil).to(self.device)
            goal_state = self.preprocess(goal_image_pil).to(self.device)

            sim = self.model(rollout_state, goal_state)
            res.append(sim)
        res = torch.cat(res, dim=0) # B
        print(f"ObjectiveFn: {res.mean().item()}")
        return res
    
# def ObjectiveImageProjection: 
#     def __init__(self, device):
#         self.device = device

class Evaluator(WaypointWM):
    def __init__(self, *args,
                 skip_last_peva=True,
                 **kwargs):
        kwargs['skip_last_peva'] = skip_last_peva
        super().__init__(*args, **kwargs)
        
        if not self.skip_last_peva:
            print("WARNING: Evaluator is not skipping last PEVA rollout")
    
    def eval_actions(self, actions_mu, state_0, state_g):
        """
        actions_mu: B, T, action_dim
        gt_dict: dict of gt_actions, skel
        """
        waypoints = actions_mu 
        curr_obs = state_0['images'] # B, peva_context_size, 3, H, W
        goal_obs = state_0['goal_image'] # B, 3, H, W
        context_poses = state_0['context_poses'] # B, peva_context_size, 48
        goal_image_coords = state_g["goal_image_coords"] # B, 23, 2
        device = curr_obs.device
        
        _, pred_actions = waypoint_sample(self.policy_model, self.policy_diffusion,
                    self.peva_model, self.peva_diffusion, self.peva_vae, self.peva_stats,
                    waypoints, context_poses, curr_obs, goal_obs,
                    self.policy_pred_horizon, self.policy_action_dim,
                    self.image_size, 
                    self.policy_context_size, self.peva_context_size, self.latent_size,
                    device,
                    skip_last_peva=self.skip_last_peva)
        
        deltas_gt = state_g["deltas"]
        first_pose = state_g["first_pose"] # B, 1, 48
        xsens_offsets = state_g["xsens_offsets"]
        skel = XsensSkeleton(xsens_offsets)
        
        pred_actions = get_action_smpl_torch(first_pose, pred_actions, XSensConstants.upper_body_num_parts) # B, T, 48
        gt_actions = get_action_smpl_torch(first_pose, deltas_gt, XSensConstants.upper_body_num_parts) # B, T, 48
        eval_metrics = _compute_pose_and_loss(pred_actions[:, -1], gt_actions[:, -1], skel)
        
        start_distances = _compute_pose_and_loss(first_pose[:, -1], gt_actions[:, -1], skel)
        leaf_start_distances = {k: v for k, v in start_distances.items() if any(x in k.lower() for x in ["head", "hand", "pelvis"])}
        leaf_start_avg_xyz_distance = sum([v.mean().item() for k, v in leaf_start_distances.items() if "xyz" in k.lower()]) / len(leaf_start_distances)
        leaf_start_avg_angular_distance = sum([v.mean().item() for k, v in leaf_start_distances.items() if "angular" in k.lower()]) / len(leaf_start_distances)
        print(f"start-leaf-xyz_distance: {leaf_start_avg_xyz_distance}")
        print(f"start-leaf-angular_distance: {leaf_start_avg_angular_distance}")
        
        res = {}
        for k, v in eval_metrics.items():
            part_name = k.split("-")[0]
            part_idx = XSensConstants.part_names.index(part_name)
            if any(x in k.lower() for x in ["head", "hand" "pelvis"]):
                leaf_key = "leaf-" + k.split("-")[1]
                if leaf_key not in res: res[leaf_key] = []
                res[leaf_key].append(v)
                
                if all(goal_image_coords[0, part_idx] != -1):
                    visible_key = "visible_leaf-" + k.split("-")[1]
                    if visible_key not in res: res[visible_key] = []
                    res[visible_key].append(v)
        for k, v in res.items():
            res[k] = torch.cat(v, dim=0).mean().item()
            print(f"{k}: {res[k]}")
        # for k, v in eval_metrics.items():
        #     res[k] = v.mean().item()
        res["start_leaf-xyz_distance"] = leaf_start_avg_xyz_distance
        res["start_leaf-angular_distance"] = leaf_start_avg_angular_distance
        return res