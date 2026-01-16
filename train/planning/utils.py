import torch
import yaml
import copy
import wandb
import json

from torchvision import transforms
from dreamsim import dreamsim

from diffusers.models import AutoencoderKL

from peva.models import CDiT_models
from peva.diffusion import create_diffusion
from scipy.spatial.transform import Rotation as R
import numpy as np

from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.models.nomad.conditional_uned1dnomad import ConditionalUnet1D_NoMaD
from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.data.vint_dataset import ViNT_Nymeria_Dataset
from vint_train.training.nymeria_training_utils import forward_kinematics_wrapper, get_action_smpl_torch
from planning.cem import CEMPlanner

from torchvision.utils import save_image
from torchvision.utils import draw_keypoints

def _compute_pose_and_loss(actn, gt_actn, skel, actn_mask=None):
        gt_xyz, gt_rpy = forward_kinematics_wrapper(gt_actn, skel, XSensConstants.upper_body_num_parts, return_euler=True) # B, num_segments, 3
        pred_xyz, pred_rpy = forward_kinematics_wrapper(actn, skel, XSensConstants.upper_body_num_parts, return_euler=True) # B, num_segments, 3
        res = {}
        for i, body_part_name in enumerate(XSensConstants.part_names[:XSensConstants.upper_body_num_parts]):
            R_gt = R.from_euler('xyz', gt_rpy[:, i, :].detach().cpu().numpy(), degrees=False)
            R_pred = R.from_euler('xyz', pred_rpy[:, i, :].detach().cpu().numpy(), degrees=False)
            ang_dist = torch.from_numpy((R_gt.inv() * R_pred).magnitude() / np.pi * 180).to(actn.device).float() # B
            xyz_dist = torch.norm(gt_xyz[:, i, :] - pred_xyz[:, i, :], dim=-1) # B
            if actn_mask is not None:
                ang_dist = ang_dist * actn_mask
                xyz_dist = xyz_dist * actn_mask
            res[f"{body_part_name}-angular_distance"] = ang_dist
            res[f"{body_part_name}-xyz_distance"] = xyz_dist
        return res

LEAF_INDICES = torch.tensor([XSensConstants.part_names.index(x) for x in ["Pelvis", "Head", "R_Hand", "L_Hand"]])
def _compute_part_distance_matrices(pred_actn, gt_actn, skel, aggregate_indices=LEAF_INDICES):
    """
    Args:
        pred_actn: B, 48
        gt_actn: B, 48
        skel: XsensSkeleton
        aggregate_indices: 1D torch.tensor of indices to aggregate, or None to not aggregate
    
    Returns:
        xyz_dist_matrix: B, XSensConstants.num_parts
        ang_dist_matrix: B, XSensConstants.num_parts
    """
    B = pred_actn.shape[0]
    num_parts = XSensConstants.upper_body_num_parts
    xyz_dist_matrix = torch.zeros(B, num_parts, device=pred_actn.device)
    ang_dist_matrix = torch.zeros(B, num_parts, device=pred_actn.device)
    
    gt_xyz, gt_rpy = forward_kinematics_wrapper(gt_actn, skel, XSensConstants.upper_body_num_parts, return_euler=True) # B, num_segments, 3
    pred_xyz, pred_rpy = forward_kinematics_wrapper(pred_actn, skel, XSensConstants.upper_body_num_parts, return_euler=True) # B, num_segments, 3
    
    for i in range(XSensConstants.upper_body_num_parts):
        R_gt = R.from_euler('xyz', gt_rpy[:, i, :].detach().cpu().numpy(), degrees=False)
        R_pred = R.from_euler('xyz', pred_rpy[:, i, :].detach().cpu().numpy(), degrees=False)
        ang_dist = torch.from_numpy((R_gt.inv() * R_pred).magnitude() / np.pi * 180).to(pred_actn.device).float() # B
        xyz_dist = torch.norm(gt_xyz[:, i, :] - pred_xyz[:, i, :], dim=-1) # B
        xyz_dist_matrix[:, i] = xyz_dist
        ang_dist_matrix[:, i] = ang_dist
        
    if aggregate_indices is not None:
        leaf_xyz = xyz_dist_matrix[:, aggregate_indices].mean(dim=-1).to(pred_actn.device)
        ang_xyz = ang_dist_matrix[:, aggregate_indices].mean(dim=-1).to(pred_actn.device)
        return xyz_dist_matrix, ang_dist_matrix, leaf_xyz, ang_xyz
    return xyz_dist_matrix, ang_dist_matrix

def draw_waypoints(obs, waypoints, color_order=["red", "green", "blue", "yellow"]):
    """
    Draws waypoint as circles on an image
    
    Args:
        obs: B, 3, H, W 
        waypoints: B, 8 or B, 4, 2
        color_order: list of colors
    """
    B = obs.shape[0]
    
    device = obs.device
    output_images = []
    if waypoints.shape[-1] == 8:
        waypoints = waypoints.reshape(B, 4, 2)
    for b in range(B):
        image = torch.clone(obs[b]) 
        for index, color in enumerate(color_order):
            image = draw_keypoints(image, waypoints[b, index:index+1, None, :], colors=color, radius=4)
        output_images.append(image)
    return torch.stack(output_images, dim=0).to(device)

def get_nymeria_dataset(config, context_size=15-1, split="test", goal_timestep_offset=None):
    """
    Utility for building the Nymeria dataset for CEM planning.
    Notably, does not normalize loaded deltas.
    """
    data_config = config["datasets"]["nymeria"]
    
    if "waypoint_spacing" not in data_config:
        data_config["waypoint_spacing"] = 1
    if "negative_goals" not in data_config:
        data_config["negative_goals"] = False
    if "end_slack" not in data_config:
        data_config["end_slack"] = 0
    if "goals_per_obs" not in data_config:
        data_config["goals_per_obs"] = 1
    if goal_timestep_offset is not None:
        config['distance']['min_dist_cat'] = goal_timestep_offset
        config['distance']['max_dist_cat'] = goal_timestep_offset

    ### EVAL ONLY -- DO NOT NORMALIZE THE DELTAS
    config["normalize"] = False
    
    # transform = ([
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # transform = transforms.Compose(transform)
    transform = torch.nn.Identity()
    
    if context_size is None:
        context_size = config["context_size"]
    
    dataset = ViNT_Nymeria_Dataset(
        data_folder=data_config["data_folder"],
        data_split_folder=data_config[split],
        dataset_name="nymeria",
        image_size=config["image_size"],
        transform=transform,
        waypoint_spacing=data_config["waypoint_spacing"],
        min_dist_cat=config["distance"]["min_dist_cat"],
        max_dist_cat=config["distance"]["max_dist_cat"],
        min_action_distance=config["action"]["min_dist_cat"],
        max_action_distance=config["action"]["max_dist_cat"],
        negative_goals=data_config["negative_goals"],
        len_traj_pred=config["len_traj_pred"],
        context_size=context_size,
        goal_type=config.get("goal_type", None),
        preserve_pose_up_down=data_config.get("preserve_pose_up_down", False),
        end_slack=data_config["end_slack"],
        goals_per_obs=data_config["goals_per_obs"],
        normalize=config["normalize"],
        gaussian_normalization_stats_path=data_config["gaussian_normalization_stats_path"],
    )
    return dataset

def load_peva(peva_config_file, peva_checkpoint, diffusion_steps=250, inference_context_size=15, device='cpu'):
    with open(peva_config_file, 'r') as f:
        peva_config = yaml.safe_load(f)
    num_cond = peva_config['context_size']
    if inference_context_size is None:
        inference_context_size = num_cond
    else: 
        peva_config['context_size'] = inference_context_size
    
    if peva_config.get('full_body', False):
        num_head_joint_cond = 23
    else:
        num_head_joint_cond = 15
        
    model = CDiT_models[peva_config['model']](
        context_size=num_cond, 
        inference_context_size=inference_context_size,
        num_head_joint_cond=num_head_joint_cond, 
        input_size=peva_config['image_size'] // 8, 
        in_channels=4, 
        diffusion_forcing=peva_config.get('diffusion_forcing', 0), 
        is_eval=1, skip_action_embedding=peva_config.get('skip_action_embedding', True), total_feature_dim=peva_config.get('total_feature_dim', None)).to(device)
    
    try:
        model = torch.compile(model)
        print("Compiled model")
    except:
        print("Failed to compile model")
        model = model

    try:
        ckp = torch.load(peva_checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckp["ema"], strict=True)
    except:
        print(f"Checkpoint {peva_checkpoint} not found. Running with random init.")
        
    model.eval()

    diffusion = create_diffusion(str(diffusion_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    # model_without_ddp = model.module
    model_without_ddp = model
    
    peva_stats = {"min": torch.tensor([-2, -1, -1, -1, -1, -1], dtype=torch.float32, device=device)[None], # 1,  6
                  "max": torch.tensor([2, 1, 1, 1, 1, 1], dtype=torch.float32, device=device)[None]} # 1, 6
    
    return model, model_without_ddp, diffusion, vae, peva_stats, peva_config

def load_policy(nomad_config_file, nomad_checkpoint, device='cpu'):
    with open(nomad_config_file, "r") as f:
        config = yaml.safe_load(f)
        
    def get_vision_encoder():
        if config.get("goal_type", None) in ["2d", "2d5050"]:
            goal_coordinate_dims = 8
        else:
            goal_coordinate_dims = 0
        vision_encoder = NoMaD_ViNT(
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
            pool_features=config.get("pool_features", True),
            image_size=config["image_size"],
            proprioception=config.get("proprioception", False),
            project_encoding=config.get("project_encoding", False),
            pos_enc_3d=config.get("pos_enc_3d", False),
            pool_curr_obs=config.get("pool_curr_obs", False),
            goal_coordinate_dims=goal_coordinate_dims,
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)
        return vision_encoder
    
    vision_encoder = get_vision_encoder()
        
    if config.get("goal_type", None) == "cheat":
        goal_pose_dim = 48
    elif config.get("goal_type", None) == "point":
        goal_pose_dim = 4 * 3 # 3 dimensions each for (Head, LHand, RHand, Pelvis)
    else:
        goal_pose_dim = 0
    noise_pred_net = ConditionalUnet1D_NoMaD(input_dim=config['input_dims'],
                                                global_cond_dim=config["encoding_size"],
                                                down_dims=config["down_dims"],
                                                cond_predict_scale=config["cond_predict_scale"],
                                                goal_pose_dims=goal_pose_dim)
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    model = NoMaD(vision_encoder, noise_pred_net, dist_pred_network)
    noise_scheduler = DDPMScheduler(num_train_timesteps=config["num_diffusion_iters"], beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon')
    
    model = model.to(device)
    model = model.eval()

    loaded_state_dict = torch.load(nomad_checkpoint, map_location=device) 
    for key in list(loaded_state_dict.keys()):
        if "module" in key:
            loaded_state_dict[key.replace("module.", "")] = loaded_state_dict[key]
            del loaded_state_dict[key] # remove the module prefix
            
    res = model.load_state_dict(loaded_state_dict, strict=True)
    print("model loaded with: ", res)

    with open(config['datasets']['nymeria']['gaussian_normalization_stats_path'], 'r') as f:
        stats_json = json.load(f)
    stats_dict = {"mean": stats_json['pelvis_xyz']['mean'], "var": stats_json['pelvis_xyz']['var']}
    for part_name in XSensConstants.part_names[:XSensConstants.upper_body_num_parts]:
        stats_dict["mean"] += stats_json['rpy'][part_name]['mean']
        stats_dict["var"] += stats_json['rpy'][part_name]['var']
                
    nomad_stats = {
        "mean": torch.tensor(stats_dict["mean"], dtype=torch.float32)[None, None], # 1, 1, 48
        "var": torch.tensor(stats_dict["var"], dtype=torch.float32)[None, None] # 1, 1, 48
    }
    
    return model, noise_scheduler, nomad_stats, config