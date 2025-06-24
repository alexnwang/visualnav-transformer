import os
import tqdm
from vint_train.data.misc import XSensConstants
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn


from vint_train.data.vint_dataset import ViNT_Nymeria_Dataset
import json


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                dataset = ViNT_Nymeria_Dataset(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[data_split_type],
                    dataset_name=dataset_name,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    context_type=config["context_type"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=False,
                    goal_type=config["goal_type"],
                )
                if data_split_type == "train":
                    train_dataset.append(dataset)
                else:
                    dataset_type = f"{dataset_name}_{data_split_type}"
                    if dataset_type not in test_dataloaders:
                        test_dataloaders[dataset_type] = {}
                    test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True if config["num_workers"] > 0 else False,
    )
            
    import matplotlib.pyplot as plt

    num_batches = len(train_loader)
    xyz_list = []
    rpy_lists = {part: [] for part in XSensConstants.part_names[:XSensConstants.upper_body_num_parts]}
    print("done setup")
    train_loader = tqdm.tqdm(train_loader, desc="Loading batches", total=num_batches)
    for batch_idx, data in enumerate(train_loader):
        if num_batches > 0 and batch_idx >= num_batches:
            break
        # print(f"Processing batch {batch_idx + 1}/{num_batches}")
        (
            obs_image,
            goal_image,
            deltas,
            distance,
            goal_pos,
            dataset_idx,
            action_mask,
            first_pose,
        ) = data

        pelvis_xyz = deltas[:, :, :3].reshape(-1, 3).cpu().numpy()
        xyz_list.append(pelvis_xyz)

        for idx, part_name in enumerate(XSensConstants.part_names[:XSensConstants.upper_body_num_parts]):
            rpy = deltas[:, :, 3 + idx * 3:3 + (idx + 1) * 3].reshape(-1, 3).cpu().numpy()
            rpy_lists[part_name].append(rpy)

    xyz_all = np.concatenate(xyz_list, axis=0)
    rpy_all = {part: np.concatenate(rpy_lists[part], axis=0) for part in rpy_lists}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(xyz_all[:, 0], bins=50, alpha=0.7)
    axes[0].set_title('Pelvis X Distribution')
    axes[1].hist(xyz_all[:, 1], bins=50, alpha=0.7)
    axes[1].set_title('Pelvis Y Distribution')
    axes[2].hist(xyz_all[:, 2], bins=50, alpha=0.7)
    axes[2].set_title('Pelvis Z Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(config["project_folder"], "pelvis_xyz_distribution.png"))
    plt.close()

    for part, rpy in rpy_all.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].hist(rpy[:, 0], bins=50, alpha=0.7)
        axes[0].set_title(f'{part} Roll Distribution')
        axes[1].hist(rpy[:, 1], bins=50, alpha=0.7)
        axes[1].set_title(f'{part} Pitch Distribution')
        axes[2].hist(rpy[:, 2], bins=50, alpha=0.7)
        axes[2].set_title(f'{part} Yaw Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(config["project_folder"], f"{part}_rpy_distribution.png"))
        plt.close()
            
    
        # Compute mean and variance for pelvis xyz
        xyz_mean = xyz_all.mean(axis=0)
        xyz_var = xyz_all.var(axis=0)

        # Compute mean and variance for each part's rpy
        rpy_stats = {}
        for part, rpy in rpy_all.items():
            rpy_stats[part] = {
                "mean": rpy.mean(axis=0).tolist(),
                "var": rpy.var(axis=0).tolist(),
            }

        stats = {
            "pelvis_xyz": {
                "mean": xyz_mean.tolist(),
                "var": xyz_var.tolist(),
            },
            "rpy": rpy_stats,
        }

        with open(os.path.join(config["project_folder"], "mean_var_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
            
        # Plot normalized distributions for pelvis xyz
        xyz_norm = (xyz_all - xyz_mean) / np.sqrt(xyz_var)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].hist(xyz_norm[:, 0], bins=50, alpha=0.7)
        axes[0].set_title('Normalized Pelvis X')
        axes[1].hist(xyz_norm[:, 1], bins=50, alpha=0.7)
        axes[1].set_title('Normalized Pelvis Y')
        axes[2].hist(xyz_norm[:, 2], bins=50, alpha=0.7)
        axes[2].set_title('Normalized Pelvis Z')
        plt.tight_layout()
        plt.savefig(os.path.join(config["project_folder"], "pelvis_xyz_normalized_distribution.png"))
        plt.close()

        # Plot normalized distributions for each part's rpy
        for part, rpy in rpy_all.items():
            mean = np.array(rpy_stats[part]["mean"])
            var = np.array(rpy_stats[part]["var"])
            rpy_norm = (rpy - mean) / np.sqrt(var)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].hist(rpy_norm[:, 0], bins=50, alpha=0.7)
            axes[0].set_title(f'{part} Normalized Roll')
            axes[1].hist(rpy_norm[:, 1], bins=50, alpha=0.7)
            axes[1].set_title(f'{part} Normalized Pitch')
            axes[2].hist(rpy_norm[:, 2], bins=50, alpha=0.7)
            axes[2].set_title(f'{part} Normalized Yaw')
            plt.tight_layout()
            plt.savefig(os.path.join(config["project_folder"], f"{part}_rpy_normalized_distribution.png"))
            plt.close()
    
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)
    
    config["project_name"] = "nomad_analysis"
    config["run_name"] += "analysis"
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
        exist_ok=True
    )
    config['num_workers'] = 12  # set to 0 for debugging, change to >0 for training

    # if config["use_wandb"]:
    #     # wandb.login()
    #     wandb.init(
    #         project=config["project_name"],
    #         settings=wandb.Settings(start_method="fork"),
    #         entity="alexandernwang", # TODO: change this to your wandb entity
    #     )
    #     wandb.save(args.config, policy="now")  # save the config file
    #     wandb.run.name = config["run_name"]
    #     # update the wandb args with the training configurations
    #     if wandb.run:
    #         wandb.config.update(config)

    print(config)
    main(config)
