import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

"""
IMPORT YOUR MODEL HERE
"""
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from vint_train.data.vint_dataset import ViNT_Dataset, ViNT_Nymeria_Dataset
from vint_train.training.train_eval_loop import (
    # train_eval_loop,
    train_eval_loop_nomad,
    load_model,
)


def init_distributed(port=37124, rank_and_world_size=(None, None)):
    rank, world_size = rank_and_world_size
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(port))
    print("Using port", os.environ['MASTER_PORT'])

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            gpu = int(os.environ["LOCAL_RANK"])
        except Exception:
            print('torchrun env vars not sets')

    elif "SLURM_PROCID" in os.environ:
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            gpu = rank % torch.cuda.device_count()
            if 'HOSTNAME' in os.environ:
                os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
            else:
                os.environ['MASTER_ADDR'] = '127.0.0.1'
        except Exception:
            print('SLURM vars not set')
    
    else:
        rank = 0
        world_size = 1
        gpu = 0
        os.environ['MASTER_ADDR'] = '127.0.0.1'

    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        device_id=gpu,
    )
    print("initialized distributed process group")

    # setup_for_distributed(rank == 0)
    return world_size, rank, gpu, True


def main(rank, world_size, config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"Using DDP with {world_size} GPUs")
        print(f"Master process rank: {rank}")

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        # cudnn.deterministic = True

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

    assert len(config['datasets']) == 1, "Currently only one dataset is supported to enable unnormalization of rpy angles for graphing."
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
                    normalize=config["normalize"],
                    goal_type=config["goal_type"],
                    gaussian_normalization_stats_path= data_config.get("gaussian_normalization_stats_path", None),
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

    # Use DistributedSampler for training data
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,  # Use DistributedSampler instead of shuffle
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=False
        )
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            sampler=sampler,  # Use DistributedSampler instead of shuffle
            num_workers=config['num_workers'],
            drop_last=False,
            persistent_workers=False
        )

    print("Creating model...")
    # Create the model
    vision_encoder = NoMaD_ViNT(
        obs_encoder=config["obs_encoder"],
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        pool_features=config.get("pool_features", True),
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    
    global_cond_dim = config["encoding_size"]
    if config.get("proprioception", False):
        global_cond_dim += 45
    
    noise_pred_net = ConditionalUnet1D(
            input_dim=config['input_dims'],
            global_cond_dim=global_cond_dim,
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    # Move model to device first
    model = model.to(device)
    print(f"model moved to device {device}, wrapping with ddp")

    # Wrap model with DDP
    model_without_ddp = model
    model = DDP(model, device_ids=[rank])
    print(f"rank {rank} ddp model created")
    model._set_static_graph()

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"] * len(train_loader)
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=int(config["warmup_epochs"]*len(train_loader)),
                after_scheduler=scheduler,
            )

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in model: {num_params}")
        if config["model_type"] == "nomad":
            num_vision_encoder_params = sum(p.numel() for p in vision_encoder.parameters() if p.requires_grad)
            num_noise_pred_net_params = sum(p.numel() for p in noise_pred_net.parameters() if p.requires_grad)
            num_dist_pred_net_params = sum(p.numel() for p in dist_pred_network.parameters() if p.requires_grad)
            print(f"Number of trainable parameters in vision_encoder: {num_vision_encoder_params}")
            print(f"Number of trainable parameters in noise_pred_net: {num_noise_pred_net_params}")
            print(f"Number of trainable parameters in dist_pred_network: {num_dist_pred_net_params}")

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path, map_location=device)
        load_model(model.module, config["model_type"], latest_checkpoint)  # Use model.module for DDP
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1
        else:
            epochs = [x for x in os.listdir(load_project_folder) if "epoch_" in x]
            current_epoch = max(int(epoch.split("_")[-1]) for epoch in epochs)
        print(f"Loaded model from epoch {current_epoch}")

        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    # Set epoch for DistributedSampler
    train_sampler.set_epoch(current_epoch)

    train_eval_loop_nomad(
        train_model=config["train"],
        model=model,
        proprioception=config.get("proprioception", False),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        noise_scheduler=noise_scheduler,
        train_loader=train_loader,
        test_dataloaders=test_dataloaders,
        transform=transform,
        goal_mask_prob=config["goal_mask_prob"],
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        print_log_freq=config["print_log_freq"],
        wandb_log_freq=config["wandb_log_freq"],
        image_log_freq=config["image_log_freq"],
        num_images_log=config["num_images_log"],
        current_epoch=current_epoch,
        alpha=float(config["alpha"]),
        use_wandb=config["use_wandb"],
        eval_fraction=config["eval_fraction"],
        eval_freq=config["eval_freq"],
        rank=rank,
    )

    if rank == 0:
        print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer with DDP")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    parser.add_argument(
        "--world_size",
        "-w",
        default=torch.cuda.device_count(),
        type=int,
        help="Number of GPUs to use",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)
    world_size, rank, gpu, _ = init_distributed()

    # config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    # for ddp cannot use seconds.
    config["run_name"] = time.strftime("%Y_%m_%d_%H_%M") + ":" + config["run_name"]
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    
    if rank == 0:
        os.makedirs(
            config[
                "project_folder"
            ],  # should error if dir already exists to avoid overwriting and old project,
            exist_ok=True,
        )

        print(config)
    
    
    if config["use_wandb"] and rank == 0:
        # wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
            entity="alexandernwang", # TODO: change this to your wandb entity
        )
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)
    
    torch.cuda.set_device(gpu)
    
    main(rank, world_size, config)


