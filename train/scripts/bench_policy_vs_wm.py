"""One-time benchmark of wallclock + FLOPs for one policy call vs one PEVA WM call.

Run from train/ with: python -m scripts.bench_policy_vs_wm
"""
import argparse
import sys
import os

import torch
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.utils import load_peva, load_policy
from planning.sampling import peva_sample, policy_sample


DEFAULT_NOMAD_CONFIG = "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_03_22_01_13:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask/config.yaml"
DEFAULT_NOMAD_CKPT   = "/home/anw2067/visualnav-transformer/train/logs/nomad-minimal/2026_03_22_01_13:nomad-minimal-proprioception-cat8-dinov3_unpool_3dposemb-proj-lr5e-4-pool_curr_obs-goaldraw-waypointMask/ema_9.pth"
DEFAULT_PEVA_CONFIG  = "/home/anw2067/visualnav-transformer/train/peva/config/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_-64to_64_1_goal_emb_relative_xxl.yaml"
DEFAULT_PEVA_CKPT    = "/scratch/anw2067/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_cancel_scaler_-64to_64_xxl_280_0180000.pth.tar"


def cuda_time(fn, n_warmup, n_iter):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_iter):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))
    return torch.tensor(times)


def profile_flops(fn, n_warmup=1):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_flops=True, record_shapes=False) as prof:
        fn()
    torch.cuda.synchronize()
    return sum(int(getattr(e, "flops", 0) or 0) for e in prof.key_averages())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8, help="batch size (== CEM num_samples)")
    p.add_argument("--peva_diffusion_steps", type=int, default=64)
    p.add_argument("--peva_context_size", type=int, default=7)
    p.add_argument("--ar_horizon", type=int, default=None,
                   help="WM autoregressive horizon (default = policy len_traj_pred, usually 8)")
    p.add_argument("--nomad_config",     type=str, default=DEFAULT_NOMAD_CONFIG)
    p.add_argument("--nomad_checkpoint", type=str, default=DEFAULT_NOMAD_CKPT)
    p.add_argument("--peva_config",      type=str, default=DEFAULT_PEVA_CONFIG)
    p.add_argument("--peva_checkpoint",  type=str, default=DEFAULT_PEVA_CKPT)
    p.add_argument("--n_iter_policy",  type=int, default=5)
    p.add_argument("--n_iter_wm",      type=int, default=3)
    p.add_argument("--skip_flops",     action="store_true")
    args = p.parse_args()

    device = "cuda"
    B = args.n

    # ---- load policy (uses its trained num_diffusion_iters) ----
    policy, policy_diffusion, _, nomad_config = load_policy(args.nomad_config, args.nomad_checkpoint, device=device)
    image_size         = nomad_config["image_size"][0]
    policy_ctx         = nomad_config["context_size"] + 1
    policy_pred_horizon = nomad_config["len_traj_pred"]
    policy_action_dim  = nomad_config["input_dims"]
    policy_diff_steps  = nomad_config["num_diffusion_iters"]

    # ---- load WM (64 denoise steps, 7 context frames) ----
    peva_model, _, peva_diffusion, peva_vae, peva_stats, peva_config = load_peva(
        args.peva_config, args.peva_checkpoint, device=device,
        inference_context_size=args.peva_context_size,
        diffusion_steps=args.peva_diffusion_steps,
    )
    peva_latent_size = image_size // 8
    peva_ctx = peva_config["context_size"]
    ar_horizon = args.ar_horizon or policy_pred_horizon

    # ---- dummy inputs ----
    obs_imgs       = torch.randn(B, policy_ctx, 3, image_size, image_size, device=device)
    goal_img       = torch.randn(B, 3, image_size, image_size, device=device)
    context_poses  = torch.randn(B, policy_ctx, 48, device=device)
    goal_type      = nomad_config.get("goal_type", None)
    if goal_type in ("2d", "2d5050"):
        goal_coordinates = torch.rand(B, 8, device=device) * image_size
    elif goal_type in ("3d5050",):
        goal_coordinates = torch.randn(B, 12, device=device)
    else:
        goal_coordinates = None

    wm_curr_obs = torch.rand(B, peva_ctx, 3, image_size, image_size, device=device)
    wm_deltas   = torch.randn(B, ar_horizon, 48, device=device) * 0.01

    def run_policy():
        with torch.no_grad():
            return policy_sample(policy, policy_diffusion,
                                 obs_imgs, goal_img, context_poses,
                                 policy_pred_horizon, policy_action_dim, device,
                                 goal_coordinates=goal_coordinates)

    def run_wm():
        with torch.no_grad():
            return peva_sample(peva_model, peva_diffusion, peva_vae, peva_stats,
                               wm_curr_obs, wm_deltas,
                               peva_ctx, peva_latent_size,
                               image_size, device)

    n_policy_params = sum(p.numel() for p in policy.parameters())
    n_peva_params   = sum(p.numel() for p in peva_model.parameters())
    n_vae_params    = sum(p.numel() for p in peva_vae.parameters())

    print(f"[config] B={B}  image_size={image_size}")
    print(f"         policy: ctx={policy_ctx}  pred_horizon={policy_pred_horizon}  diff_steps={policy_diff_steps}  goal_type={goal_type}")
    print(f"         peva:   ctx={peva_ctx}  ar_horizon={ar_horizon}  diff_steps={args.peva_diffusion_steps}  (denoising calls = {ar_horizon * args.peva_diffusion_steps})")
    print(f"[params] policy={n_policy_params/1e6:.1f}M  peva={n_peva_params/1e6:.1f}M  vae={n_vae_params/1e6:.1f}M")

    # Compile + cudnn-autotune warmup so the first timed iter doesn't pay the JIT cost
    print("\n[warmup] one untimed run of each (compile / autotune)")
    run_policy(); run_wm(); torch.cuda.synchronize()

    print("\n=== Wall-clock (one full sample call) ===")
    t_p = cuda_time(run_policy, n_warmup=2, n_iter=args.n_iter_policy)
    print(f"Policy: {t_p.mean().item():7.1f} ms (std {t_p.std().item():.2f}, n={args.n_iter_policy})")
    t_w = cuda_time(run_wm,     n_warmup=2, n_iter=args.n_iter_wm)
    print(f"WM:     {t_w.mean().item():7.1f} ms (std {t_w.std().item():.2f}, n={args.n_iter_wm})")
    print(f"Ratio WM / Policy: {t_w.mean().item() / t_p.mean().item():.1f}x")

    print(f"\n[peak GPU mem] {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    if not args.skip_flops:
        print("\n=== FLOPs (torch.profiler, with_flops=True) ===")
        print("Note: only matmul/conv/addmm are counted; treat as a lower bound, comparable across the two.")
        f_p = profile_flops(run_policy)
        print(f"Policy: {f_p/1e9:8.1f} GFLOPs")
        f_w = profile_flops(run_wm)
        print(f"WM:     {f_w/1e9:8.1f} GFLOPs")
        if f_p > 0:
            print(f"Ratio WM / Policy: {f_w / f_p:.1f}x")


if __name__ == "__main__":
    main()
