"""Figure: visualize the CEM *sample-and-score* step for waypoint planning.

This SIMULATES the CEM sampling step (it does not re-run the ~hours-long search).
For each of N iterations the mean is FIXED (not CEM-updated):
  - iter 0:  center-stacked waypoints (all 4 leaf coords at 0.5)
  - iter 1:  a loaded planned-solution mu from a finished plan_cem run
            (--source_log_dir / results.pth, step --solution_step)
Per iteration we draw K random waypoint samples with per-iteration variance,
push each through the policy + PEVA world model, and score the final imagined
frame with DreamSIM against the goal.

Per sample we save (overlays on the high-res 1408 curr frame; WM at native 224):
  waypoints.png         (1) the sampled waypoints drawn on curr_obs
  actions_skin.webp     (2) the policy action sequence (SMPL mesh + skeleton)
  wm_generations.webp   (3) the PEVA imagined rollout (low-res)
  final_frame.png       (4) the last WM frame
The DreamSIM score is encoded in the sample dir name: sampleN_ds{score}.
GIFs are produced afterwards by converting the .webp files with ImageMagick.

Single task only: pass --track/--curr_time/--goal_time (defaults to the
thomas_brown talk example).
"""
import argparse, os, pickle, random, shutil, subprocess, sys, tempfile

_TRAIN_DIR = "/home/anw2067/visualnav-transformer/train"
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)
os.chdir(_TRAIN_DIR)

import numpy as np
import torch
from einops import repeat
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image

from peva.diffusion import create_diffusion
from plan_cem import build_waypoint_cem, MODEL_DIRECTORY
from planning.cem import move_to_device
from planning.nymeria_dataset import NymeriaPlanningDataset
from planning.wrappers import Preprocessor, build_skeleton_top_seq
from planning.utils import _compute_part_distance_matrices
from planning.vis_utils import disable_logging
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.training.nymeria_training_utils import get_action_smpl_torch

from paper_figure_generations.task_figure_viz import save_png, save_webp
from paper_figure_generations.figure_planning_highres import search_waypoints_frame


def score_mje(deltas, first_pose, deltas_gt, xsens_offsets):
    """Final-pose joint error vs GT per sample: returns (all_xyz mean (K,), leaf_xyz (K,))."""
    K = deltas.shape[0]
    skel = XsensSkeleton(xsens_offsets)                      # (15, 3)
    fp = first_pose.expand(K, -1, -1).contiguous()           # (K, 1, 48)
    gt = deltas_gt.expand(K, -1, -1).contiguous()            # (K, T, 48)
    pred = get_action_smpl_torch(fp, deltas, XSensConstants.upper_body_num_parts)
    gta = get_action_smpl_torch(fp, gt, XSensConstants.upper_body_num_parts)
    xyz, _, leaf, _ = _compute_part_distance_matrices(pred[:, -1], gta[:, -1], skel)
    return xyz.mean(dim=-1), leaf


def webp_to_gif(webp_path, magick_bin):
    """Convert an animated .webp to .gif with ImageMagick (-coalesce keeps full frames)."""
    gif_path = webp_path[:-5] + ".gif"
    subprocess.run([magick_bin, webp_path, "-coalesce", gif_path], check=True)
    print("wrote", gif_path, flush=True)


def render_iteration(args, cem_planner, name, mu, sigma, P, curr_hr, render_size,
                     trans_obs_0, trans_obs_g, first_pose, xsens_offsets, deltas_gt,
                     fisheye_params, R_C_pelvis, t_C_pelvis, out_dir, device, webp_paths):
    """Sample K waypoints ~ N(mu, sigma); for each, keep the best-MJE of P policy
    draws (P>1) or a single draw (P<=1); PEVA-roll and render every sample."""
    K = args.num_samples
    H = mu.shape[1]
    action_dim = mu.shape[2]
    step_ms = round(1000 / args.fps * args.slow)

    # --- draw K noised waypoints (cf. planning/cem.py: randn * sigma + mu) ---
    act = torch.randn(K, H, action_dim, device=device) * sigma + mu  # (K, H, 8)
    if args.clamp_samples:
        act = act.clamp(0.0, 1.0)

    if P <= 1:
        # one policy draw per waypoint; batched policy + PEVA over all K
        batched0 = {k: repeat(v, "1 ... -> n ...", n=K) for k, v in trans_obs_0.items()}
        with torch.no_grad():
            rollout = cem_planner.wm.rollout(batched0, act)
        generated_obs = rollout["generated_obs"]       # (K, T, 3, 224, 224)
        deltas = rollout["deltas"]                      # (K, T, 48)
        ma, ml = score_mje(deltas, first_pose, deltas_gt, xsens_offsets)
        mje_all, mje_leaf = ma.cpu().tolist(), ml.cpu().tolist()
    else:
        # best-of-P: per waypoint draw P policy actions (policy-only, no PEVA),
        # keep the lowest final-pose-MJE one; then batched-PEVA the K winners.
        best_deltas, best_goal, mje_all, mje_leaf = [], [], [], []
        for k in range(K):
            b0 = {key: repeat(v, "1 ... -> n ...", n=P) for key, v in trans_obs_0.items()}
            wp_k = repeat(act[k:k + 1], "1 ... -> n ...", n=P)
            with torch.no_grad():
                ps = cem_planner.wm.policy_only_rollout(b0, wp_k)
            dP = ps["deltas"]                           # (P, T, 48)
            ax, lx = score_mje(dP, first_pose, deltas_gt, xsens_offsets)   # (P,)
            bi = int(torch.argmin(ax).item())
            best_deltas.append(dP[bi:bi + 1])
            gi = ps.get("goal_images")
            best_goal.append(gi[bi:bi + 1] if gi is not None else None)
            mje_all.append(ax[bi].item()); mje_leaf.append(lx[bi].item())
            print(f"    [{name}] wp{k}: best MJE {ax[bi].item():.3f} of {P} "
                  f"(pool mean {ax.mean().item():.3f})", flush=True)
        deltas = torch.cat(best_deltas, dim=0)          # (K, T, 48)
        goal_images = None if any(g is None for g in best_goal) else torch.cat(best_goal, dim=0)
        bK = {key: repeat(v, "1 ... -> n ...", n=K) for key, v in trans_obs_0.items()}
        with torch.no_grad():
            wms = cem_planner.wm.wm_only_rollout(bK, deltas, goal_images=goal_images)
        generated_obs = wms["generated_obs"]            # (K, T, 3, 224, 224)
    T = generated_obs.shape[1]

    # --- DreamSIM(final imagined frame, goal) per sample (reuse the planner's model) ---
    ds_model = cem_planner.objective_fn.model
    ds_pre = cem_planner.objective_fn.preprocess
    goal_emb = ds_pre(transforms.ToPILImage()(trans_obs_g["images"][0].cpu())).to(device)
    ds_scores = []
    with torch.no_grad():
        for i in range(K):
            pred_emb = ds_pre(transforms.ToPILImage()(generated_obs[i, -1].cpu())).to(device)
            ds_scores.append(ds_model(pred_emb, goal_emb).item())

    iterdir = os.path.join(out_dir, name)
    # iteration reference: the fixed mean drawn on the curr frame (no rollout)
    save_png(os.path.join(iterdir, "mean_waypoints.png"),
             search_waypoints_frame(curr_hr, mu, render_size))

    for i in range(K):
        sdir = os.path.join(iterdir, f"sample{i}_mje{mje_all[i]:.3f}_leaf{mje_leaf[i]:.3f}_ds{ds_scores[i]:.3f}")
        # (1) sampled waypoints on curr_obs (high-res)
        save_png(os.path.join(sdir, "waypoints.png"),
                 search_waypoints_frame(curr_hr, act[i:i + 1], render_size))
        # (2) policy action sequence: SMPL mesh + skeleton on curr_obs (high-res)
        seq = build_skeleton_top_seq(
            curr_hr, deltas[i:i + 1].to(device), first_pose, xsens_offsets,
            fisheye_params, R_C_pelvis, t_C_pelvis, render_size, T,
            overlay="both", smpl_alpha=args.smpl_alpha, show_text=False)
        wp = os.path.join(sdir, "actions_skin.webp")
        save_webp(wp, list(seq), [step_ms] * len(seq)); webp_paths.append(wp)
        # (3) PEVA imagined rollout (low-res)
        gen = generated_obs[i]                  # (T, 3, 224, 224)
        wp = os.path.join(sdir, "wm_generations.webp")
        save_webp(wp, list(gen), [step_ms] * gen.shape[0]); webp_paths.append(wp)
        # (4) final WM frame
        save_png(os.path.join(sdir, "final_frame.png"), gen[-1])

    print(f"  [{name}] DreamSIM: {[round(s, 3) for s in ds_scores]}  "
          f"MJE: {[round(m, 3) for m in mje_all]}  leafMJE: {[round(m, 3) for m in mje_leaf]}", flush=True)


def main(args):
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(0)
    device = "cuda"
    disable_logging()

    # ---- planner built ONCE (loads nomad policy + PEVA world model) ----
    args.algo = "waypoint"
    cem_planner, nomad_config, _ = build_waypoint_cem(args, None, args.log_dir, device)
    data_config = nomad_config["datasets"]["nymeria"]
    context_size = max(args.peva_context_size - 1, nomad_config["context_size"])

    # ---- single-task dataset via temp pkl ----
    with tempfile.NamedTemporaryFile("wb", suffix=".pkl", delete=False) as tf:
        pickle.dump([{"track": args.track, "curr_time": args.curr_time,
                      "goal_time": args.goal_time}], tf)
        tasks_file = tf.name
    ds = NymeriaPlanningDataset(
        tasks_file=tasks_file, data_folder=data_config["data_folder"],
        image_size=nomad_config["image_size"], context_size=context_size,
        goal_type=nomad_config.get("goal_type", None),
        waypoint_spacing=data_config.get("waypoint_spacing", 1),
        gaussian_normalization_stats_path=data_config.get("gaussian_normalization_stats_path", None),
    )
    batch = next(iter(DataLoader(Subset(ds, [0]), batch_size=1, shuffle=False, num_workers=0)))
    os.unlink(tasks_file)

    track = batch["dataset_track"][0]
    start_index = batch["start_index"].item()
    goal_index = batch["goal_index"].item()
    task_name = f"{track}-s{start_index}-g{goal_index}"
    print(f"[cem-sampling] {task_name}", flush=True)

    # ---- camera params (resolution-independent) ----
    cam = torch.load(os.path.join(args.camera_data_folder, track, "camera_data.pt"), weights_only=False)
    T_mat = cam["T_C_pelvis"][start_index]
    fisheye_params = cam["fisheye_params"].to(device).float()
    R_C_pelvis = T_mat[:3, :3].to(device).float()
    t_C_pelvis = T_mat[:3, 3].to(device).float()

    obs_0 = {"images": batch["obs_images"], "goal_image": batch["goal_image"],
             "context_poses": batch["context_poses"]}
    obs_g = {"images": batch["goal_obs"], "deltas": batch["deltas"],
             "first_pose": batch["first_pose"], "xsens_offsets": batch["xsens_offsets"],
             "goal_image_coords": batch["goal_image_coords"]}
    trans_obs_0 = move_to_device(Preprocessor().transform_obs(obs_0), device)
    trans_obs_g = move_to_device(Preprocessor().transform_obs(obs_g), device)

    first_pose = batch["first_pose"].to(device)            # (1, 1, 48)
    xsens_offsets = batch["xsens_offsets"][0].to(device)   # (15, 3)
    deltas_gt = batch["deltas"].to(device)                 # (1, T, 48)

    # ---- high-fidelity WM diffusion for the figure rollouts ----
    if args.peva_vis_diffusion_steps != args.peva_diffusion_steps:
        cem_planner.wm.peva_diffusion = create_diffusion(str(args.peva_vis_diffusion_steps))

    # ---- high-res curr/goal frames ----
    hr_curr = args.highres_frame or os.path.join(args.highres_frames_dir, track, f"{start_index}.png")
    hr_goal = args.highres_goal_frame or os.path.join(args.highres_frames_dir, track, f"{goal_index}.png")
    curr_hr = transforms.ToTensor()(Image.open(hr_curr).convert("RGB")).to(device)
    render_size = curr_hr.shape[-1]

    out_dir = os.path.join(args.log_dir, task_name, args.run_tag) if args.run_tag \
        else os.path.join(args.log_dir, task_name)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(hr_curr, os.path.join(out_dir, "current_obs.png"))
    if os.path.exists(hr_goal):
        shutil.copyfile(hr_goal, os.path.join(out_dir, "goal_obs.png"))
    else:
        save_png(os.path.join(out_dir, "goal_obs.png"), batch["goal_obs"][0])

    # ---- iteration means from --iters specs ("center:SIGMA" or "STEP:SIGMA") ----
    H = args.horizon
    mu_center = torch.full((1, H, 8), 0.5, device=device)
    mu_hist = None
    iters = []
    for idx, spec in enumerate(args.iters):
        parts = spec.split(":")
        step_str, sigma = parts[0], float(parts[1])
        P = int(parts[2]) if len(parts) > 2 else args.policy_pool   # best-of-P per waypoint
        ptag = f"_P{P}" if P > 1 else ""
        if step_str == "center":
            iters.append((f"iter{idx}_center_s{sigma}{ptag}", mu_center, sigma, P))
        else:
            if mu_hist is None:
                mu_hist = torch.load(f"{args.source_log_dir}/{task_name}/results.pth",
                                     weights_only=False)["mu_history"]
            si = int(step_str) % len(mu_hist)
            label = f"iter{idx}_mu{si}_s{sigma}{ptag}"
            print(f"  {label}: mean = mu_history[{si}] from {os.path.basename(args.source_log_dir)}", flush=True)
            iters.append((label, mu_hist[si].to(device), sigma, P))

    webp_paths = []
    for it, (name, mu, sigma, P) in enumerate(iters):
        torch.manual_seed(seed + it)                        # distinct draws per iteration
        render_iteration(args, cem_planner, name, mu, sigma, P, curr_hr, render_size,
                         trans_obs_0, trans_obs_g, first_pose, xsens_offsets, deltas_gt,
                         fisheye_params, R_C_pelvis, t_C_pelvis, out_dir, device, webp_paths)

    # ---- convert every webp -> gif with ImageMagick (done afterwards, not via PIL) ----
    if args.make_gif:
        magick_bin = shutil.which("magick") or shutil.which("convert")
        if magick_bin is None:
            print("WARNING: ImageMagick not found; skipping gif conversion", flush=True)
        else:
            for wp in webp_paths:
                webp_to_gif(wp, magick_bin)

    print(f"done: {task_name} -> {out_dir}", flush=True)


def build_argparser():
    p = argparse.ArgumentParser()
    # ---- single task (defaults: thomas_brown talk example) ----
    p.add_argument("--track", default="20231110_s0_thomas_brown_act4_x3t73z")
    p.add_argument("--curr_time", type=int, default=2959)
    p.add_argument("--goal_time", type=int, default=2967)
    p.add_argument("--source_log_dir",
                   default="/home/anw2067/visualnav-transformer/train/logs/cem/"
                           "2026_05_31_02_49_18:waypoint_cem-h1-n64-t8-v0.3-o8-N64-ds64-dist8-8",
                   help="finished plan_cem.py run (talk_8tasks); iter-1 mean = its results.pth mu_history[--solution_step]")
    p.add_argument("--log_dir", default="/home/anw2067/visualnav-transformer/train/logs/talk_planning_example")
    p.add_argument("--run_tag", default="",
                   help="optional subdir under {log_dir}/{task_name}/ to isolate this run's outputs")
    p.add_argument("--highres_frame", default=None, help="explicit high-res curr png (overrides derived path)")
    p.add_argument("--highres_goal_frame", default=None, help="explicit high-res goal png")
    p.add_argument("--highres_frames_dir",
                   default="/home/anw2067/visualnav-transformer/train/logs/highres_frames",
                   help="dir of {track}/{idx}.png high-res frames")
    p.add_argument("--camera_data_folder", default="/scratch/anw2067/nymeria_visibility_matrix")

    # ---- sampling ----
    # Each --iters entry is "MEAN:SIGMA"; MEAN is "center" or a mu_history step index.
    p.add_argument("--iters", nargs="+", default=["center:0.3", "5:0.05"],
                   help='per-iteration "MEAN:SIGMA" specs, e.g. center:0.3 5:0.05 6:0.0')
    # NOTE: --num_samples doubles as build_waypoint_cem's (unused) CEM batch size.
    p.add_argument("-K", "--num_samples", type=int, default=4, help="random waypoint samples per iteration")
    p.add_argument("--policy_pool", type=int, default=1,
                   help="default best-of-P policy pool per waypoint when an --iters spec omits :P "
                        "(P=1 = single draw, no MJE selection)")
    p.add_argument("-H", "--horizon", type=int, default=1)
    p.add_argument("--clamp_samples", action="store_true",
                   help="clamp samples to [0,1] (default off: out-of-image waypoints masked as the real planner does)")
    p.add_argument("--seed", type=int, default=42)

    # ---- rendering ----
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--slow", type=float, default=1.0, help="multiply webp frame duration (>1 = slow-mo)")
    p.add_argument("--smpl_alpha", type=float, default=0.9)
    p.add_argument("--no_gif", dest="make_gif", action="store_false", help="skip the ImageMagick webp->gif step")

    # ---- CEM args consumed by build_waypoint_cem (we don't run the search) ----
    p.add_argument("--use_leafxyz_as_cost", action="store_true")
    p.add_argument("-t", "--topk", type=int, default=8)
    p.add_argument("-v", "--var_scale", type=float, default=0.3)
    p.add_argument("-o", "--opt_steps", type=int, default=8)
    p.add_argument("-N", "--num_eval_samples", type=int, default=1)

    # ---- models ----
    p.add_argument("--peva_config", default="/home/anw2067/visualnav-transformer/train/peva/config/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_-64to_64_1_goal_emb_relative_xxl.yaml")
    p.add_argument("--peva_checkpoint", default="/scratch/anw2067/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_cancel_scaler_-64to_64_xxl_280_0180000.pth.tar")
    p.add_argument("--peva_context_size", type=int, default=7)  # match the talk planning runs
    p.add_argument("--peva_diffusion_steps", type=int, default=64)
    p.add_argument("--peva_vis_diffusion_steps", type=int, default=250)
    p.add_argument("--nomad_model", default="draw_mask", choices=list(MODEL_DIRECTORY.keys()))
    p.add_argument("--nomad_config", default=None)
    p.add_argument("--nomad_checkpoint", default=None)

    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--rank", type=int, default=0)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.nomad_model is not None:
        assert args.nomad_config is None and args.nomad_checkpoint is None
        args.nomad_config, args.nomad_checkpoint = MODEL_DIRECTORY[args.nomad_model]
    main(args)
