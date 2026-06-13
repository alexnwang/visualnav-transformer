"""High-res paper figures for a single PLANNING (CEM) result, rendered on the
1408 current-obs frame.

Unlike figure_gt_and_pred.py (which samples the policy directly), this runs the
full CEM planner (waypoint or peva), picks the best CEM step by per-step LEAF MJE
(`best_leaf`), then — for waypoint — selects the best of R stochastic policy
rollouts of that mu by final-pose LEAF MJE (peva is deterministic, single rollout).

Per task, under --log_dir/{algo}/{track-sC-gG}/:
  current_obs.png                         high-res curr frame (item 1, copied)
  goal_obs.png                            high-res goal frame  (item 2)
  noskin/generated_actions.webp           planned-action skeleton on curr   (item 3)
  skin/generated_actions.webp             planned-action SMPL mesh on curr  (item 3)
  wm_generations.webp                     world-model imagined rollout, LOW-RES (item 4)
  gt_waypoints.png      (waypoint only)   GT goal-pose leaf joints on curr  (item 5)
  search_waypoints.png  (waypoint only)   searched mu waypoints on curr     (item 6)

The CEM search / policy / PEVA all run at the model's native 224; the resulting
deltas + normalized waypoints are resolution-independent and rendered on the 1408
frame. WM generations are kept at their native (low) resolution per the spec.
"""
import argparse, os, pickle, random, sys, shutil, tempfile

_TRAIN_DIR = "/home/anw2067/visualnav-transformer/train"
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)
os.chdir(_TRAIN_DIR)

import numpy as np
import torch
from einops import repeat
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageDraw

from peva.diffusion import create_diffusion
from plan_cem import build_waypoint_cem, build_peva_cem, build_action_init, MODEL_DIRECTORY
from plan_cem_viz import compute_waypoint_err, multi_rollout_topk
from planning.cem import move_to_device
from planning.nymeria_dataset import NymeriaPlanningDataset, _LEAF_IDX
from planning.wrappers import Preprocessor, build_skeleton_top_seq
from planning.utils import _compute_part_distance_matrices
from planning.vis_utils import draw_image_coords, disable_logging, add_dark_glow
from vint_train.data.misc import XSensConstants, XsensSkeleton
from vint_train.training.nymeria_training_utils import get_action_smpl_torch

from paper_figure_generations.task_figure_viz import save_png, save_webp, waypoints_frame


def _score_leaf(pred_deltas, first_pose, deltas_gt, skel):
    """Return (all_xyz, leaf_xyz) length-B tensors for final-pose match vs GT."""
    pred = get_action_smpl_torch(first_pose, pred_deltas, XSensConstants.upper_body_num_parts)
    gt = get_action_smpl_torch(first_pose, deltas_gt, XSensConstants.upper_body_num_parts)
    xyz_dist, _, leaf_xyz, _ = _compute_part_distance_matrices(pred[:, -1], gt[:, -1], skel)
    return xyz_dist.mean(dim=-1), leaf_xyz


def best_rollout_by_leaf(wm, state_0, state_g, mu, num_rollouts, device, algo):
    """Roll mu out and return the single best rollout's WM state, selected by leaf MJE.

    waypoint: R cheap policy-only rollouts -> pick argmin leaf -> one expensive WM rollout.
    peva:     single deterministic rollout (num_rollouts ignored).
    Returns dict {generated_obs(1,T,3,h,w), deltas(1,T,48), goal_images|None, mje, leaf_xyz}.
    """
    first_pose = state_g["first_pose"]      # 1,1,48
    deltas_gt = state_g["deltas"]           # 1,T,48
    skel = XsensSkeleton(state_g["xsens_offsets"][0])

    if algo == "peva":
        with torch.no_grad():
            wm_state = wm.rollout(state_0=state_0, act=mu)
        all_xyz, leaf_xyz = _score_leaf(wm_state["deltas"], first_pose, deltas_gt, skel)
        return {
            "generated_obs": wm_state["generated_obs"],
            "deltas": wm_state["deltas"],
            "goal_images": wm_state.get("goal_images"),
            "mje": all_xyz[0].item(), "leaf_xyz": leaf_xyz[0].item(),
        }

    batched_state_0 = {k: repeat(v, "1 ... -> n ...", n=num_rollouts) for k, v in state_0.items()}
    batched_mu = repeat(mu, "1 ... -> n ...", n=num_rollouts)
    fp_rep = repeat(first_pose, "1 ... -> n ...", n=num_rollouts)
    gt_rep = repeat(deltas_gt, "1 ... -> n ...", n=num_rollouts)

    with torch.no_grad():
        policy_state = wm.policy_only_rollout(state_0=batched_state_0, act=batched_mu)
    pred_deltas = policy_state["deltas"]            # N,T,48
    all_xyz, leaf_xyz = _score_leaf(pred_deltas, fp_rep, gt_rep, skel)
    best = int(torch.argmin(leaf_xyz).item())      # <-- select by LEAF, not full-body MJE

    sel_state_0 = {k: v[best:best + 1] for k, v in batched_state_0.items()}
    sel_deltas = pred_deltas[best:best + 1]
    gi = policy_state["goal_images"]
    sel_goal_images = gi[best:best + 1] if gi is not None else None
    with torch.no_grad():
        wm_state = wm.wm_only_rollout(sel_state_0, sel_deltas, goal_images=sel_goal_images)
    return {
        "generated_obs": wm_state["generated_obs"],
        "deltas": wm_state["deltas"],
        "goal_images": wm_state.get("goal_images"),
        "mje": all_xyz[best].item(), "leaf_xyz": leaf_xyz[best].item(),
    }


def search_waypoints_frame(curr_image, mu, render_size, marker_scale=1.8, glow=True,
                           glow_color=(0, 0, 0)):
    """Draw the searched mu waypoints (4 leaf coords in [0,1]) on the high-res frame.

    mu: (1, H, 8) normalized waypoint coords; uses H=0 (single-step). Out-of-image
    waypoints (outside (0,1)) are masked to (-1,-1) so absent ones are not drawn.
    Leaf colors match draw_image_coords (red/green/blue/yellow). `marker_scale`
    multiplies the dot size; `glow=True` lays a soft dark halo behind each dot."""
    wp = mu.reshape(-1, 4, 2)[0].clone()                 # (4, 2) in [0,1]
    present = ((wp > 0) & (wp < 1)).all(dim=-1)
    coords = torch.full((1, XSensConstants.upper_body_num_parts, 2), -1.0, device=curr_image.device)
    for k, j in enumerate(XSensConstants.leaf_indices):
        if present[k]:
            coords[0, j] = wp[k].to(curr_image.device) * render_size
    img = Image.fromarray((curr_image.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy())
    scale = render_size / 224.0 * marker_scale
    if glow:
        img = add_dark_glow(img, coords, scale=scale, glow_color=glow_color)
    draw_image_coords(ImageDraw.Draw(img), coords, show_text=False, scale=scale, dot_stroke=0.3)
    return transforms.ToTensor()(img).to(curr_image.device)


def render_one(args, algo, cem_planner, nomad_config, data_config, context_size,
               action_init, device, track, curr_time, goal_time,
               highres_frame, highres_goal_frame, log_dir):
    # ---- single-task dataset via temp pkl ----
    with tempfile.NamedTemporaryFile("wb", suffix=".pkl", delete=False) as tf:
        pickle.dump([{"track": track, "curr_time": curr_time, "goal_time": goal_time}], tf)
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
    print(f"[{algo}] planning {task_name}", flush=True)

    # ---- camera params (resolution-independent) ----
    cam = torch.load(os.path.join(args.camera_data_folder, track, "camera_data.pt"), weights_only=False)
    T_mat = cam["T_C_pelvis"][start_index]
    fisheye_params = cam["fisheye_params"]
    R_C_pelvis = T_mat[:3, :3]
    t_C_pelvis = T_mat[:3, 3]

    obs_0 = {"images": batch["obs_images"], "goal_image": batch["goal_image"],
             "context_poses": batch["context_poses"]}
    obs_g = {"images": batch["goal_obs"], "deltas": batch["deltas"],
             "first_pose": batch["first_pose"], "xsens_offsets": batch["xsens_offsets"],
             "goal_image_coords": batch["goal_image_coords"]}

    # ---- obtain the CEM plan (mu_history + per-step leaf) ----
    if args.source_log_dir:
        # Reuse a finished plan_cem.py run: skip the (~hour/task) CEM search.
        results = torch.load(f"{args.source_log_dir}/{task_name}/results.pth", weights_only=False)
        mu_history = results["mu_history"]                     # (opt_steps, 1, H, action_dim)
        leaf_per_step = results.get("eval_metrics", {}).get("leaf_xyz", [])
        print(f"  loaded saved plan: mu_history={tuple(mu_history.shape)}", flush=True)
    else:
        cem_planner.plan(obs_0, obs_g, task_name, actions=action_init,
                         fisheye_params=fisheye_params, R_C_pelvis=R_C_pelvis,
                         t_C_pelvis=t_C_pelvis, render_skin=False)
        results = torch.load(f"{log_dir}/{task_name}/results.pth", weights_only=False)
        mu_history = results["mu_history"]
        leaf_per_step = cem_planner.accum_eval_metric_dicts.get(task_name, {}).get("leaf_xyz", [])

    # ---- per-step metrics: pick the best-MJE step and the best-waypoint-L2 step ----
    if args.source_log_dir:
        all_xyz_per_step = results.get("eval_metrics", {}).get("all_xyz", [])
    else:
        all_xyz_per_step = cem_planner.accum_eval_metric_dicts.get(task_name, {}).get("all_xyz", [])
    imsz = nomad_config["image_size"][0]
    gt_leaf_coords = batch["goal_image_coords"][0, _LEAF_IDX]
    wp_per_step = [compute_waypoint_err(mu_history[i], gt_leaf_coords, imsz) for i in range(len(mu_history))]
    wp_arr = np.where(np.isnan(wp_per_step), np.inf, wp_per_step)
    best_mje_step = int(np.argmin(all_xyz_per_step)) if len(all_xyz_per_step) else len(mu_history) - 1
    best_wp_step = int(np.argmin(wp_arr)) if np.isfinite(wp_arr).any() else best_mje_step
    print(f"  best_mje step={best_mje_step} (mje={all_xyz_per_step[best_mje_step]:.4f})  "
          f"best_wp step={best_wp_step} (wpL2={wp_per_step[best_wp_step]:.4f})", flush=True)

    # ---- high-res frame + camera params ----
    curr_hr = transforms.ToTensor()(Image.open(highres_frame).convert("RGB")).to(device)
    render_size = curr_hr.shape[-1]
    first_pose = batch["first_pose"].to(device)
    xsens_offsets = batch["xsens_offsets"][0].to(device)
    deltas = batch["deltas"].to(device)                # (1, T, 48)
    n_steps = deltas.shape[1]
    R_C_pelvis = R_C_pelvis.to(device).float()
    t_C_pelvis = t_C_pelvis.to(device).float()
    fisheye_params = fisheye_params.to(device).float()
    step_ms = round(1000 / args.fps * args.slow)

    trans_obs_0 = move_to_device(Preprocessor().transform_obs(obs_0), device)
    trans_obs_g = move_to_device(Preprocessor().transform_obs(obs_g), device)
    if args.peva_vis_diffusion_steps != args.peva_diffusion_steps:
        cem_planner.wm.peva_diffusion = create_diffusion(str(args.peva_vis_diffusion_steps))

    out = os.path.join(log_dir, task_name)
    os.makedirs(out, exist_ok=True)

    # ---- shared per-track items: current_obs, goal_obs, GT waypoints ----
    shutil.copyfile(highres_frame, f"{out}/current_obs.png")
    if highres_goal_frame and os.path.exists(highres_goal_frame):
        shutil.copyfile(highres_goal_frame, f"{out}/goal_obs.png")
    else:
        save_png(f"{out}/goal_obs.png", batch["goal_obs"][0])
        print("  WARNING: no high-res goal frame; saved low-res goal_obs", flush=True)
    goal_pose = get_action_smpl_torch(first_pose, deltas, XSensConstants.upper_body_num_parts)[:, -1]
    save_png(f"{out}/gt_waypoints.png",
             waypoints_frame(curr_hr, goal_pose, R_C_pelvis, t_C_pelvis, fisheye_params,
                             xsens_offsets, render_size))

    def render_skin_action(deltas_b, path):
        seq = build_skeleton_top_seq(curr_hr, deltas_b, first_pose, xsens_offsets,
                                     fisheye_params, R_C_pelvis, t_C_pelvis,
                                     render_size, n_steps, overlay="both",
                                     smpl_alpha=args.smpl_alpha, show_text=False)
        save_webp(path, list(seq), [step_ms] * len(seq))

    # ---- 2 mu-variants (best-MJE step, best-wpL2 step) x top-3 rollouts by MJE ----
    for vname, vstep in [("best_mje", best_mje_step), ("best_wp", best_wp_step)]:
        mu = mu_history[vstep].to(device)
        saved_states, _ = multi_rollout_topk(
            cem_planner.wm, trans_obs_0, trans_obs_g, mu,
            num_rollouts=args.num_vis_rollouts, device=device, algo=algo,
            top_k_save=3, topk_wm=3)
        vdir = os.path.join(out, f"{vname}_step{vstep}")
        save_png(f"{vdir}/search_waypoints.png", search_waypoints_frame(curr_hr, mu, render_size))
        for ri, ss in enumerate(saved_states):
            rdir = f"{vdir}/rollout{ri}_mje{ss['mje']:.3f}"
            render_skin_action(ss["state"]["deltas"].to(device), f"{rdir}/skin/generated_actions.webp")
            gen = ss["state"]["generated_obs"][0]      # (T, 3, h, w) low-res WM rollout
            save_webp(f"{rdir}/wm_generations.webp", list(gen), [step_ms] * gen.shape[0])
        print(f"  [{vname}] step={vstep} top3 mje={[round(s['mje'], 3) for s in saved_states]}", flush=True)

    print(f"done: {task_name} -> {out}", flush=True)


def main(args):
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(0)
    device = "cuda"
    disable_logging()

    algo = args.algo
    log_dir = os.path.join(args.log_dir, algo)
    os.makedirs(log_dir, exist_ok=True)

    # ---- planner built ONCE (loads nomad policy + PEVA world model) ----
    if algo == "waypoint":
        action_init = build_action_init("waypoint", args.horizon, args.waypoint_init)
        cem_planner, nomad_config, _ = build_waypoint_cem(args, None, log_dir, device)
    elif algo == "peva":
        action_init = None
        cem_planner, nomad_config, _ = build_peva_cem(args, None, log_dir, device)
    else:
        raise ValueError(algo)

    data_config = nomad_config["datasets"]["nymeria"]
    context_size = max(args.peva_context_size - 1, nomad_config["context_size"])

    # ---- task list: explicit tasks_file, or a single --track/--curr_time/--goal_time ----
    if args.tasks_file:
        tasks = pickle.load(open(args.tasks_file, "rb"))
    else:
        tasks = [{"track": args.track, "curr_time": args.curr_time, "goal_time": args.goal_time}]

    for t in tasks:
        track, curr_time, goal_time = t["track"], t["curr_time"], t["goal_time"]
        # high-res frame paths: explicit overrides, else derived from --highres_frames_dir
        hr_curr = args.highres_frame or os.path.join(args.highres_frames_dir, track, f"{curr_time}.png")
        hr_goal = args.highres_goal_frame or os.path.join(args.highres_frames_dir, track, f"{goal_time}.png")
        if not os.path.exists(hr_curr):
            print(f"SKIP {track}-s{curr_time}-g{goal_time}: missing high-res curr {hr_curr}", flush=True)
            continue
        try:
            render_one(args, algo, cem_planner, nomad_config, data_config, context_size,
                       action_init, device, track, curr_time, goal_time, hr_curr, hr_goal, log_dir)
        except Exception as e:
            import traceback
            print(f"ERROR on {track}-s{curr_time}-g{goal_time}: {e}", flush=True)
            traceback.print_exc()

    print(f"\nALL DONE [{algo}] -> {log_dir}", flush=True)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("-a", "--algo", choices=["peva", "waypoint"], default="waypoint")
    # Either pass a --tasks_file (loops all tasks) or a single --track/--curr_time/--goal_time.
    p.add_argument("--tasks_file", default=None, help="pkl list of {track,curr_time,goal_time}; loops all")
    p.add_argument("--track", default=None)
    p.add_argument("--curr_time", type=int, default=None)
    p.add_argument("--goal_time", type=int, default=None)
    p.add_argument("--highres_frame", default=None, help="explicit 1408 curr-obs png (single-task override)")
    p.add_argument("--highres_goal_frame", default=None, help="explicit 1408 goal-obs png (single-task override)")
    p.add_argument("--highres_frames_dir",
                   default="/home/anw2067/visualnav-transformer/train/logs/highres_frames",
                   help="dir of {track}/{idx}.png high-res frames; used to derive curr/goal frames per task")
    p.add_argument("--log_dir", default="/home/anw2067/visualnav-transformer/train/logs/planning_figure_hr")
    p.add_argument("--source_log_dir", default=None,
                   help="finished plan_cem.py logs/cem/... dir; loads saved mu_history and SKIPS the CEM search")

    # CEM params (match talk8 planning runs: o8, n64, t8, v(0.3 wp/0.5 peva), N(64 wp/1 peva))
    p.add_argument("--use_leafxyz_as_cost", action="store_true")
    p.add_argument("-n", "--num_samples", type=int, default=64)
    p.add_argument("-t", "--topk", type=int, default=8)
    p.add_argument("-v", "--var_scale", type=float, default=0.3)
    p.add_argument("-o", "--opt_steps", type=int, default=8)
    p.add_argument("-H", "--horizon", type=int, default=1)
    p.add_argument("-N", "--num_eval_samples", type=int, default=64)
    p.add_argument("-R", "--num_vis_rollouts", type=int, default=64)
    p.add_argument("--waypoint_init", type=str, choices=["center", "empirical"], default="center",
                   help="CEM init mean for waypoint: 'center' (0.5) or 'empirical' (dist8 leaf-waypoint mean). "
                        "Only affects re-planning; ignored when --source_log_dir loads a saved plan.")

    # rendering
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--slow", type=float, default=1.0,
                   help="multiply webp frame duration (1.0 = real-time at --fps; >1 = slow-mo)")
    p.add_argument("--smpl_alpha", type=float, default=0.9)
    p.add_argument("--skin_only", action="store_true",
                   help="render only skin/generated_actions.webp (drop the noskin skeleton webp)")
    p.add_argument("--camera_data_folder", default="/scratch/anw2067/nymeria_visibility_matrix")

    # models
    p.add_argument("--peva_config", default="/home/anw2067/visualnav-transformer/train/peva/config/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_-64to_64_1_goal_emb_relative_xxl.yaml")
    p.add_argument("--peva_checkpoint", default="/scratch/anw2067/nymeria_rel_concat_embedding_compile_beta095_ar_model_context_16_bs_16_smpl_lowebody_cancel_scaler_-64to_64_xxl_280_0180000.pth.tar")
    p.add_argument("--peva_context_size", type=int, default=15)
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
