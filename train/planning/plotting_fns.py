import os
import torch
from torchvision.utils import save_image

def save_action_obs_sequence_viz(
    save_path: str,
    goal_image: torch.Tensor,
    curr_obs: torch.Tensor,
    goal_obs: torch.Tensor,
    top_seq: torch.Tensor,
    bot_seq: torch.Tensor,
    gt_goal_image: torch.Tensor = None,
):
    """
    Save a 2-row action-and-observation sequence visualization.

    Layout (T+2 columns):
      top: [goal_image | top_seq[0] ... top_seq[T-1] | gt_goal_image or blank ]
      bot: [curr_obs   | bot_seq[0] ... bot_seq[T-1] | goal_obs               ]

    The first/last columns are always real (possibly annotated) images.
    Middle columns are caller-defined: e.g. skeleton overlays + GT frames for
    ground-truth display, or predicted annotations + WM generations for planning.

    Args:
        save_path:      output file path
        goal_image:     (3, H, W) — input to policy; curr obs with waypoints if available
        curr_obs:       (3, H, W) — current observation, no annotations
        goal_obs:       (3, H, W) — raw goal observation
        top_seq:        (T, 3, H, W) — annotated frames for top row
        bot_seq:        (T, 3, H, W) — observation frames for bottom row
        gt_goal_image:  (3, H, W) — optional; curr obs with GT waypoints, placed top-right
    """
    T = top_seq.shape[0]
    top_right = gt_goal_image if gt_goal_image is not None else torch.zeros_like(curr_obs)
    top_row = torch.cat([goal_image[None], top_seq, top_right[None]], dim=0)   # (T+2, 3, H, W)
    bot_row = torch.cat([curr_obs[None],   bot_seq, goal_obs[None]], dim=0)    # (T+2, 3, H, W)
    grid = torch.cat([top_row, bot_row], dim=0)                                # (2*(T+2), 3, H, W)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    save_image(grid, save_path, nrow=T + 2)
