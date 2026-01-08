from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
import torch

class ConditionalUnet1D_NoMaD(ConditionalUnet1D):
    def __init__(self, *args, goal_pose_dims=0, **kwargs):
        if goal_pose_dims > 0:
            self.use_goal_pose = True
            kwargs["global_cond_dim"] += goal_pose_dims
        else:
            self.use_goal_pose = False
        super().__init__(*args, **kwargs)

    def forward(self, sample, timestep, local_cond=None, global_cond=None, goal_pose=None):
        if goal_pose is not None and self.use_goal_pose:
            assert global_cond is not None
            global_cond = torch.cat([global_cond, goal_pose], dim=-1)
        return super().forward(sample, timestep, local_cond, global_cond)