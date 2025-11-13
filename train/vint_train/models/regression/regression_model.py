import os
import argparse
import time
import pdb

import torch
import torch.nn as nn

from vint_train.models.nomad.nomad_vint import NoMaD_ViNT


class RegressionModel(nn.Module):
    def __init__(self, vision_encoder: NoMaD_ViNT, output_dim: int):
        super(RegressionModel, self).__init__()
        self.vision_encoder = vision_encoder
        self.output_dim = output_dim
        self.regression_head = nn.Linear(self.vision_encoder.obs_encoding_size, self.output_dim)
    
    def forward(self, 
                obs_img,
                goal_img,
                context_poses):
        obs_encoding = self.vision_encoder(obs_img, goal_img, input_goal_mask=None,context_poses=context_poses)
        return self.regression_head(obs_encoding)



