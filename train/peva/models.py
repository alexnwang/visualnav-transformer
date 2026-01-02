# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from peva.diffusion import create_diffusion # DF Baseline

def blockwise_spatial_mask(b, h, q_idx, kv_idx, num_frames=5, num_tokens=196):
    n = num_tokens
    
    # Iterate over frames
    for i in range(num_frames):

        start_i = i * n
        end_i = (i+1) * n
        q_mask_1 = start_i <= q_idx
        q_mask_2 = q_idx < end_i
        q_mask = q_mask_1 &  q_mask_2

        kv_idx_1 = start_i <= kv_idx
        kv_idx_2 = kv_idx < end_i
        kv_mask = kv_idx_1 & kv_idx_2

        curr_m = q_mask * kv_mask

        if i == 0:
            m = curr_m
        else:
            m = m | curr_m
    return m

def blockwise_spatial_mask_eval(b, h, q_idx, kv_idx, num_frames=5, num_tokens=196):
    n = num_tokens
    i = num_frames - 1
    start_i = i * n
    end_i = (i+1) * n
    q_mask_1 = start_i <= q_idx
    q_mask_2 = q_idx < end_i
    q_mask = q_mask_1 &  q_mask_2

    kv_idx_1 = start_i <= kv_idx
    kv_idx_2 = kv_idx < end_i
    kv_mask = kv_idx_1 & kv_idx_2

    return q_mask * kv_mask


def blockwise_temporal_mask(b, h, q_idx, kv_idx, num_frames=5, num_tokens=196):
    n = num_tokens
    # Iterate over frames
    m = None
    for i in range(num_frames):
        start_i = i * n
        end_i = start_i + n
        q_mask_1 = start_i <= q_idx
        q_mask_2 = q_idx < end_i
        q_mask = q_mask_1 &  q_mask_2
        kv_mask = kv_idx < start_i
        curr_m = q_mask * kv_mask
        if m is None:
            m = curr_m
        else:
            m = m | curr_m
    return m

def blockwise_temporal_mask_eval(b, h, q_idx, kv_idx, num_frames=5, num_tokens=196):
    n = num_tokens
    # Only last frame cross attends to past frames
    m = None
    i = num_frames - 1
    start_i = i * n
    end_i = start_i + n
    q_mask_1 = start_i <= q_idx
    q_mask_2 = q_idx < end_i
    q_mask = q_mask_1 &  q_mask_2
    kv_mask = kv_idx < start_i
    curr_m = q_mask * kv_mask
    if m is None:
        m = curr_m
    else:
        m = m | curr_m
    return m


class MHA(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        block_mask=None,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = num_heads
        self.dropout = dropout
        self.packed_proj = nn.Linear(hidden_size, hidden_size * 3, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)
        assert hidden_size % num_heads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = hidden_size // num_heads
        self.bias = bias
        if block_mask is not None:
            self.block_mask = block_mask


    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if query is key and key is value:
            result = self.packed_proj(query)
            query, key, value = torch.chunk(result, 3, dim=-1)
        else:
            q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim=0)
            if self.bias:
                q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim=0)
            else:
                q_bias, k_bias, v_bias = None, None, None
            query, key, value = F.linear(query, q_weight, q_bias), F.linear(key, k_weight, k_bias), F.linear(value, v_weight, v_bias)

        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        attn_output = flex_attention(query, key, value, block_mask=self.block_mask)

        attn_output = attn_output.transpose(1, 2).flatten(-2)
        attn_output = self.out_proj(attn_output)

        return attn_output


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
#################################################################################
#                                 Core CDiT Model                                #
#################################################################################

class CDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, adain_input_size, mlp_ratio=4.0, spatial_mask=None, temporal_mask=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MHA(hidden_size, num_heads=num_heads, bias=True, block_mask=spatial_mask, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = MHA(hidden_size, num_heads=num_heads, bias=True, block_mask=temporal_mask, **block_kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adain_input_size, 11 * hidden_size, bias=True)
        )

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c, num_cond, x_clean):
        shift_msa, scale_msa, gate_msa, shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(11, dim=-1)

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa).flatten(1,2)
        x = x + gate_msa.unsqueeze(2) * self.attn(query=x_norm, key=x_norm, value=x_norm).unflatten(1, (num_cond + 1, -1))
        
        x_cond_norm = modulate(self.norm_cond(x_clean), shift_ca_xcond, scale_ca_xcond).flatten(1,2)
        x_norm = modulate(self.norm2(x), shift_ca_x, scale_ca_x).flatten(1,2)
        x = x + gate_ca_x.unsqueeze(2) * self.cttn(query=x_norm, key=x_cond_norm, value=x_cond_norm).unflatten(1, (num_cond + 1, -1))
        
        x_norm = modulate(self.norm3(x), shift_mlp, scale_mlp)
        return x + gate_mlp.unsqueeze(2) * self.mlp(x_norm)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, adain_input_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adain_input_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x).flatten(0, 1)
        return x


class CDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        context_size=2,
        inference_context_size=None, # For using less context frames during inference, None to use context_size
        num_head_joint_cond=22,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        skip_action_embedding=True,
        max_diffusion_time=1000,
        smpl_pose=1,
        is_eval=0,
        diffusion_forcing=0,
        first_pose=0,
        total_feature_dim=None,
    ):
        super().__init__()
        self.num_actions = 2
        self.context_size = context_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        self.skip_action_embedding = skip_action_embedding
        self.max_diffusion_time = max_diffusion_time
        self.smpl_pose = smpl_pose
        self.is_eval = is_eval
        self.diffusion_forcing = diffusion_forcing
        self.first_pose = first_pose
        if self.smpl_pose == 1:
            action_dim = 3 + num_head_joint_cond * 3
        else:
            action_dim = num_head_joint_cond * 6  # xyz + rpy for each joint

        if not self.skip_action_embedding:
            individual_action_dim = total_feature_dim // action_dim
            total_feature_dim = action_dim * (individual_action_dim)
            self.time_embedder = TimestepEmbedder(total_feature_dim)
            self.t_embedder = TimestepEmbedder(total_feature_dim)
            self.y_embedder = TimestepEmbedder(individual_action_dim)
        else:
            if total_feature_dim is None:
                self.use_c_fc_embed = False
                total_feature_dim = action_dim + 2  # action features + diffusion time + relative time
            else:
                self.use_c_fc_embed = True
                pre_embed_total_feature_dim = action_dim + 2  # action features + diffusion time + relative time
                self.c_activation_fn = nn.GELU(approximate="tanh")
                self.c_fc1 = nn.Linear(pre_embed_total_feature_dim, 512)
                self.c_fc2 = nn.Linear(512, 512)
                self.c_fc_embed = nn.Linear(512, total_feature_dim)
        
        if inference_context_size is None:
            inference_context_size = context_size
        window_size = self.x_embedder.num_patches*(inference_context_size + 1)
        
        if self.is_eval == 1:
            blockwise_spatial_partial = partial(blockwise_spatial_mask_eval, num_frames=inference_context_size + 1, num_tokens=self.x_embedder.num_patches)
            blockwise_temporal_partial = partial(blockwise_temporal_mask_eval, num_frames=inference_context_size + 1, num_tokens=self.x_embedder.num_patches)
        else:
            blockwise_spatial_partial = partial(blockwise_spatial_mask, num_frames=inference_context_size + 1, num_tokens=self.x_embedder.num_patches)
            blockwise_temporal_partial = partial(blockwise_temporal_mask, num_frames=inference_context_size + 1, num_tokens=self.x_embedder.num_patches)
        spatial_mask = create_block_mask(blockwise_spatial_partial, B=1, H=1, Q_LEN=window_size, KV_LEN=window_size)
        temporal_mask = create_block_mask(blockwise_temporal_partial, B=1, H=1, Q_LEN=window_size, KV_LEN=window_size)
        self.temporal_mask = temporal_mask

        self.blocks = nn.ModuleList([CDiTBlock(hidden_size, num_heads, total_feature_dim, mlp_ratio=mlp_ratio, spatial_mask=spatial_mask, temporal_mask=temporal_mask) for _ in range(depth)])
        self.pos_embed = nn.Parameter(torch.zeros(self.context_size + 1, self.x_embedder.num_patches, hidden_size), requires_grad=True)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, adain_input_size=total_feature_dim)
        self.initialize_weights()
    
    def normalize_and_concat_features(self, y, t, rel_t):
        """Normalize all features and concatenate into 1D list"""
        # Normalize diffusion time to 0-1
        t_norm = t.float() / self.max_diffusion_time
        # Normalize relative time (assumed to be already 0-1)
        rel_t_norm = rel_t.float()
        features = torch.cat([y.flatten(2), t_norm.unsqueeze(-1), rel_t_norm.unsqueeze(-1)], dim=-1)
        return features

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)


        # Initialize action embedding:
        if not self.skip_action_embedding:
            nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
            nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)
                
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward


    def forward(self, x, t, y, num_cond,rel_t, x_cond=None, t_cond=None, x_clean=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if x_cond is not None:
            x = torch.cat([x_cond, x.unsqueeze(1)], dim=1)
            x = x.flatten(0, 1)

            t = torch.cat([t_cond, t.unsqueeze(1)], dim=1)
            t = t.flatten(0, 1)

        x = self.x_embedder(x)
        x =  x.unflatten(0, (-1, num_cond + 1)) + self.pos_embed[:num_cond+1]

        if x_clean is not None:
            x_clean = self.x_embedder(x_clean)
            x_clean = x_clean.unflatten(0, (-1, num_cond + 1)) + self.pos_embed[:num_cond+1]
        
        t =  t.unflatten(0, (-1, num_cond + 1))
        if self.first_pose == 1:
            y =  y.unflatten(0, (-1, num_cond + 1))
            rel_t =  rel_t.unflatten(0, (-1, num_cond + 1))
        elif self.first_pose == 0:
            y =  y.unflatten(0, (-1, num_cond))
            rel_t =  rel_t.unflatten(0, (-1, num_cond))
            
        if self.skip_action_embedding:
            if self.first_pose == 0:
                # for now pad first action with zeros (first image doesnt have an action):
                y = torch.cat([torch.zeros_like(y)[:, :1], y], dim=1)
                rel_t = torch.cat([torch.zeros_like(rel_t)[:, :1], rel_t], dim=1)
            
            c = self.normalize_and_concat_features(y, t, rel_t)
            if self.use_c_fc_embed:
                c = self.c_activation_fn(self.c_fc1(c))
                c = self.c_activation_fn(self.c_fc2(c))
                c = self.c_fc_embed(c)
        else:
            t = self.t_embedder(t[..., None])
            y = self.y_embedder(y.unsqueeze(-1)).flatten(-2)
            time_emb = self.time_embedder(rel_t[..., None])
            c = torch.cat([
                t.narrow(1, 0, 1),  # Keep first column unchanged
                t.narrow(1, 1, t.size(1)-1) + y + time_emb
            ], dim=1)
            
        if self.diffusion_forcing == 1:
            x_context = x
        else:
            x_context =  x_clean

        for block in self.blocks:
            if self.is_eval:
                x = block(x, c, num_cond, x_context)
            else:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, num_cond, x_context, use_reentrant=False)       # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if x_cond is not None:
            return x.unflatten(0, (-1, num_cond + 1))[:, -1]
        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   CDiT Configs                                  #
#################################################################################

def CDiT_XXL_2(**kwargs):
    return CDiT(depth=32, hidden_size=1360, patch_size=2, num_heads=16, **kwargs)

def CDiT_XL_2(**kwargs):
    return CDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def CDiT_L_2(**kwargs):
    return CDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def CDiT_B_2(**kwargs):
    return CDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def CDiT_S_2(**kwargs):
    return CDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


CDiT_models = {
    'CDiT-XXL/2': CDiT_XXL_2, 
    'CDiT-XL/2': CDiT_XL_2, 
    'CDiT-L/2':  CDiT_L_2, 
    'CDiT-B/2':  CDiT_B_2, 
    'CDiT-S/2':  CDiT_S_2
}
