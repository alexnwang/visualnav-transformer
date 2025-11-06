import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from vint_train.models.vint.self_attention import PositionalEncoding
import timm

class NoMaD_ViNT(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        pool_features: Optional[bool] = True,
        image_size: Optional[Tuple[int, int]] = (224, 224),
        proprioception: Optional[bool] = False,
    ) -> None:
        """
        NoMaD ViNT Encoder class
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size
        self.pool_features = pool_features
        self.proprioception = proprioception
        
        if "efficientnet" in obs_encoder:
            # Initialize the observation encoder
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.encoder_type = "efficientnet"
            # Initialize the goal encoder
            self.goal_encoder = EfficientNet.from_name(obs_encoder, in_channels=6) # obs+goal
            self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
            self.num_goal_features = self.goal_encoder._fc.in_features
        elif "dinov3-s" in obs_encoder:
            model = timm.create_model('vit_small_plus_patch16_dinov3.lvd1689m', pretrained=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            self.encoder = model
            self.num_goal_features = self.num_obs_features = model.num_features
            self.encoder_type = obs_encoder
        elif "resnet50" in obs_encoder:
            model = timm.create_model('resnet50', pretrained=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            self.encoder = model
            self.num_goal_features = self.num_obs_features = model.num_features
            self.encoder_type = obs_encoder
        else:
            raise ValueError(f"Invalid encoder type: {obs_encoder}")

        # Initialize compression layers if necessary
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        # Initialize positional encoding and self-attention layers
        if self.pool_features:
            self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2)
        else:
            num_patches = (image_size[0] // 32) * (image_size[1] // 32)
            self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=(self.context_size + 2) * num_patches)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)
        
        if self.proprioception:
            self.proprioception_encoder = nn.Linear(45, obs_encoding_size)

        # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
        self.goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True # Mask out the goal 
        self.no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool) 
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat([1 - self.no_mask.float(), (1 - self.goal_mask.float()) * ((self.context_size + 2)/(self.context_size + 1))], dim=0)


    def extract_features(self, img: torch.tensor, mode="obs" or "goal") -> torch.tensor:
        """
        Extract features from the image using the encoder.
        Args:
            img: torch.tensor, the image to extract features from.
            mode: str, "obs" or "goal".
        Returns:
            torch.tensor, the extracted features. Shape: N, L, D, L=1 if pool_features=True
        """
        assert mode in ["obs", "goal"], "Invalid mode"
        if self.encoder_type == "efficientnet":
            encoder = self.goal_encoder if mode == "goal" else self.obs_encoder
            compress_enc = self.compress_goal_enc if mode == "goal" else self.compress_obs_enc
            encoding = encoder.extract_features(img)
            if self.pool_features:
                encoding = encoder._avg_pooling(encoding)
            encoding = encoding.flatten(start_dim=2)
            if encoder._global_params.include_top:
                encoding = encoder._dropout(encoding)
            encoding = encoding.permute(0, 2, 1) # N, L, D
            encoding = compress_enc(encoding)
            return encoding
        elif "dinov3-s" in self.encoder_type:
            encoder = self.encoder 
            compress_enc = self.compress_obs_enc if mode == "obs" else self.compress_goal_enc
            with torch.no_grad():
                if self.pool_features:
                    encoding = encoder(img)[:, None]
                else:
                    encoding = encoder.forward_features(img)
            encoding = compress_enc(encoding)
            return encoding
        elif "resnet50" in self.encoder_type:
            encoder = self.encoder 
            compress_enc = self.compress_obs_enc if mode == "obs" else self.compress_goal_enc
            with torch.no_grad():
                if self.pool_features:
                    encoding = encoder.forward_features(img)
                    encoding = encoder.global_pool(encoding)[:, None]
                else:
                    encoding = encoder.forward_features(img).flatten(start_dim=2)
                encoding = encoding.permute(0, 2, 1) # N, L, D
            encoding = compress_enc(encoding)
            return encoding

    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor,
                input_goal_mask: torch.tensor = None,
                context_poses: torch.tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = obs_img.device
        
        if self.proprioception:
            assert context_poses is not None, "Context poses are required for proprioception"
            if context_poses.shape[-1] == 48:
                context_poses = context_poses[:, :, 3:] # B, C+1, 45
            assert context_poses.shape[-1] == 45, "Context poses must have 45 dimensions"
            encoded_context_pose = self.proprioception_encoder(context_poses) # B, C+1, self.obs_encoding_size
        
        # Get the input goal mask 
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)

        # Get the goal encoding
        obsgoal_img = torch.cat([obs_img[:, self.context_size], goal_img], dim=1) if self.encoder_type == "efficientnet" else goal_img
        goal_encoding = self.extract_features(obsgoal_img, mode="goal")
        
        # Get the observation encoding
        B, Cplus1 = obs_img.shape[:2]
        obs_img = obs_img.flatten(0, 1) # B*(C+1), 3, *image_size
        obs_encoding = self.extract_features(obs_img, mode="obs").unflatten(0, (B, Cplus1)) # B, (C+1), L, self.obs_encoding_size
        L = obs_encoding.shape[2]
        if self.proprioception:
            obs_encoding = obs_encoding + encoded_context_pose[:, :, None]
        
        obs_encoding = obs_encoding.flatten(1, 2) # flatten to B, (C+1)*L, self.obs_encoding_size, L=1 if pool_features=True
        obs_encoding = torch.cat((obs_encoding, goal_encoding), dim=1) # B, (C+2)*L, self.obs_encoding_size; L = 1 if pool_features=True
        
        # If a goal mask is provided, mask some of the goal tokens
        if goal_mask is not None:
            no_goal_mask = goal_mask.long()
            src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, no_goal_mask) # B, C+2
            if not self.pool_features:
                src_key_padding_mask = src_key_padding_mask[..., None].repeat(1, 1, L).flatten(1, 2) # B, C+2, 1 -> B, C+2, L -> B, (C+2)*L
        else:
            src_key_padding_mask = None
        
        # Apply positional encoding 
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)

        obs_encoding_tokens = self.sa_encoder(obs_encoding, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            avg_mask = torch.index_select(self.avg_pool_mask.to(device), 0, no_goal_mask).unsqueeze(-1)
            if not self.pool_features:
                avg_mask = avg_mask.repeat(1, 1, L).flatten(1, 2)[..., None] / L
            obs_encoding_tokens = obs_encoding_tokens * avg_mask
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)
        return obs_encoding_tokens

# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module



    