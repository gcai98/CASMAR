# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Union
from timm.models.vision_transformer import PatchEmbed, Mlp
import einops
import torch.nn.functional as F

# ---------------------- FlashAttention (原逻辑保留，作为回退) ----------------------
try:
    from flash_attn_interface import flash_attn_func
    USE_FLASH_ATTN3 = True
    USE_FLASH_ATTN = True
    print("[INFO] Using flash_attn_interface (FlashAttention v3).")
except ImportError:
    try:
        from flash_attn import flash_attn_func
        USE_FLASH_ATTN3 = False
        USE_FLASH_ATTN = True
        print("[INFO] Using flash_attn (FlashAttention v2).")
    except ImportError:
        flash_attn_func = None
        USE_FLASH_ATTN = False
        print("[WARNING] flash_attn not found. Falling back to standard attention.")
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            ATTENTION_MODE = 'flash'
        else:
            try:
                import xformers
                import xformers.ops
                ATTENTION_MODE = 'xformers'
            except:
                ATTENTION_MODE = 'math'
        print(f'attention mode is {ATTENTION_MODE}')





class SALAAttention2D(nn.Module):
    """
    Scaffold-Anchored Local Attention (SALA)

    x: [B, N, C]      当前 Stage-2 高分辨率 token
    z: [B, Nz, C]     Stage-1 scaffold condition tokens

    输出:
    out: [B, N, C]

    结构:
      1) local branch: 在 x 上做局部窗口注意力
      2) anchor branch: x 查询压缩后的 scaffold anchors
      3) gated fusion: out = o_loc + g * o_anc
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        kernel_size: int = 7,
        dilation: Union[int, Tuple[int, int]] = 1,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        anchor_pool_size: int = 2,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel_size = int(kernel_size)

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.padding = (
            self.dilation[0] * (self.kernel_size - 1) // 2,
            self.dilation[1] * (self.kernel_size - 1) // 2,
        )

        self.anchor_pool_size = anchor_pool_size

        # -------- local branch --------
        self.q_proj_loc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj_loc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj_loc = nn.Linear(dim, dim, bias=qkv_bias)

        # -------- anchor branch --------
        self.q_proj_anc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj_anc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj_anc = nn.Linear(dim, dim, bias=qkv_bias)

        # -------- gate + output --------
        self.gate_proj = nn.Linear(dim * 3, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.anchor_norm = nn.LayerNorm(dim)

    def _pool_anchors(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, Nz, C] -> anchors: [B, Na, C]
        """
        B, N, C = z.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"SALA anchor tokens must form square grid, got N={N}"

        z = z.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]
        z = F.avg_pool2d(
            z,
            kernel_size=self.anchor_pool_size,
            stride=self.anchor_pool_size
        )  # [B,C,Ha,Wa]
        z = z.permute(0, 2, 3, 1).contiguous().view(B, -1, C)     # [B,Na,C]
        z = self.anchor_norm(z)
        return z

    def _local_branch(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,N,C]
        out_loc: [B,N,C]
        """
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"SALA local branch expects square token grid, got N={N}"

        k = self.kernel_size
        dil_h, dil_w = self.dilation
        pad_h, pad_w = self.padding

        x_2d = x.view(B, H, W, C).contiguous()

        q = self.q_proj_loc(x_2d)
        k_ = self.k_proj_loc(x_2d)
        v_ = self.v_proj_loc(x_2d)

        q = q.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 4, 1, 2)   # [B,h,d,H,W]
        k_ = k_.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 4, 1, 2)
        v_ = v_.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 4, 1, 2)

        q_flat = q.reshape(B * self.num_heads, self.head_dim, H * W)   # [Bh,d,HW]
        k_4d = k_.reshape(B * self.num_heads, self.head_dim, H, W)
        v_4d = v_.reshape(B * self.num_heads, self.head_dim, H, W)

        k_unf = F.unfold(
            k_4d,
            kernel_size=(k, k),
            dilation=(dil_h, dil_w),
            padding=(pad_h, pad_w),
            stride=1,
        )  # [Bh, d*k*k, HW]

        v_unf = F.unfold(
            v_4d,
            kernel_size=(k, k),
            dilation=(dil_h, dil_w),
            padding=(pad_h, pad_w),
            stride=1,
        )

        K2 = k * k
        k_unf = k_unf.view(B * self.num_heads, self.head_dim, K2, H * W)   # [Bh,d,K2,HW]
        v_unf = v_unf.view(B * self.num_heads, self.head_dim, K2, H * W)

        attn_logits = (q_flat.unsqueeze(2) * k_unf).sum(dim=1) * self.scale   # [Bh,K2,HW]
        attn = F.softmax(attn_logits, dim=1)

        out_flat = (attn.unsqueeze(1) * v_unf).sum(dim=2)   # [Bh,d,HW]
        out = out_flat.view(B, self.num_heads, self.head_dim, H, W)
        out = out.permute(0, 3, 4, 1, 2).contiguous().view(B, N, C)
        return out

    def _anchor_branch(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: [B,N,C]
        z: [B,Nz,C]
        out_anc: [B,N,C]
        """
        B, N, C = x.shape
        anchors = self._pool_anchors(z)  # [B,Na,C]
        Na = anchors.shape[1]

        q = self.q_proj_anc(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)       # [B,h,N,d]
        k = self.k_proj_anc(anchors).view(B, Na, self.num_heads, self.head_dim).transpose(1, 2) # [B,h,Na,d]
        v = self.v_proj_anc(anchors).view(B, Na, self.num_heads, self.head_dim).transpose(1, 2) # [B,h,Na,d]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B,h,N,Na]
        attn = attn.softmax(dim=-1)

        out = attn @ v                                  # [B,h,N,d]
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return out

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: [B,N,C]
        z: [B,Nz,C]
        """
        o_loc = self._local_branch(x)       # [B,N,C]
        o_anc = self._anchor_branch(x, z)   # [B,N,C]

        g = torch.sigmoid(self.gate_proj(torch.cat([x, o_loc, o_anc], dim=-1)))  # [B,N,C]
        out = o_loc + g * o_anc
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def modulate2(x, shift, scale):
    return x * (1 + scale) + shift
def modulate3(x, scale):
    return x * (1 + scale)



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
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings




class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    使用 SALA Attention 替代原来的 ProximalAttention / NAttention。
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=2.0,
        na_kernel_size: int = 7,
        na_dilation: Union[int, Tuple[int, int]] = 1,
        anchor_pool_size: int = 2,
        qkv_bias: bool = True,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = SALAAttention2D(
            dim=hidden_size,
            num_heads=num_heads,
            kernel_size=na_kernel_size,
            dilation=na_dilation,
            qkv_bias=qkv_bias,
            proj_drop=0.0,
            anchor_pool_size=anchor_pool_size,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, z):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        x_norm = modulate2(self.norm1(x), shift_msa, scale_msa)

        attn_out = self.attn(x_norm, z)

        x = x + gate_msa * attn_out
        x = x + gate_mlp * self.mlp(modulate2(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate2(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        c_channels=1024,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=2.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        na_kernel_size: int = 7,
        na_dilation: Union[int, Tuple[int,int]] = 1,
        anchor_pool_size: int = 2,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.c_channels = c_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.c_embedder = PatchEmbed(input_size, patch_size, c_channels, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        c_num_patches = self.c_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.c_pos_embed = nn.Parameter(torch.zeros(1, c_num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                na_kernel_size=na_kernel_size,
                na_dilation=na_dilation,
                anchor_pool_size=anchor_pool_size,
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        c_pos_embed = get_2d_sincos_pos_embed(
            self.c_pos_embed.shape[-1],
            int(self.c_embedder.num_patches ** 0.5)
        )
        self.c_pos_embed.data.copy_(torch.from_numpy(c_pos_embed).float().unsqueeze(0))
        

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            if hasattr(block.attn, "gate_proj"):
                nn.init.zeros_(block.attn.gate_proj.weight)
                nn.init.constant_(block.attn.gate_proj.bias, -2.0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

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

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y, z, mask=None, tokens=None, force_drop_ids=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        z: (N, P, H, W) 
        """
        b, d, h, w = x.shape
        if torch.any(torch.isnan(x)):
            print("nan")
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        if torch.any(torch.isnan(x)):
            print("nan")
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training, force_drop_ids=force_drop_ids) if y is not None else None
        z = self.c_embedder(z) + self.c_pos_embed

        if y is not None:
            c = t.unsqueeze(1) + z + y.unsqueeze(1)
        else:
            c = t.unsqueeze(1) + z
        for block in self.blocks:
            x = block(x, c, z) 
        if torch.any(torch.isnan(x)):
            print("nan")
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if torch.any(torch.isnan(x)):
            print("nan")
        return x

    def forward_with_cfg(self, x, t, y, z, cfg_scale, mask=None, tokens=None):
        """
        Forward pass of DiT with classifier-free guidance.
        First half: conditional
        Second half: unconditional
        """
        half = len(x) // 2

        half_x = x[:half]
        half_t = t[:half]
        half_z = z[:half]
        half_y = y[:half] if y is not None else None

        combined_x = torch.cat([half_x, half_x], dim=0)
        combined_t = torch.cat([half_t, half_t], dim=0)
        combined_z = torch.cat([half_z, half_z], dim=0)

        if half_y is not None:
            # 前一半 conditional，后一半 unconditional
            force_drop_ids = torch.cat([
                torch.zeros(half, device=half_y.device, dtype=torch.long),
                torch.ones(half, device=half_y.device, dtype=torch.long)
            ], dim=0)
            combined_y = torch.cat([half_y, half_y], dim=0)
        else:
            force_drop_ids = None
            combined_y = None

        model_out = self.forward(
            combined_x, combined_t, combined_y, combined_z,
            mask=mask, tokens=tokens, force_drop_ids=force_drop_ids
        )

        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, half, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


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
