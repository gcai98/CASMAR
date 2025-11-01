from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional
from timm.layers import PatchEmbed, Mlp, DropPath, RmsNorm
from timm.models.vision_transformer import LayerScale, Attention
from models.diffloss import DiffLoss, GlobalDiffLoss
import scipy.stats as stats


def modulate3(x, scale):
    return x * (1 + scale)


class MaskRatioGenerator:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def rvs(self, i):
        r = torch.rand(i) * (self.max_val - self.min_val) + self.min_val
        mask_rate = torch.cos(r * math.pi * 0.5)
        return mask_rate


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()],
                            src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


def mask_by_order_step(mask_len, order, bsz, seq_len, cond_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()],
                            src=torch.ones(bsz, seq_len).cuda()).bool()
    masking[:, cond_len:] = 1
    return masking


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        num_tasks: int = 2,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adaLN_scale = nn.Parameter(torch.zeros((num_tasks, 1, 1, 4 * dim)), requires_grad=True)
        self.adaLN_shift = nn.Parameter(torch.cat([
            torch.zeros((num_tasks, 1, 1, dim)),
            torch.ones((num_tasks, 1, 1, dim)),
            torch.zeros((num_tasks, 1, 1, dim)),
            torch.ones((num_tasks, 1, 1, dim))
        ], dim=-1), requires_grad=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor, task: int) -> torch.Tensor:
        c = c * self.adaLN_scale[task] + self.adaLN_shift[task]
        scale_msa, gate_msa, scale_mlp, gate_mlp = c.chunk(4, dim=-1)
        x = x + gate_msa * self.drop_path1(self.ls1(self.attn(modulate3(self.norm1(x), scale_msa))))
        x = x + gate_mlp * self.drop_path2(self.ls2(self.mlp(modulate3(self.norm2(x), scale_mlp))))
        return x


class casmar(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 cond_drop_prob=0.5,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 diff_seqlen=False,
                 step_warmup=100,
                 step_stage2_rate=0.5,
                 cond_scale=4,
                 cond_dim=16,
                 two_diffloss=False,
                 seq_len=None,
                 global_dm=False,
                 gdm_w=768,
                 gdm_d=3,
                 head=8,
                 ratio=4,
                 cos=True,
                 train_use_varmask: bool = True,
                 train_mask_ratio_var: float = 0.10,
                 train_saliency_delta: float = 0.5,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.two_diffloss = two_diffloss
        self.global_dm = global_dm
        self.cond_scale = cond_scale
        self.cond_dim = cond_dim
        self.step_warmup = step_warmup
        self.step_stage2_rate = step_stage2_rate
        self.vae_embed_dim = vae_embed_dim
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.diff_seqlen = diff_seqlen
        self.seq_len = self.seq_h * self.seq_w if seq_len is None else seq_len
        self.token_embed_dim = vae_embed_dim * patch_size ** 2
        self.grad_checkpointing = grad_checkpointing
        self.decoder_depth = decoder_depth
        self.epoch = 0
        self.gdm_w = gdm_w
        self.gdm_d = gdm_d

        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        if cos:
            self.mask_ratio_generator2 = MaskRatioGenerator()
        else:
            self.mask_ratio_generator2 = stats.truncnorm((0.38 - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------

        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        self.cond_z_proj = nn.Linear(decoder_embed_dim, encoder_embed_dim, bias=True)
        self.buffer_size = buffer_size
        self.cond_tokens = nn.Parameter(torch.zeros(1, self.cond_scale * self.cond_scale, self.cond_dim))
        self.cond_proj = nn.Linear(self.cond_dim, encoder_embed_dim, bias=True)

        self.seq_len = self.seq_len + self.cond_scale * self.cond_scale
        self.small_seqlen = self.seq_len
        self.mask_token = nn.Parameter(torch.zeros(1, self.small_seqlen + self.seq_len, encoder_embed_dim))

        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.small_seqlen + 2 * self.buffer_size, encoder_embed_dim))
        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_task_emb = nn.Embedding(2, encoder_embed_dim)
        self.encoder_blocks_ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(encoder_embed_dim, 4 * encoder_embed_dim, bias=True)
        )
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.small_seqlen + 2 * self.buffer_size, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_task_emb = nn.Embedding(2, decoder_embed_dim)
        self.decoder_blocks_ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(decoder_embed_dim, 4 * decoder_embed_dim, bias=True)
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_embed_dim = decoder_embed_dim
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))
        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing,
        )
        self.cond_diffloss = GlobalDiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=gdm_w,
            depth=gdm_d,
            num_sampling_steps=num_sampling_steps,
            head=head,
            ratio=ratio,
        )
        self.diffusion_batch_mul = diffusion_batch_mul

        self._patch_diffusion_engines_force_long()

        self.train_use_varmask = train_use_varmask
        self.train_mask_ratio_var = train_mask_ratio_var
        self.train_saliency_delta = train_saliency_delta

    # --------------------------------------------------------------------------
    def saliency_guided_masking(self, x, base_mask_ratio, mask_ratio_var, delta):

        N, L, D = x.shape

        aff = torch.matmul(x, x.permute(0, 2, 1))
        aff = nn.functional.softmax(aff, dim=2)
        aff_sum = torch.sum(aff, dim=1)

        minv = aff_sum.min(dim=1, keepdim=True)[0]
        maxv = aff_sum.max(dim=1, keepdim=True)[0]
        aff_sum_normalized = (aff_sum - minv) / (maxv - minv + 1e-6)

        y = (aff_sum_normalized > delta).sum(dim=1)

        y_max = aff_sum_normalized.size(1)
        y_normalized = y.float().mean() / y_max

        dynamic_mask_ratios = base_mask_ratio - mask_ratio_var + 2 * mask_ratio_var * y_normalized
        dynamic_mask_ratios = float(max(0.0, min(1.0, dynamic_mask_ratios)))
        len_keep = max(1, int(L * (1 - dynamic_mask_ratios)))   

        noise = torch.rand(N, L, device=x.device) / 2
        saliency_guided_noise = aff_sum_normalized + noise

        ids_shuffle = torch.argsort(saliency_guided_noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    # --------------------------------------------------------------------------
    def build_training_masks_variable(self, x_tokens, cond_tokens):

        device = x_tokens.device
        bsz, L_big, _ = x_tokens.shape
        cond_len = self.cond_scale * self.cond_scale

        with torch.no_grad():
            small_feat = self.cond_proj(cond_tokens)    # (B, L_small, E)
            big_feat   = self.z_proj(x_tokens)          # (B, L_big,  E)

            base_small = float(self.mask_ratio_generator.rvs(1)[0])
            base_big   = float(self.mask_ratio_generator2.rvs(1)[0])

            _, small_mask_only, _ = self.saliency_guided_masking(
                x=small_feat,
                base_mask_ratio=base_small,
                mask_ratio_var=self.train_mask_ratio_var,
                delta=self.train_saliency_delta
            )
            _, big_mask_only, _ = self.saliency_guided_masking(
                x=big_feat,
                base_mask_ratio=base_big,
                mask_ratio_var=self.train_mask_ratio_var,
                delta=self.train_saliency_delta
            )

        seq_len_full = cond_len + L_big

        mask_stage1 = torch.ones(bsz, seq_len_full, device=device)
        mask_stage1[:, :cond_len] = small_mask_only
        mask_stage1[:, cond_len:] = 1

        mask_stage2 = torch.zeros(bsz, seq_len_full, device=device)
        mask_stage2[:, :cond_len] = 0
        mask_stage2[:, cond_len:] = big_mask_only

        return mask_stage1, mask_stage2

    def _patch_diffusion_engines_force_long(self):
        engines = []
        for holder in [getattr(self, "diffloss", None), getattr(self, "cond_diffloss", None)]:
            if holder is None:
                continue
            for attr in ["gen_diffusion", "diffusion"]:  # 兼容不同实现的命名
                eng = getattr(holder, attr, None)
                if eng is not None and eng not in engines:
                    engines.append(eng)

        import types as _types

        for eng in engines:
            if hasattr(eng, "__call__") and not getattr(eng.__call__, "_force_long_patched", False):
                _orig_call = eng.__call__

                def _patched_call(self_, x, ts, **kw):
                    if not torch.is_tensor(ts):
                        ts = torch.as_tensor(ts, dtype=torch.long, device=x.device)
                    elif ts.dtype != torch.long and ts.dtype != torch.int64:
                        ts = ts.to(dtype=torch.long, device=x.device)
                    else:
                        ts = ts.to(device=x.device)
                    return _orig_call(x, ts, **kw)

                _patched_call._force_long_patched = True
                eng.__call__ = _types.MethodType(_patched_call, eng)

    # --------------------------------------------------------------------------

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.cond_tokens, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def unpatchify_small(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.cond_dim
        h_, w_ = self.cond_scale, self.cond_scale

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders_step(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for i in range(bsz):
            order = list(range(self.cond_scale * self.cond_scale))
            np.random.shuffle(order)
            order1 = list(range(self.cond_scale * self.cond_scale, self.seq_len))
            np.random.shuffle(order1)
            order = np.array(order + order1)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        _ = torch.rand(1)
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def random_masking_step(self, device, seq_len, bsz, orders, mode):
        # generate token mask
        mask = torch.zeros(bsz, seq_len, device=device)

        if mode:
            _ = torch.rand(1)
            mask_rate = self.mask_ratio_generator.rvs(1)[0]
            # 前 cond_len 个 token 按 mask_rate 做 mask，后面的全 mask
            num_masked_tokens = int(np.ceil(self.cond_scale * self.cond_scale * mask_rate))
            mask = torch.zeros(bsz, seq_len, device=device)
            mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                                 src=torch.ones(bsz, seq_len, device=device))
            mask[:, self.cond_scale * self.cond_scale:] = 1
        else:
            _ = torch.rand(1)
            mask_rate = self.mask_ratio_generator2.rvs(1)[0]
            # 前 cond_len 个 token 不 mask，后面的按照 mask_rate 做 mask
            num_masked_tokens = int(np.ceil((seq_len - self.cond_scale * self.cond_scale) * mask_rate))
            mask = torch.zeros(bsz, seq_len, device=device)
            mask = torch.scatter(mask, dim=-1,
                                 index=orders[:,
                                        self.cond_scale * self.cond_scale:self.cond_scale * self.cond_scale + num_masked_tokens],
                                 src=torch.ones(bsz, seq_len, device=device))

        return mask

    def forward_mae_encoder_stage1(self, x, mask, class_embedding, task, cond):
        if self.training:
            # train small scale, so small_cond_mask is False
            small_x = self.cond_proj(cond)
            x = torch.cat([small_x, self.z_proj(x)], dim=1)
        else:
            x = torch.cat([self.cond_proj(cond), self.z_proj(x)], dim=1)

        bsz, seq_len, embed_dim = x.shape
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        if self.training:
            drop_latent_mask = torch.rand(bsz, device=x.device) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        x = x + self.encoder_pos_embed_learned[:, :self.small_seqlen + self.buffer_size, :]

        x = self.z_proj_ln(x)

        x = x[(1 - mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
        task_c = self.encoder_task_emb(torch.tensor([[task]], device=x.device))
        ada = self.encoder_blocks_ada(task_c)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x, ada, task)
        else:
            for block in self.encoder_blocks:
                x = block(x, ada, task)
        x = self.encoder_norm(x)
        return x

    def forward_mae_encoder_stage2(self, x, mask, class_embedding, task, cond=None, small_z=None, small_mask=None):
        if self.training:
            small_cond_mask = (torch.rand(x.shape[0], device=x.device)
                               <= 1.0 - self.step_stage2_rate * min(1, self.epoch / self.step_warmup))
            small_cond_mask = small_cond_mask.unsqueeze(-1).repeat(1, self.cond_scale * self.cond_scale).to(x.dtype).unsqueeze(-1)

            if small_z is None:
                small_x = small_cond_mask * self.cond_proj(self.cond_tokens).repeat(x.shape[0], 1, 1) + \
                          (1 - small_cond_mask) * self.cond_proj(cond)
            else:
                small_x1 = small_mask.unsqueeze(-1) * self.cond_proj(self.cond_tokens).repeat(x.shape[0], 1, 1) + \
                           (1 - small_mask.unsqueeze(-1)) * self.cond_z_proj(small_z)
                small_x = small_cond_mask * self.cond_proj(self.cond_tokens).repeat(x.shape[0], 1, 1) + \
                          (1 - small_cond_mask) * small_x1

            x = torch.cat([small_x, self.z_proj(x)], dim=1)
        else:
            if small_z is None:
                x = torch.cat([self.cond_proj(cond), self.z_proj(x)], dim=1)
            else:
                x = torch.cat([self.cond_z_proj(small_z), self.z_proj(x)], dim=1)

        bsz, seq_len, embed_dim = x.shape
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        if self.training:
            drop_latent_mask = torch.rand(bsz, device=x.device) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        x = x + self.encoder_pos_embed_learned[:, self.small_seqlen + self.buffer_size:, :]

        x = self.z_proj_ln(x)

        x = x[(1 - mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
        task_c = self.encoder_task_emb(torch.tensor([[task]], device=x.device))
        ada = self.encoder_blocks_ada(task_c)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x, ada, task)
        else:
            for block in self.encoder_blocks:
                x = block(x, ada, task)
        x = self.encoder_norm(x)
        return x

    def forward_mae_decoder_stage1(self, x, mask, task):
        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        mask_tokens = torch.cat([
            torch.zeros(mask_with_buffer.shape[0], self.buffer_size, self.mask_token.shape[2], dtype=x.dtype,
                        device=x.device),
            self.mask_token[:, :self.small_seqlen, :].repeat(mask_with_buffer.shape[0], 1, 1).to(x.dtype)
        ], dim=1)

        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x = x_after_pad + self.decoder_pos_embed_learned[:, :self.small_seqlen + self.buffer_size, :]
        task_c = self.decoder_task_emb(torch.tensor([[task]], device=x.device))
        ada = self.decoder_blocks_ada(task_c)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x, ada, task)
        else:
            for block in self.decoder_blocks:
                x = block(x, ada, task)
        x = self.decoder_norm(x)
        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_mae_decoder_stage2(self, x, mask, task):
        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        mask_tokens = torch.cat([
            torch.zeros(mask_with_buffer.shape[0], self.buffer_size, self.mask_token.shape[2], dtype=x.dtype,
                        device=x.device),
            self.mask_token[:, self.small_seqlen:, :].repeat(mask_with_buffer.shape[0], 1, 1).to(x.dtype)
        ], dim=1)

        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x = x_after_pad + self.decoder_pos_embed_learned[:, self.small_seqlen + self.buffer_size:, :]

        task_c = self.decoder_task_emb(torch.tensor([[task]], device=x.device))
        ada = self.decoder_blocks_ada(task_c)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x, ada, task)
        else:
            for block in self.decoder_blocks:
                x = block(x, ada, task)
        x = self.decoder_norm(x)
        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def global_forward_loss(self, z, target, mask, c):
        bsz, _, h, w = target.shape
        target = target.repeat(self.diffusion_batch_mul, 1, 1, 1)
        z = z.reshape(bsz, h, w, -1).permute(0, 3, 1, 2).repeat(self.diffusion_batch_mul, 1, 1, 1)
        mask = mask.reshape(bsz, h, w).repeat(self.diffusion_batch_mul, 1, 1)
        c = c.repeat(self.diffusion_batch_mul)
        loss = self.cond_diffloss(z=z, target=target, mask=mask, c=c)
        return loss

    def forward(self, imgs, labels, cond=None):
        class_embedding = self.class_emb(labels)

        x = self.patchify(imgs)   # (B, L_big, token_dim)
        cond = self.patchify(cond)  # (B, L_small, cond_dim)

        gt_latents = x.clone().detach()
        cond_gt_latents = cond.clone().detach()

        if self.train_use_varmask:
            mask, big_mask = self.build_training_masks_variable(x, cond)
        else:
            orders = self.sample_orders_step(bsz=x.size(0))
            mask = self.random_masking_step(x.device, x.shape[1] + cond.shape[1], x.shape[0], orders, mode=True)

        x_small_c = self.forward_mae_encoder_stage1(x, mask, class_embedding, task=0, cond=cond)

        z_small_c = self.forward_mae_decoder_stage1(x_small_c, mask, task=0)

        z_small_c = z_small_c[:, :self.cond_scale * self.cond_scale, :]
        gt_latents_small_c = cond_gt_latents
        small_mask_for_loss = mask[:, :self.cond_scale * self.cond_scale]
        loss1 = self.forward_loss(z=z_small_c, target=gt_latents_small_c, mask=small_mask_for_loss)
        small_mask = 1.0 - small_mask_for_loss.clone()
        small_z = z_small_c.clone().detach()

        if not self.train_use_varmask:
            big_mask = self.random_masking_step(x.device, x.shape[1] + cond.shape[1], x.shape[0], orders, mode=False)

        x_big_c = self.forward_mae_encoder_stage2(
            x, big_mask, class_embedding, task=1, cond=cond, small_z=small_z, small_mask=small_mask
        )
        z_big_c = self.forward_mae_decoder_stage2(x_big_c, big_mask, task=1)
        z_big_c = z_big_c[:, self.cond_scale * self.cond_scale:, :]
        gt_latents_big_c = gt_latents
        big_mask_for_loss = big_mask[:, self.cond_scale * self.cond_scale:]
        loss2 = self.global_forward_loss(z=z_big_c, target=imgs, mask=big_mask_for_loss, c=labels)

        if torch.any(torch.isnan(loss1)) or torch.any(torch.isnan(loss2)):
            print("nan")
        return 2.0 * loss1, 0.75 * loss2

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, re_cfg=1.0, cfg_schedule="linear",
                      labels=None, temperature=1.0, progress=False, vq_model=None, cond=None, stage=1, resmall_tokens=None):

        device = torch.device("cuda")
        cond_len = self.cond_scale * self.cond_scale

        mask = torch.ones(bsz, self.seq_len, device=device)
        tokens = torch.zeros(bsz, self.seq_len - cond_len, self.token_embed_dim, device=device)
        cond_tokens = torch.zeros(bsz, cond_len, self.cond_dim, device=device)

        small_z = torch.zeros(bsz * 2, cond_len, self.decoder_embed_dim, device=device)


        small_num_iter = self.cond_scale * 4
        for step in range(small_num_iter):
            cur_cond_tokens = cond_tokens.clone()
            cur_tokens = tokens.clone()

            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)

            if not re_cfg == 1.0:
                cond_tokens = torch.cat([cond_tokens, cond_tokens], dim=0)
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            x = self.forward_mae_encoder_stage1(tokens, mask, class_embedding, task=0, cond=cond_tokens)
            z = self.forward_mae_decoder_stage1(x, mask, task=0)  # (B or 2B, seq_len, dec_dim)

            z_small = z[:bsz, :cond_len, :]
            base_ratio = float(np.cos(math.pi / 2. * (step + 1) / small_num_iter))  
            _, small_mask_next_only, _ = self.saliency_guided_masking(
                x=z_small,
                base_mask_ratio=base_ratio,
                mask_ratio_var=self.train_mask_ratio_var,
                delta=self.train_saliency_delta
            )
            mask_next = torch.ones(bsz, self.seq_len, device=device)  # 1 = masked
            mask_next[:, :cond_len] = small_mask_next_only
            mask_next[:, cond_len:] = 1

            if step >= small_num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
                mask_to_pred[:, cond_len:] = False
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())

            if mask_to_pred.shape[0] > 0:
                for bi in range(bsz):
                    if not mask_to_pred[bi].any():
                        with torch.no_grad():
                            ref = z_small[bi:bi + 1]  # (1, Ls, D)
                            aff = torch.matmul(ref, ref.transpose(-1, -2)).softmax(dim=2).sum(dim=1)[0]  # (Ls,)
                            masked_now = mask[bi, :cond_len].bool()
                            scores = aff.masked_fill(~masked_now, float('-inf'))
                            top_idx = torch.argmax(scores).clamp(min=0)
                            full = torch.zeros_like(mask_to_pred[bi])
                            full[top_idx] = True
                            mask_to_pred[bi] = full

            if not re_cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            small_z[mask_to_pred[:, :cond_len].nonzero(as_tuple=True)] = \
                z[mask_to_pred[:, :cond_len].nonzero(as_tuple=True)]

            z_sel = z[mask_to_pred.nonzero(as_tuple=True)].contiguous()

            if cfg_schedule == "linear":
                keep_fraction_next = (mask_next[:, :cond_len] == 0).float().mean().item()
                cfg_iter = 1 + (re_cfg - 1) * keep_fraction_next
            elif cfg_schedule == "constant":
                cfg_iter = re_cfg
            else:
                raise NotImplementedError

            sampled_token_latent = self.diffloss.sample(z_sel, temperature, cfg_iter)
            if not re_cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_cond_tokens[mask_to_pred[:, :cond_len].nonzero(as_tuple=True)] = sampled_token_latent

            tokens = cur_tokens.clone()
            cond_tokens = cur_cond_tokens.clone()
            mask = mask_next  

        small_tokens = self.unpatchify_small(cond_tokens)
        mask[:, :cond_len] = False


        if not cfg == 1.0:
            small_z = torch.cat([small_z[:bsz], small_z[:bsz]], dim=0)
        else:
            small_z = small_z[:bsz]

        for step in range(num_iter):
            cur_tokens = tokens.clone()

            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)

            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                cond_tokens = torch.cat([cond_tokens, cond_tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            x = self.forward_mae_encoder_stage2(tokens, mask, class_embedding, task=1,
                                                cond=cond_tokens, small_z=small_z)
            z = self.forward_mae_decoder_stage2(x, mask, task=1)

            this_mask = mask[:, cond_len:].clone().bool().reshape(mask.shape[0], self.seq_h, self.seq_w)
            this_token = tokens.clone().reshape(tokens.shape[0], self.seq_h, self.seq_w, -1).permute(0, 3, 1, 2)
            last_mask = mask.clone().bool() 

            z_big = z[:bsz, cond_len:, :]
            base_ratio = float(np.cos(math.pi / 2. * (step + 1) / num_iter))
            _, big_mask_next_only, _ = self.saliency_guided_masking(
                x=z_big,
                base_mask_ratio=base_ratio,
                mask_ratio_var=self.train_mask_ratio_var,
                delta=self.train_saliency_delta
            )
            mask_next = torch.zeros(bsz, self.seq_len, device=device)
            mask_next[:, :cond_len] = 0
            mask_next[:, cond_len:] = big_mask_next_only

            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())

            if mask_to_pred.shape[0] > 0:
                for bi in range(bsz):
                    if not mask_to_pred[bi].any():
                        with torch.no_grad():
                            ref = z_big[bi:bi + 1]  # (1, Lb, D)
                            aff = torch.matmul(ref, ref.transpose(-1, -2)).softmax(dim=2).sum(dim=1)[0]  # (Lb,)
                            masked_now = mask[bi, cond_len:].bool()
                            scores = aff.masked_fill(~masked_now, float('-inf'))
                            top_idx = torch.argmax(scores).clamp(min=0)
                            full = torch.zeros_like(mask_to_pred[bi])
                            full[top_idx + cond_len] = True  
                            mask_to_pred[bi] = full

            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            mask = mask_next

            z_big_2d = z[:, cond_len:, :].reshape(z.shape[0], self.seq_h, self.seq_w, -1).permute(0, 3, 1, 2)

            if cfg_schedule == "linear":
                keep_fraction_next = (mask_next[:, cond_len:] == 0).float().mean().item()
                cfg_iter = 1 + (cfg - 1) * keep_fraction_next
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError

            if cfg != 1.0:
                null_label = (self.num_classes) * torch.ones_like(labels, device=labels.device)
                cfg_labels = torch.cat([labels, null_label], dim=0)
            else:
                cfg_labels = labels

            sampled_token_latent = self.cond_diffloss.sample(
                z_big_2d, cfg_labels, temperature, cfg_iter, this_mask, this_token
            )
            sampled_token_latent = sampled_token_latent.reshape(sampled_token_latent.shape[0],
                                                                sampled_token_latent.shape[1], -1).permute(0, 2, 1)
            sampled_token_latent = sampled_token_latent[last_mask[:, cond_len:].nonzero(as_tuple=True)]

            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
                last_mask, _ = last_mask.chunk(2, dim=0)
                cond_tokens, _ = cond_tokens.chunk(2, dim=0)

            cur_tokens[last_mask[:, cond_len:].nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        big_cond_tokens = self.unpatchify(tokens)
        return small_tokens, big_cond_tokens


def casmar_base(**kwargs):
    model = casmar(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def casmar_large(**kwargs):
    model = casmar(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def casmar_huge(**kwargs):
    model = casmar(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
