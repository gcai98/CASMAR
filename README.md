# CASMAR

![Qualitative Results](./demo/results.jpg)

**Coarse-to-Fine Continuous Masked Autoregressive Image Generation via Cross-Scale Scaffold-Guided Refinement**

This is the official repository for the paper **"Coarse-to-Fine Continuous Masked Autoregressive Image Generation via Cross-Scale Scaffold-Guided Refinement"**.



## Overview

We present **CASMAR (Cascaded / Coarse-to-Fine Continuous Masked Autoregression)**, a two-stage framework for scalable and structure-aware image generation in continuous latents.

Unlike single-scale continuous MAR, CASMAR explicitly **decouples coarse structure formation from fine-detail synthesis**. **Stage 1** learns a low-resolution **cross-scale scaffold** that stabilizes global scene organization, while **Stage 2** performs **scaffold-guided refinement** to recover high-resolution details. To better adapt masking difficulty to image complexity, CASMAR further introduces **Salience-Aware Mask Scheduling (SAMS)** with an **Adaptive Mask Ratio (AMR)**. In Stage 2, we use **Scaffold-Anchored Local Attention (SALA)** to combine efficient local refinement with scaffold-based structural guidance.

CASMAR is evaluated on **ImageNet-1K** for large-scale class-conditional generation and on **AID-HeSA** for structure-sensitive remote-sensing scene generation.



## 🔍 What CASMAR Solves

- **Entangled structure and detail generation** in single-scale continuous MAR
- **Fixed/random masking** that does not adapt to heterogeneous image complexity
- **Expensive dense global refinement** at high resolution
- **Weak structural guidance** during local detail reconstruction



## 🧠 How CASMAR Works

- **Stage 1 — Cross-scale scaffold formation.**  
  CASMAR first predicts a low-resolution structural scaffold that captures coarse spatial layout and semantic organization.

- **Stage 2 — Scaffold-guided detail refinement.**  
  Stage 2 refines high-resolution latent tokens conditioned on the Stage 1 scaffold. The scaffold features are detached and projected into cross-stage condition features before being reused for refinement.

- **SAMS + AMR.**  
  Instead of using a fixed global mask ratio, CASMAR estimates token saliency and dynamically adjusts both **which tokens remain visible** and **how much content is masked**, improving robustness across simple and complex scenes.

- **SALA.**  
  Stage 2 uses **Scaffold-Anchored Local Attention**, a dual-branch gated refinement mechanism:
  - a **local branch** for texture continuity and boundary refinement,
  - an **anchor branch** for structure-aware correction using compact scaffold anchors from Stage 1.

- **Unified continuous masked denoising.**  
  CASMAR operates in continuous latent space with diffusion-based denoising heads. Stage 1 uses a token-wise diffusion head, while Stage 2 uses a spatial conditional diffusion head for structure-consistent refinement.



## 📈 Highlights

- **Strong ImageNet-1K performance.**  
  CASMAR consistently improves over MAR across model scales.

- **Structure-aware generation.**  
  The coarse-to-fine scaffold design is especially beneficial when scene organization matters.

- **Efficient refinement.**  
  SALA reduces the cost of dense global refinement while preserving structural consistency.

- **Flexible masking.**  
  SAMS with AMR adaptively calibrates masking difficulty based on token saliency.



## 🧱 Model Zoo

| Model    | CASMAR Transformer |             | Diff. Head1 |             | Diff. Head2 |             | #Params |
|----------|--------------------|-------------|-------------|-------------|-------------|-------------|---------|
|          | Layers             | Hidden size | Layers      | Hidden size | Layers      | Hidden size |         |
| CASMAR-B | 24                 | 768         | 6           | 1024        | 6           | 512         | 254.4M  |
| CASMAR-L | 32                 | 1024        | 8           | 1280        | 8           | 512         | 542.3M  |
| CASMAR-H | 40                 | 1280        | 12          | 1536        | 12          | 768         | 1133.9M |



## 🏃🏼 Quick Start

<details open>
<summary><strong>Environment Requirement</strong></summary>

Clone the repo:

```bash
git clone https://github.com/HiDream-ai/casmar.git
cd casmar
```



Install dependencies:

```
conda env create -f environment.yaml
conda activate casmar
```



## ⚡ Caching VAE Latents

Given that the training pipeline only uses lightweight image augmentations, the VAE latents can be precomputed and stored at `CACHED_PATH`:

```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_cache.py \
--img_size 256 \
--vae_path pretrained_models/vae/kl16.ckpt \
--vae_embed_dim 16 \
--batch_size 128 \
--data_path ${IMAGENET_PATH} \
--cached_path ${CACHED_PATH}
```

After caching, you can train CASMAR directly from the saved latent files to reduce repeated VAE encoding overhead.

## 🏋️ Training

Below are example scripts for training CASMAR on **ImageNet 256×256**.

### Train CASMAR-B

```
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=${NODE_RANK} \
--master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_casmar.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model casmar_base --diffloss_d 6 --diffloss_w 1024 \
--epochs 400 --warmup_epochs 100 --batch_size 64 --blr 1.0e-4 --diffusion_batch_mul 4 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} \
--cached_path ${CACHED_PATH} \
--num_iter 4 --cfg 2.5 --re_cfg 2.7 --cfg_schedule linear \
--cond_scale 8 --cond_dim 16 \
--two_diffloss --global_dm --gdm_d 6 --gdm_w 512 \
--head 8 --ratio 4 --cos
```

### Train CASMAR-L

```
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=${NODE_RANK} \
--master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_casmar.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model casmar_large --diffloss_d 8 --diffloss_w 1280 \
--epochs 400 --warmup_epochs 100 --batch_size 64 --blr 1.0e-4 --diffusion_batch_mul 4 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} \
--cached_path ${CACHED_PATH} \
--num_iter 4 --cfg 3.5 --re_cfg 3.5 --cfg_schedule linear \
--cond_scale 8 --cond_dim 16 \
--two_diffloss --global_dm --gdm_d 8 --gdm_w 512 \
--head 8 --ratio 4 --cos
```

### Train CASMAR-H

```
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=${NODE_RANK} \
--master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_casmar.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model casmar_huge --diffloss_d 12 --diffloss_w 1536 \
--epochs 400 --warmup_epochs 100 --batch_size 64 --blr 1.0e-4 --diffusion_batch_mul 4 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} \
--cached_path ${CACHED_PATH} \
--num_iter 12 --cfg 3.2 --re_cfg 5.5 --cfg_schedule linear \
--cond_scale 8 --cond_dim 16 \
--two_diffloss --global_dm --gdm_d 12 --gdm_w 768 \
--head 12 --ratio 4 --cos
```

> Notes:
>
> - `diffloss_d` / `diffloss_w` correspond to the **Stage 1 diffusion head**.
> - `gdm_d` / `gdm_w` correspond to the **Stage 2 diffusion head**.
> - `two_diffloss` and `global_dm` enable the two-stage coarse-to-fine training setup.
> - If your local training script does not accept `--cached_path`, simply remove it and keep `--data_path`.

## 🧪 Evaluation

### Evaluate CASMAR-B on ImageNet 256×256

```python
torchrun --nproc_per_node=8 --nnodes=1 \
main_casmar.py \
--img_size 256 --vae_path /path/to/vae --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model casmar_base --diffloss_d 6 --diffloss_w 1024 \
--output_dir ./casmar_base_test --resume /path/to/CASMAR-B \
--num_images 50000 --num_iter 4 \
--cfg 2.5 --re_cfg 2.7 --cfg_schedule linear \
--cond_scale 8 --cond_dim 16 \
--two_diffloss --global_dm --gdm_d 6 --gdm_w 512 \
--eval_bsz 256 --load_epoch -1 \
--head 8 --ratio 4 --cos --evaluate
```

### Evaluate CASMAR-L on ImageNet 256×256

```python
torchrun --nproc_per_node=8 --nnodes=1 \
main_casmar.py \
--img_size 256 --vae_path /path/to/vae --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model casmar_large --diffloss_d 8 --diffloss_w 1280 \
--output_dir ./casmar_large_test --resume /path/to/CASMAR-L \
--num_images 50000 --num_iter 4 \
--cfg 3.5 --re_cfg 3.5 --cfg_schedule linear \
--cond_scale 8 --cond_dim 16 \
--two_diffloss --global_dm --gdm_d 8 --gdm_w 512 \
--eval_bsz 256 --load_epoch -1 \
--head 8 --ratio 4 --cos --evaluate
```

### Evaluate CASMAR-H on ImageNet 256×256

```python
torchrun --nproc_per_node=8 --nnodes=1 \
main_casmar.py \
--img_size 256 --vae_path /path/to/vae --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model casmar_huge --diffloss_d 12 --diffloss_w 1536 \
--output_dir ./casmar_huge_test --resume /path/to/CASMAR-H \
--num_images 50000 --num_iter 12 \
--cfg 3.2 --re_cfg 5.5 --cfg_schedule linear \
--cond_scale 8 --cond_dim 16 \
--two_diffloss --global_dm --gdm_d 12 --gdm_w 768 \
--eval_bsz 256 --load_epoch -1 \
--head 12 --ratio 4 --cos --evaluate
```

## 🎯 Sampling / Generation

You can also use the same evaluation script for unconditional batch sampling or class-conditional generation by adjusting:

- `--num_images`
- `--cfg`
- `--re_cfg`
- `--cfg_schedule`
- checkpoint path in `--resume`

For high-fidelity sampling, we recommend starting from the default configurations reported above.

## 🧪 Method Summary

CASMAR contains three key components:

1. Two-stage coarse-to-fine generation
   - Stage 1 forms a cross-scale scaffold
   - Stage 2 refines details conditioned on the scaffold
2. SAMS with AMR
   - saliency-aware visible-token preservation
   - adaptive masking difficulty calibration
3. SALA
   - local detail aggregation
   - scaffold-guided structural correction

## 📊 Main Results on ImageNet-1K (256×256)

| Model    | #Params | FID (w/o CFG) ↓ | IS (w/o CFG) ↑ | FID (w/ CFG) ↓ | IS (w/ CFG) ↑ |
| -------- | ------- | --------------- | -------------- | -------------- | ------------- |
| CASMAR-B | 254.4M  | 2.55            | 231.20         | 2.21           | 275.4         |
| CASMAR-L | 542.3M  | 2.11            | 251.50         | 1.62           | 305.8         |
| CASMAR-H | 1133.9M | 1.84            | 275.30         | 1.51           | 309.5         |

## 🙏 Acknowledgement

Thanks to the contribution of [MAR](https://github.com/LTH14/mar).