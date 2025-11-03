# CASMAR

![Qualitative Results](./demo/results.jpg)

**Cascaded Masked Autoregression for Image Generation: Unified Salience Scheduling and Proximal Self-Attention Synergy**<be>

This is the official repository for the Paper "Cascaded Masked Autoregression for Image Generation: Unified Salience Scheduling and Proximal Self-Attention Synergy"



## Overview

We present **CASMAR (Cascaded Masked Autoregression)** — a two-stage, coarse-to-fine framework for high-resolution image generation in continuous latents. **Stage 1** predicts a compact set of scaffold tokens to stabilize global structure; **Stage 2** reconstructs masked regions to recover fine details. A **Salience-Aware Mask Scheduling (SAMS)** strategy with an **Adaptive Mask Ratio (AMR)** tailors masking to each image, while **Proximal Self-Attention (PSA)** confines attention to local neighborhoods for near-linear scaling.

#### 🔍 What We’re Solving

- **Unreliable global context** at early steps in single-scale AR
- **Fixed/random masking** that ignores per-image structural diversity
- **Quadratic cost** and weak cross-window communication from global self-attention at large resolutions
- **Train–inference mismatch** when conditioning refinement on ground truth

#### 🧠 How CASMAR Works

- **Stage 1 — Low-res scaffold.** Predict a small set of key tokens to establish a reliable global layout.
- **Stage 2 — PSA–DiT refinement.** Reconstruct masked regions at higher resolution; PSA restricts attention to a dilatable (k\times k) neighborhood, reducing (\mathcal{O}(N^2)) to (\mathcal{O}(N k^2)), while a Diffusion-Transformer head preserves long-range interactions.
- **SAMS + AMR.** Rank tokens by saliency and **adapt the mask ratio per sample**, dynamically balancing visible vs. predicted regions during training and inference.
- **Unified diffusion objective.** Continuous tokens are trained with a diffusion loss; **Pred-Cond** conditions Stage 2 on Stage 1 predictions to reduce train–test mismatch.

#### 📈 Highlights

- **ImageNet-1K & MS-COCO:** Strong FID/IS under both class- and text-conditional settings, while maintaining or improving **Precision/Recall**.
- **Efficiency:** PSA cuts memory/compute at high resolution; ablations show **complementary gains** from the two-stage design, SAMS, and PSA.
- **Scalable & controllable:** Works naturally with temperature, top-p, and CFG in an AR/MAR sampling loop.

> TL;DR — **Coarse-to-fine masked autoregression + salience-aware masking + proximal attention** → scalable, efficient high-res generation.


## 🔥 Updates

- [x] **\[2025.11.03\]** Upload inference code and pretrained class-conditional CASMAR models trained on ImageNet 256x256.

## 🏃🏼 Inference

<details open>
<summary><strong>Environment Requirement</strong></summary>



Clone the repo:

```
git clone https://github.com/HiDream-ai/casmar.git
cd casmar
```

</details>

<details open>
<summary><strong>Evaluation</strong></summary>


Evaluate CASMAR-B on ImageNet256x256:

```
torchrun --nproc_per_node=8 --nnodes=1 main_casmar.py --img_size 256 --vae_path /path/to/vae --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model casmar_base --diffloss_d 6 --diffloss_w 1024 --output_dir ./casmar_base_test --resume /path/to/CASMAR-B  --num_images 50000 --num_iter 4 --cfg 2.5 --re_cfg 2.7 --cfg_schedule linear --cond_scale 8 --cond_dim 16 --two_diffloss --global_dm --gdm_d 6 --gdm_w 512 --eval_bsz 256 --load_epoch -1 --head 8 --ratio 4 --cos --evaluate
```

Evaluate CASMAR-L on ImageNet256x256:

```
torchrun --nproc_per_node=8 --nnodes=1 main_casmar.py --img_size 256 --vae_path /path/to/vae --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model casmar_large --diffloss_d 8 --diffloss_w 1280 --output_dir ./casmar_large_test --resume /path/to/CASMAR-L  --num_images 50000 --num_iter 4 --cfg 3.5 --re_cfg 3.5 --cfg_schedule linear --cond_scale 8 --cond_dim 16 --two_diffloss --global_dm --gdm_d 8 --gdm_w 512 --eval_bsz 256 --load_epoch -1 --head 8 --ratio 4 --cos --evaluate
```

Evaluate CASMAR-H on ImageNet256x256:

```
torchrun --nproc_per_node=8 --nnodes=1 main_casmar.py --img_size 256 --vae_path /path/to/vae --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model casmar_huge --diffloss_d 12 --diffloss_w 1536 --output_dir ./casmar_huge_test --resume /path/to/CASMAR-H  --num_images 50000 --num_iter 12 --cfg 3.2 --re_cfg 5.5 --cfg_schedule linear --cond_scale 8 --cond_dim 16 --two_diffloss --global_dm --gdm_d 12 --gdm_w 768 --eval_bsz 256 --load_epoch -1 --head 12 --ratio 4 --cos --evaluate
```

</details>






## 💖 Acknowledgement

<span id="acknowledgement"></span>

Thanks to the contribution of [MAR](https://github.com/LTH14/mar)
