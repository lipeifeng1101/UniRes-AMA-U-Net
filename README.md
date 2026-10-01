# UniRes-AMA U-Net

## A Unified Coordination Framework for Retinal Vessel Segmentation

**Official PyTorch implementation of the manuscript**

> **A Unified Coordination Framework for Retinal Vessel Segmentation**

**Status:** Under review at *Pattern Analysis and Applications* (2026)

UniRes-AMA U-Net is a retinal vessel segmentation framework designed to coordinate vessel representations across different stages of an encoder-decoder network. The method focuses on improving the delineation of small vessels and vascular structural continuity through complementary feature refinement, prediction fusion, and training objectives.

> **Important:** The proposed framework does not impose an explicit topological constraint or guarantee formal topology preservation. Structural continuity is evaluated empirically using standard segmentation metrics together with centerline-based clDice.

---

## 🚀 Overview

Retinal vessel segmentation is an important step in quantitative fundus image analysis. Accurate segmentation remains challenging because small and low-contrast vessels can weaken during downsampling, while features from different encoder and decoder stages may contain different spatial and semantic information.

**UniRes-AMA U-Net** follows a symmetric encoder-decoder architecture and coordinates vessel representations at several stages of the network.

The framework contains the following main components:

### 1. Window Attention BotBlock

A lightweight window-based attention block is applied once to the shallow high-resolution representation.

Instead of performing global self-attention over the complete image, the feature map is divided into local non-overlapping windows. This provides controlled spatial interaction between neighboring vessel features while retaining the high spatial resolution available at the beginning of the network.

### 2. Unified Residual Block (Uni-ResBlock)

Uni-ResBlock is the main feature extraction unit and integrates two complementary mechanisms:

- **Simplified Spatial Attention (SSA)** emphasizes spatially relevant vessel regions using channel-wise average and maximum pooling.
- **Balanced Vessel Attention (BVA)** uses horizontal, vertical, diagonal, and pointwise convolutions to refine directional responses of elongated vascular structures.

SSA and BVA operate on different intermediate features within the same sequential residual pathway and provide complementary spatial and directional refinement.

### 3. Adaptive Multi-Directional Attention (AMA)

AMA is applied to intermediate decoder features before multiscale feature fusion.

It recalibrates feature responses across three complementary planes:

- \(C-W\)
- \(C-H\)
- \(H-W\)

AMA uses lightweight gating operations rather than query-key self-attention. Its role is to coordinate spatial and channel information during decoding.

### 4. Seq-MLP

Seq-MLP is placed at the bottleneck of the encoder-decoder network.

The bottleneck feature is reshaped into a sequence representation, after which the same affine transformation is applied independently to the channel vector at each spatial position.

Therefore, **Seq-MLP performs pointwise channel refinement rather than long-range spatial token interaction**. It changes the channel composition while leaving the spatial index unchanged.

### 5. Adaptive Multi-Branch Fusion (AMBF)

AMBF combines three parallel prediction heads:

- **Main branch**
- **Small Vessel branch**
- **Continuity branch**

The three predictions are combined using globally shared learnable coefficients. The raw fusion parameters are optimized jointly with the network and normalized using Softmax during the forward pass.

The coefficients are shared across input samples and are **not dynamically generated for each image**.

The names of the auxiliary branches describe their intended refinement roles. They should not be interpreted as independently supervised functional decompositions.

---

## 🎯 Training Objective

The final fused prediction is optimized using

\[
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{main}}
+
\lambda_1\mathcal{L}_{\mathrm{small}}
+
\lambda_2\mathcal{L}_{\mathrm{cont}}.
\]

The objective contains:

- **Main segmentation loss:** binary cross-entropy + soft Dice loss.
- **Small-vessel weighting term:** gives greater emphasis to vessels with small calibre.
- **Finite-difference continuity term:** penalizes mismatched local transitions between the predicted vessel map and the reference mask.

The main experiments use

```text
λ1 = 0.5
λ2 = 0.5
```

The maximum microvessel diameter is set to **4 pixels** on the \(512 \times 512\) training grid.

These objectives encourage fine-vessel reconstruction and local structural consistency but do not impose a global topological invariant.

---

## 📊 Quantitative Results

UniRes-AMA U-Net was evaluated on three public retinal vessel segmentation datasets: **DRIVE**, **STARE**, and **CHASEDB1**.

| Dataset | AUC | F1-score | clDice | ACC | SE | SP |
|---|---:|---:|---:|---:|---:|---:|
| **DRIVE** | **0.9871** | **0.8369** | **0.8421** | **0.9693** | **0.8465** | **0.9849** |
| **STARE** | **0.9872** | **0.8494** | **0.8530** | **0.9773** | **0.8516** | **0.9892** |
| **CHASEDB1** | **0.9891** | **0.8342** | **0.8395** | **0.9761** | **0.8662** | **0.9872** |

The clDice values provide complementary evidence based on agreement between the predicted and reference vessel centerlines.

The three-dataset results above are **single-run aggregate point estimates** and are not reported as mean ± standard deviation over repeated training runs.

---

## ⏱️ Computational Efficiency

All runtime measurements reported in the manuscript were obtained using an input resolution of \(512 \times 512\) on a single **NVIDIA GeForce RTX 2080 Ti** GPU.

| Model | Parameters | Training Time / Epoch | Inference Time / Image | FPS |
|---|---:|---:|---:|---:|
| U-Net | 31.0 M | 2.80 s | 0.150 s | 6.7 |
| Attention U-Net | 35.5 M | 3.10 s | 0.162 s | 6.2 |
| **UniRes-AMA U-Net** | **46.3 M** | **3.38 s** | **0.174 s** | **5.7** |

Runtime depends on hardware and implementation details.

The additional feature refinement and structural modeling increase the computational cost relative to vanilla U-Net. Under the reported implementation, UniRes-AMA U-Net is intended primarily for **offline or batch retinal image analysis rather than real-time diagnostic use**.

---

## 🛠️ Requirements

The experiments were implemented using:

```text
Python >= 3.8
PyTorch == 1.12
Torchvision
NumPy
SciPy
scikit-learn
OpenCV
```

The experiments reported in the manuscript were performed on:

```text
NVIDIA GeForce RTX 2080 Ti
```

---

## 📂 Data Preparation

The framework was evaluated on DRIVE, STARE, and CHASEDB1.

| Dataset | Training Pool | Test Set | Total | Original Resolution | Network Input | Format |
|---|---:|---:|---:|---|---|---|
| DRIVE | 20 | 20 | 40 | 584 × 565 | 512 × 512 | `.tif` |
| STARE | 15 | 5 | 20 | 700 × 605 | 512 × 512 | `.ppm` |
| CHASEDB1 | 23 | 5 | 28 | 990 × 960 | 512 × 512 | `.jpg` |

### Dataset Links

**DRIVE**

https://drive.grand-challenge.org/

**STARE**

https://cecas.clemson.edu/~ahoover/stare/

**CHASEDB1**

https://blogs.kingston.ac.uk/retinal/chasedb1/

### Suggested Directory Structure

```text
data/
├── DRIVE/
│   ├── training/
│   └── test/
├── STARE/
└── CHASEDB1/
```

---

## 🔧 Preprocessing

All images are processed using the same general preprocessing pipeline:

1. Illumination correction.
2. Contrast enhancement.
3. Resizing to \(512 \times 512\).
4. Intensity normalization.

During training, data augmentation includes:

- random horizontal flipping,
- random vertical flipping,
- random rotation,
- random scaling.

The network is trained using \(512 \times 512\) inputs.

**No \(64 \times 64\) training crops are used in the reported experiments.**

---

## ⚙️ Training Configuration

The main training configuration reported in the manuscript is:

| Setting | Value |
|---|---|
| Framework | PyTorch 1.12 |
| GPU | NVIDIA GeForce RTX 2080 Ti |
| Optimizer | Adam |
| Maximum epochs | 100 |
| Initial learning rate | \(5 \times 10^{-4}\) |
| Scheduler | Cosine annealing |
| Weight decay | Not set in the available optimizer configuration |
| Early stopping | Patience = 50 epochs |
| Main experiment seed | 2021 |
| Uni-ResBlock \(\gamma\) initialization | 0.2 |
| AMBF raw weights | \([1.0, 0.3, 0.2]\) |
| \(\lambda_1\) | 0.5 |
| \(\lambda_2\) | 0.5 |
| Maximum microvessel diameter | 4 pixels |

The AMBF raw coefficients correspond to the Main, Small Vessel, and Continuity branches and are normalized using Softmax during inference and training.

---

## 🔬 Experimental Protocol

### Main Three-Dataset Experiments

The test sets were kept separate from model selection and early stopping.

For the originally reported main experiments, the preserved experimental records indicate that an internal validation subset was sampled from each training pool without overlap with the test set.

The exact internal validation ratios and image identifiers of these original runs are not available in the preserved configuration.

### DRIVE Revision Experiments

The additional DRIVE ablation and loss-weight sensitivity experiments follow a separately documented source-image-level protocol.

For each run:

```text
Original DRIVE training pool: 20 images
Training source images:       18
Validation source images:      2
Held-out DRIVE test images:   20
```

The 18 training images generate **540 fixed training sampling entries**, while the two validation images generate **60 fixed validation sampling entries**.

These sampling entries **do not represent 64 × 64 image crops**.

For the five-run ablation study:

```text
Seeds: 2021, 2022, 2023, 2024, 2025
```

Within each run, all compared configurations use the same source-image split. The source-image split is regenerated between independent runs.

The loss-weight sensitivity analysis uses seed **2021** and a fixed 18/2 source-image split.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/lipeifeng1101/UniRes-AMA-U-Net.git
cd UniRes-AMA-U-Net
```

### 2. Training

For example, to train the model on DRIVE:

```bash
python train.py --dataset DRIVE --epochs 100 --lr 0.0005
```

The reported configuration uses Adam with an initial learning rate of \(5\times10^{-4}\), cosine annealing, and early stopping with a patience of 50 epochs.

The available optimizer configuration used for the reported experiments **does not set weight decay**.

### 3. Testing

To evaluate a trained model:

```bash
python test.py --dataset DRIVE --weights checkpoints/best_model.pth
```

The evaluation reports:

- AUC
- F1-score
- clDice
- Accuracy
- Sensitivity
- Specificity

The same skeletonization procedure is used for predictions and ground-truth masks when computing clDice across the evaluated datasets.

---

## 📌 Interpretation of Structural Results

UniRes-AMA U-Net is designed to improve vascular structural continuity through coordinated feature representation and local supervision.

In particular:

- BVA models local directional vessel responses.
- AMA recalibrates decoder features before multiscale fusion.
- Seq-MLP refines bottleneck channel responses.
- AMBF combines parallel prediction heads.
- Calibre-based weighting emphasizes small vessels.
- Finite-difference supervision penalizes mismatched local transitions.

The reported **clDice** results provide centerline-based evidence of structural continuity.

However, the method **does not impose an explicit topology-preserving operator, persistent-homology constraint, or global topological invariant**. The reported results should therefore be interpreted as empirical segmentation and structural-continuity performance under the evaluated settings.

---

## ⚠️ Current Scope and Limitations

The current evaluation is limited to the three retinal fundus benchmarks used in the manuscript.

The reported evidence does not establish performance on other vascular imaging modalities or larger external clinical datasets.

In addition:

- the original three-dataset main results are single-run point estimates;
- the exact internal validation subsets of the original main experiments cannot be fully reconstructed from the preserved records;
- repeated-run variability analysis is currently provided for the DRIVE ablation experiments;
- UniRes-AMA U-Net does not provide a mathematical guarantee of topology preservation;
- the measured inference speed is 5.7 FPS on the reported hardware and is therefore more appropriate for offline or batch analysis than real-time diagnostic use.

---

## 📜 Citation

If you find this repository useful, please consider citing the manuscript:

```bibtex
@unpublished{li2026uniresama,
  title  = {A Unified Coordination Framework for Retinal Vessel Segmentation},
  author = {Li, Peifeng and Meng, Xianjing and Li, Hengwu and Dou, Changhao},
  note   = {Manuscript under review at Pattern Analysis and Applications},
  year   = {2026}
}
```

The citation information will be updated after publication.

---

## 📬 Contact

For questions regarding the implementation or experiments, please open an issue in this repository.

---

## Acknowledgement

We thank the developers and maintainers of the DRIVE, STARE, and CHASEDB1 datasets and the open-source research community that supports retinal vessel segmentation research.
