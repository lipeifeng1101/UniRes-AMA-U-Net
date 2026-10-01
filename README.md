# UniRes-AMA U-Net

## A Unified Coordination Framework for Retinal Vessel Segmentation

**Official PyTorch implementation of the manuscript**

> **A Unified Coordination Framework for Retinal Vessel Segmentation**

**Status:** Under review at *Pattern Analysis and Applications* (2026)

UniRes-AMA U-Net is a retinal vessel segmentation framework designed to coordinate vessel representations across different stages of an encoder-decoder network. The method focuses on improving small-vessel delineation and vascular structural continuity through complementary feature refinement, prediction fusion, and training objectives.

> **Note:** The proposed framework does not impose an explicit topological constraint or guarantee formal topology preservation. Structural continuity is evaluated empirically using standard segmentation metrics together with centerline-based clDice.

---

## 🚀 Overview

Retinal vessel segmentation is an important step in quantitative fundus image analysis. Accurate segmentation remains challenging because small and low-contrast vessels can weaken during repeated downsampling, while features from different encoder and decoder stages may contain different spatial and semantic information.

**UniRes-AMA U-Net** follows a symmetric encoder-decoder architecture and coordinates vascular representations at several stages of the network.

The framework mainly contains the following components:

### 1. Window Attention BotBlock

A lightweight window-based attention block is applied once to the shallow high-resolution representation.

Instead of applying global self-attention to the complete image, the feature map is divided into local non-overlapping windows. This provides controlled spatial interaction among neighboring vessel features while retaining the high spatial resolution available at the beginning of the network.

In the reported configuration:

```text
Window size: 8 × 8
Number of attention heads: 4
```

---

### 2. Unified Residual Block (Uni-ResBlock)

Uni-ResBlock is the main feature extraction unit and integrates two complementary mechanisms:

- **Simplified Spatial Attention (SSA)** emphasizes spatially relevant vessel regions using channel-wise average and maximum pooling.
- **Balanced Vessel Attention (BVA)** uses horizontal, vertical, diagonal, and pointwise convolutions to refine directional responses of elongated vascular structures.

SSA and BVA operate on different intermediate features within the same sequential residual pathway.

Their roles are complementary:

```text
SSA → spatial localization
BVA → directional vessel refinement
```

The BVA enhancement coefficient γ is initialized to:

```text
γ = 0.2
```

---

### 3. Adaptive Multi-Directional Attention (AMA)

AMA is applied to intermediate decoder features before multiscale feature fusion.

It recalibrates feature responses across three complementary planes:

```text
C-W
C-H
H-W
```

where:

```text
C = channel dimension
H = feature height
W = feature width
```

AMA uses lightweight gating operations rather than query-key self-attention.

Its role is to coordinate spatial and channel information during decoding before multiscale feature fusion.

---

### 4. Seq-MLP

Seq-MLP is placed at the bottleneck of the encoder-decoder network.

The bottleneck feature is reshaped into a sequence representation, after which the same affine transformation is applied independently to the channel vector at each spatial position.

The operation can be summarized as:

```text
Bottleneck feature
      ↓
Sequence reshape
      ↓
Pointwise channel projection
      ↓
ReLU
      ↓
Spatial reshape
      ↓
Residual addition + BatchNorm
```

Therefore, **Seq-MLP performs pointwise channel refinement rather than long-range spatial token interaction**.

It changes the channel composition while leaving the spatial index unchanged.

Its spatial mixing scope is comparable to a 1 × 1 convolution.

---

### 5. Adaptive Multi-Branch Fusion (AMBF)

AMBF combines three parallel prediction heads:

```text
P₁ → Main branch
P₂ → Small Vessel branch
P₃ → Continuity branch
```

The three predictions are combined using globally shared learnable coefficients.

The raw fusion parameters are initialized as:

```text
[1.0, 0.3, 0.2]
```

corresponding to:

```text
Main branch
Small Vessel branch
Continuity branch
```

A Softmax operation converts these parameters into normalized fusion coefficients during the forward pass.

Conceptually:

```text
Final Prediction
      =
α₁ × P₁
      +
α₂ × P₂
      +
α₃ × P₃
```

where:

```text
α₁ + α₂ + α₃ = 1
```

The coefficients are global model parameters shared across all input images. They are **not dynamically generated for each individual image**.

The branch names describe their intended refinement roles. They should not be interpreted as independently supervised functional decompositions.

---

## 🎯 Training Objective

The final fused prediction is optimized using three complementary loss components:

```text
L_total = L_main + λ₁ × L_small + λ₂ × L_cont
```

where:

```text
L_main   → main segmentation loss
L_small  → small-vessel weighting loss
L_cont   → finite-difference continuity loss
```

### Main Segmentation Loss

The main segmentation objective combines:

```text
Binary Cross-Entropy Loss
+
Soft Dice Loss
```

### Small-Vessel Weighting

The small-vessel term gives greater emphasis to vessels with small calibre.

The maximum microvessel diameter is set to:

```text
4 pixels
```

on the 512 × 512 training grid.

### Local Continuity Supervision

The continuity term uses finite differences to penalize mismatched local transitions between the predicted vessel map and the reference mask.

The main experiments use:

```text
λ₁ = 0.5
λ₂ = 0.5
```

These objectives encourage fine-vessel reconstruction and local structural consistency, but they do **not** impose a global topological invariant.

---

## 📊 Quantitative Results

UniRes-AMA U-Net was evaluated on three public retinal vessel segmentation datasets:

- DRIVE
- STARE
- CHASEDB1

| Dataset | AUC | F1-score | clDice | ACC | SE | SP |
|---|---:|---:|---:|---:|---:|---:|
| **DRIVE** | **0.9871** | **0.8369** | **0.8421** | **0.9693** | **0.8465** | **0.9849** |
| **STARE** | **0.9872** | **0.8494** | **0.8530** | **0.9773** | **0.8516** | **0.9892** |
| **CHASEDB1** | **0.9891** | **0.8342** | **0.8395** | **0.9761** | **0.8662** | **0.9872** |

The reported clDice values provide complementary evidence based on the agreement between predicted and reference vessel centerlines.

The three-dataset results above are **single-run aggregate point estimates** and are not reported as mean ± standard deviation over repeated training runs.

---

## 📈 Evaluation Metrics

The following metrics are reported:

```text
AUC      → Area Under the ROC Curve
F1-score → Harmonic mean of precision and recall
clDice   → Centerline-based Dice measure
ACC      → Accuracy
SE       → Sensitivity
SP       → Specificity
```

clDice is used as a complementary indicator of vascular structural continuity.

It should not be interpreted as evidence that the proposed network mathematically guarantees topology preservation.

---

## ⏱️ Computational Efficiency

All runtime measurements reported in the manuscript were obtained using:

```text
Input resolution: 512 × 512
GPU: NVIDIA GeForce RTX 2080 Ti
```

| Model | Parameters | Training Time / Epoch | Inference Time / Image | FPS |
|---|---:|---:|---:|---:|
| U-Net | 31.0 M | 2.80 s | 0.150 s | 6.7 |
| Attention U-Net | 35.5 M | 3.10 s | 0.162 s | 6.2 |
| **UniRes-AMA U-Net** | **46.3 M** | **3.38 s** | **0.174 s** | **5.7** |

Runtime depends on hardware and implementation details.

Compared with vanilla U-Net, UniRes-AMA U-Net introduces additional computational overhead due to the added feature refinement and prediction fusion mechanisms.

Under the reported implementation:

```text
Inference time: 0.174 s / image
FPS: 5.7
Parameters: 46.3 M
```

The current implementation is therefore intended primarily for **offline or batch retinal image analysis rather than real-time diagnostic use**.

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

Hardware used for the reported experiments:

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

### DRIVE

**Digital Retinal Images for Vessel Extraction**

```text
Total images: 40
Training pool: 20
Test images: 20
Original resolution: 584 × 565
Network input: 512 × 512
```

Download:

https://drive.grand-challenge.org/

### STARE

**STructured Analysis of the REtina**

```text
Total images: 20
Training pool: 15
Test images: 5
Original resolution: 700 × 605
Network input: 512 × 512
```

Download:

https://cecas.clemson.edu/~ahoover/stare/

### CHASEDB1

**Child Heart And Health Study in England**

```text
Total images: 28
Training pool: 23
Test images: 5
Original resolution: 990 × 960
Network input: 512 × 512
```

Download:

https://blogs.kingston.ac.uk/retinal/chasedb1/

---

## 📁 Directory Structure

Download the datasets and organize them under the `data/` directory.

```text
data/
├── DRIVE/
│   ├── training/
│   └── test/
├── STARE/
└── CHASEDB1/
```

Adjust the dataset paths if your local directory structure is different.

---

## 🔧 Preprocessing

All images undergo the same general preprocessing pipeline:

```text
Fundus image
    ↓
Illumination correction
    ↓
Contrast enhancement
    ↓
Resize to 512 × 512
    ↓
Normalization
    ↓
Network input
```

During training, data augmentation includes:

- random horizontal flipping,
- random vertical flipping,
- random rotation,
- random scaling.

The network input remains:

```text
512 × 512
```

during training.

**No 64 × 64 training crops are used in the reported experiments.**

---

## ⚙️ Training Configuration

The main training configuration reported in the manuscript is summarized below.

| Setting | Value |
|---|---|
| Framework | PyTorch 1.12 |
| GPU | NVIDIA GeForce RTX 2080 Ti |
| Optimizer | Adam |
| Maximum epochs | 100 |
| Initial learning rate | 5 × 10⁻⁴ |
| Scheduler | Cosine annealing |
| Weight decay | Not set |
| Early stopping | Patience = 50 epochs |
| Main experiment seed | 2021 |
| Window size | 8 × 8 |
| Attention heads | 4 |
| Uni-ResBlock γ initialization | 0.2 |
| AMBF raw weights | [1.0, 0.3, 0.2] |
| λ₁ | 0.5 |
| λ₂ | 0.5 |
| Maximum microvessel diameter | 4 pixels |
| Input resolution | 512 × 512 |

The available optimizer configuration used for the reported experiments does **not** set weight decay.

---

## 🔬 Experimental Protocol

### Main Three-Dataset Experiments

The test sets were kept separate from model selection and early stopping.

For the originally reported main experiments, the preserved experimental records indicate that a validation subset was sampled from each training pool without overlap with the corresponding test set.

However, the exact internal validation ratios and image identifiers of these original runs cannot be reconstructed from the preserved experimental configuration.

No k-fold cross-validation results are reported for the main three-dataset experiments.

---

### DRIVE Revision Experiments

The additional DRIVE ablation and loss-weight sensitivity experiments use a separately documented source-image-level protocol.

The original DRIVE training pool contains:

```text
20 source images
```

For each revision experiment:

```text
Training source images:   18
Validation source images:  2
Test source images:       20
```

The source-image split is performed **before** generating training sampling entries.

The 18 training source images generate:

```text
540 fixed training sampling entries
```

The two validation source images generate:

```text
60 fixed validation sampling entries
```

These entries are reused across epochs.

Importantly:

> The 540 training indices and 60 validation indices are sampling entries and do **not** represent 64 × 64 image crops.

No source image is shared between the training and validation sampling lists.

---

### Five-Run Ablation Study

The DRIVE ablation study uses five independent training runs:

```text
Seed 2021
Seed 2022
Seed 2023
Seed 2024
Seed 2025
```

For each run:

- all compared configurations use the same source-image split;
- the split is regenerated between independent runs;
- all other training and evaluation settings are kept fixed;
- evaluation is performed on the same held-out 20-image DRIVE test set.

---

### Loss-Weight Sensitivity Analysis

The loss-weight sensitivity experiment uses:

```text
Seed: 2021
Training / validation source split: 18 / 2
```

The sensitivity analysis was conducted after the main configuration had already been fixed.

The test-set results were not used to select the loss weights for the main experiments.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/lipeifeng1101/UniRes-AMA-U-Net.git
cd UniRes-AMA-U-Net
```

---

### 2. Prepare the Dataset

Download the desired retinal vessel dataset and place it under the `data/` directory.

For example:

```text
data/
└── DRIVE/
    ├── training/
    └── test/
```

---

### 3. Training

For example, to train UniRes-AMA U-Net on DRIVE:

```bash
python train.py --dataset DRIVE --epochs 100 --lr 0.0005
```

The reported configuration uses:

```text
Optimizer: Adam
Initial learning rate: 5 × 10⁻⁴
Scheduler: Cosine annealing
Maximum epochs: 100
Early stopping patience: 50
```

The optimizer configuration used in the reported experiments does not set weight decay.

---

### 4. Testing

Evaluate a trained model using:

```bash
python test.py --dataset DRIVE --weights checkpoints/best_model.pth
```

The evaluation reports:

```text
AUC
F1-score
clDice
Accuracy
Sensitivity
Specificity
```

The same skeletonization procedure is applied consistently to predictions and ground-truth masks when computing clDice across the evaluated datasets.

---

## 🧩 Role of Each Component

A concise overview of the framework is given below.

| Component | Network Stage | Main Role |
|---|---|---|
| Window Attention BotBlock | Shallow feature stage | Local spatial interaction |
| SSA | Uni-ResBlock | Spatial vessel localization |
| BVA | Uni-ResBlock | Directional vessel refinement |
| AMA | Decoder | Spatial-channel feature coordination |
| Seq-MLP | Bottleneck | Pointwise channel refinement |
| AMBF | Prediction stage | Learnable fusion of three prediction heads |
| Calibre weighting | Training objective | Emphasis on small vessels |
| Finite-difference loss | Training objective | Local structural continuity |

These mechanisms operate at different stages of the network and jointly support the representation of fine vessels and vascular structural continuity.

---

## 📌 Interpretation of Structural Results

UniRes-AMA U-Net is designed to improve vascular structural continuity through coordinated feature representation and local supervision.

Specifically:

```text
BVA
→ models local directional vessel responses

AMA
→ recalibrates decoder features before multiscale fusion

Seq-MLP
→ refines bottleneck channel responses

AMBF
→ combines parallel prediction heads

Calibre-based weighting
→ emphasizes small vessels

Finite-difference supervision
→ penalizes mismatched local transitions
```

The reported clDice results provide centerline-based evidence of vascular structural continuity.

However, UniRes-AMA U-Net does **not** impose:

- a persistent-homology constraint,
- an explicit skeleton supervision branch,
- a global topology-preserving operator,
- or a mathematical topological invariant.

Accordingly, the method should be interpreted as a segmentation framework that encourages structural continuity rather than one that formally guarantees topology preservation.

---

## ⚠️ Current Scope and Limitations

The current evaluation is limited to three public retinal fundus benchmarks:

```text
DRIVE
STARE
CHASEDB1
```

Several limitations should be considered when interpreting the reported results:

1. The main three-dataset results are single-run point estimates rather than repeated-run mean ± standard deviation results.

2. For the originally reported main experiments, the exact internal validation ratios and image identifiers cannot be fully reconstructed from the preserved experimental records.

3. Repeated-run variability analysis is currently provided for the DRIVE ablation study.

4. The framework encourages structural consistency but does not mathematically guarantee topology preservation.

5. The current evaluation does not establish generalization to other vascular imaging modalities or larger external clinical datasets.

6. The measured inference time is 0.174 s per image, corresponding to 5.7 FPS on an NVIDIA GeForce RTX 2080 Ti.

Therefore, the current implementation is primarily intended for **offline or batch retinal image analysis rather than real-time diagnostic use**.

---

## 📜 Citation

The manuscript is currently under review.

If you find this repository useful, please consider citing:

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

For questions regarding the implementation or experimental setup, please open an issue in this repository.

---

## Acknowledgements

We thank the developers and maintainers of the DRIVE, STARE, and CHASEDB1 datasets and the open-source research community supporting retinal vessel segmentation research.
