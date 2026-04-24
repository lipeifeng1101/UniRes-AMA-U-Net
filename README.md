# UniRes-AMA-U-Net
**Official Implementation of**  
_"Unified Coordination Framework for Topology-Preserving Retinal Vessel Segmentation"_  
*(Under Review, The visual computer 2026)*

This repository contains the official PyTorch implementation of the paper:

🚀 Introduction

Retinal vessel segmentation serves as a foundational task in ophthalmic image analysis and computer-aided diagnosis. However, existing encoder-decoder models suffer from cross-level feature misalignment, scale inconsistency, and local-global isolation, leading to fragmented microvessel predictions and poor topological integrity.

UniRes-AMA U-Net addresses these challenges through a unified coordination framework that integrates four core components:

1.Unified Residual Block (Uni-ResBlock): Synergistically integrates Simplified Spatial Attention (SSA) and Balanced Vessel Attention (BVA) for local and anisotropic feature enhancement.

2.Adaptive Multi-Directional Attention (AMA): A cross-scale coordination module that enforces feature consistency along three orthogonal planes (C-W, C-H, H-W) during the decoding stage.

3.Sequence-based MLP Global Context Module (Seq-MLP): Captures long-range structural dependencies and vascular topology efficiently at the deepest encoder stage.

4.Adaptive Multi-Branch Fusion (AMBF): Adaptively weights contributions from main vessel, microvessel, and connectivity-aware branches to ensure topological integrity in the final prediction.

📊 Performance Metrics

Extensive experiments on three public datasets demonstrate that our method achieves competitive performance, particularly in topology-preserving metrics (clDice) and segmentation accuracy.

| Dataset | AUC | F1-score | clDice | ACC | SE | SP |

| DRIVE | 0.9871 | 0.8369 | 0.8421 | 0.9693 | 0.8465 | 0.9849 |

| STARE | 0.9872 | 0.8494 | 0.8530 | 0.9773 | 0.8516 | 0.9892 |

| CHASEDB1 | 0.9891 | 0.8342 | 0.8395 | 0.9761 | 0.8662 | 0.9872 |

🛠️ Requirements

Python >= 3.8
PyTorch == 1.12
Torchvision
NumPy, SciPy, scikit-learn, OpenCV
(Note: All experiments and benchmarks in the paper were conducted on a single NVIDIA GeForce RTX 2080 Ti GPU.)

📂 Data Preparation

Please download the publicly available datasets and organize them into the data/ directory as follows:
data/
├── DRIVE/
│   ├── training/
│   └── test/
├── STARE/
└── CHASEDB1/

Preprocessing Details: All images undergo illumination correction, contrast enhancement, and normalization. They are resized to 512x512 resolution. Data augmentation strategies, including random rotations, horizontal flips, and scale perturbations, are applied during training.

⚙️ Quick Start

1. Training
To train the model from scratch using the default hyperparameters (e.g., on the DRIVE dataset):
python train.py --dataset DRIVE --epochs 100 --lr 0.0005
Optimization Configuration: The model is trained using the Adam optimizer with an initial learning rate of , a cosine annealing scheduler, and a weight decay of . Early stopping with a patience of 50 epochs is applied.
2. Testing
To evaluate the model using pre-trained weights and output evaluation metrics (clDice, AUC, F1, etc.):
python test.py --dataset DRIVE --weights checkpoints/best_model.pth


📜 Citation

If you find this code or research helpful in your work, please consider citing our paper:
@article{li2026uniresama,
  title={Unified Coordination Framework for Topology-Preserving Retinal Vessel Segmentation},
  author={Li, Peifeng and Meng, Xianjing and Li, Hengwu and Dou, Changhao},
  journal={The visual computer},
  year={2026}
}



