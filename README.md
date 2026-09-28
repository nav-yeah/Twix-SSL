# Twix

**Can a model learn to see without being taught what it's looking at?**

Twix is a self-supervised learning (SSL) model built on **Barlow Twins** with a **ResNet-18** backbone. It learns image representations from unlabeled data alone, and is then evaluated with a linear probe on STL-10.

> This project was made as part of **Tilde 5.0**, the summer mentoring program organised by **[Homebrew](https://github.com/homebrew-ec-foss)**, the FOSS community of **HSP (Hackerspace) PESU ECC**.

---

## Demo Videos

| # | Video | Link |
|---|-------|------|
| 1 | Project Overview | [Watch on YouTube](https://www.youtube.com/watch?v=x53T88ed6WU&t=5766s) |
| 2 | Training & Implementation Walkthrough | [Watch on YouTube](https://www.youtube.com/watch?v=lDwkOKVd_NQ&t=11s) |
| 3 | Results & Evaluation | [Watch on YouTube](https://www.youtube.com/watch?v=LELDmFJF5Pc) |

---

## Team

**Mentors**
- Andey Hemanth
- Aditi Vignesh

**Mentees**
- Navya Suresh
- Aaditya Prakash
- Sinchana S

---

## How It Works

Twix runs in two phases.

**Phase 1: Pretrain (no labels used)**
1. Take an unlabeled image from STL-10 (100,000 unlabeled images).
2. Create two augmented views (A and B) of the same image using random crops and colour changes.
3. Pass both views through a shared **ResNet-18 encoder**, then a **projection head**.
4. Apply the **Barlow Twins loss**, which pushes the cross-correlation matrix of the two views towards the **identity matrix**. This makes matching views agree while keeping features non-redundant.

**Phase 2: Evaluate (linear probe)**
1. Keep the pretrained encoder and **freeze** its weights.
2. Train only a **linear classifier** on top, using 5,000 labeled training images.
3. Measure accuracy on the 8,000-image STL-10 test set.

---

## Design Choices

| Component | Choice | Purpose |
|-----------|--------|---------|
| Input | 96 × 96 RGB images | Input format from STL-10 |
| Data augmentation | Two augmented views per image | Different views of the same image for Barlow Twins |
| Encoder / backbone | ResNet-18 | Extracts visual features |
| Encoder output | 512-dimensional vector | Represents the image in feature space |
| Projection head | Linear MLP | Maps features into the space used by the loss |
| Loss function | Barlow Twins loss | Maximises similarity between views, reduces feature redundancy |
| Initial optimizer | Adam | Baseline |
| Initial LR schedule | Constant | Baseline |
| Initial training | 100 epochs | Baseline experiment |
| Updated optimizer | AdamW | Adam with decoupled weight decay |
| Batch size | 128 | More samples for batch-level statistics |
| Updated training | 200 epochs | Longer representation learning |
| LR warm-up | Added | Stable training at the start |
| LR scheduler | Cosine annealing | Finer convergence towards the end |

---

## Progress

1. Built the linear classifier
2. Tested the model
3. Tuned parameters to improve accuracy:
   - Switched the optimizer from **Adam** to **AdamW**
   - Added **learning-rate warm-up** and **cosine annealing** to stabilise the embeddings
   - Increased the number of **training epochs**
   - Added more features to the **image transformations**

---

## Results

| Metric | Value |
|--------|-------|
| **Linear-probe test accuracy** | **78.79%** |
| Random-guessing baseline (10 classes) | 10% |
| Evaluation set | STL-10 test set (8,000 images) |

The encoder learned useful visual features from unlabeled images alone. A simple linear classifier on top of the frozen encoder is enough to separate the 10 classes.

---

## Repository Structure

```
Twix-SSL/
├── barlow_run/                    # Training run outputs
├── checkpoints/                   # Intermediate checkpoints
├── codes.py                       # Helper code
├── dataset.py                     # STL-10 loading and augmentations
├── model.py                       # ResNet-18 encoder + projection head
├── loss.py                        # Barlow Twins loss
├── train.py                       # Self-supervised pretraining
├── linear_probe.py                # Trains the linear classifier on the frozen encoder
├── linear_eval.py                 # Evaluates the linear classifier on the test set
├── encoder_checkpoint_final.pth   # Final pretrained encoder weights
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch and torchvision
- A GPU is recommended for pretraining

### Installation

```bash
git clone https://github.com/nav-yeah/Twix-SSL.git
cd Twix-SSL
pip install torch torchvision
```

### Usage

```bash
# 1. Pretrain the encoder with Barlow Twins (no labels)
python train.py

# 2. Train a linear classifier on the frozen encoder
python linear_probe.py

# 3. Evaluate on the STL-10 test set
python linear_eval.py
```

> The pretrained encoder (`encoder_checkpoint_final.pth`) is included, so you can skip step 1 and go straight to the linear probe.

---

## Future Work

1. Ablation studies (augmentations, λ, batch size)
2. Extend pretraining to ImageNet (domain transfer test)
3. Compare a frozen linear probe against full fine-tuning
4. Benchmark against SimCLR and BYOL on the same setup
