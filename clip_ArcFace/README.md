# ViT-B/32 with Cross-Entropy + ArcFace

This repository contains the source code for training and evaluating a **CLIP ViT-B/32** model enhanced with a **custom classification head and ArcFace loss** for the task of **face image retrieval**.

## Overview

We fine-tune a CLIP-based Vision Transformer (ViT-B/32) for image retrieval by combining the **Cross-Entropy Loss** with **Additive Angular Margin** to improve class separability in the embedding space. This approach is motivated by ArcFace’s effectiveness in face recognition tasks.

The best-performing model in our benchmark used:

* **Backbone**: CLIP ViT-B/32 (visual encoder only)
* **Head**: Linear projection + ArcFace
* **Loss**: Cross-Entropy with Angular Margin (ArcFace)

## Project Structure

```
clip_ArcFace/
├── repository/              # Model checkpoints and outputs
│   ├── all_weights/         # All model weights per epoch
│   ├── best_model/          # Best model based on val loss
│   ├── checkpoints/         # Checkpoints with optimizer state
│   ├── metrics/             # Stored metrics during training
│   └── results/             # Final evaluation outputs
│
├── src/                     # Source code
│   ├── data_loading.py      # Dataloaders and transforms
│   ├── distance.py          # Similarity functions (cosine)
│   ├── loss.py              # Loss functions (ArcFace)
│   ├── model.py             # CLIP-based architecture
│   ├── test.py              # Evaluation loop (retrieval metrics)
│   └── train.py             # Training loop with early stopping
│
├── main.py                  # Entry point for training/testing/validation
├── wandb/                   # Weights & Biases logs
└── README.md                # This file
```

## Training

Run the model with set the model in Main as training = True

This will:
* Load the dataset
  * training
  * validation
  * test
* Start training 
  * tracking train_losses and validation_losses
* Save checkpoints and metrics in `repository/`
  * all_weights: contain the model for each epoch
  * checkpoints: contain the weights for each epoch (allow the resume of the trainings)
  * metrics: contain csv with train_losses and validation_losses

## Testing

Run the model with set the model in Main as training = False
It will test the model in a specific epoch. To perform an overall testing use the other script evaluate_all, that eill crreate a csv that shows epoch and accurancy


Install with:

```bash
pip install -r requirements.txt
```

## Results

The ViT-B/32 + ArcFace model demonstrated strong generalization, especially on face-like datasets.

---
