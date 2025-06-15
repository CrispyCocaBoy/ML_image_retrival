# ViT-B/32 Siamese with Contrastive Loss

This folder contains the implementation of a **Siamese Network** based on **CLIP ViT-B/32**, trained with **Contrastive Loss** for the task of **face image retrieval**.

## 🧠 Overview

In this approach, we fine-tune a **Siamese architecture** where two input images are passed through **shared CLIP ViT-B/32 encoders** to produce embeddings. These are then compared using **cosine distance**, and the model is trained to distinguish similar from dissimilar pairs using **Contrastive Loss**.

The training process focuses on maximizing similarity between positive pairs (same identity) and minimizing it for negative pairs (different identities).

## 🧱 Architecture Summary

* **Encoder**: CLIP ViT-B/32 (visual branch only)
* **Shared Weights**: Yes (Siamese structure)
* **Loss**: Contrastive Loss
* **Distance**: Cosine similarity

---

## 📁 Project Structure

```
siamese_network/
├── repository/              # Saved models and training artifacts
│   ├── all_weights/         # Weights saved after each epoch
│   ├── best_model/          # Best weights based on validation performance
│   └── checkpoints/         # Includes optimizer and training state
│
├── src/                     # Source code
│   ├── data_loading.py      # Dataset loading and pair construction
│   ├── distance.py          # Distance metrics (e.g., cosine)
│   ├── loss.py              # Contrastive loss
│   ├── model.py             # Siamese architecture with CLIP encoder
│   ├── test.py              # Evaluation and metrics
│   └── train.py             # Training loop
│
├── main.py                  # Entry point (configure train/test/eval mode)
├── wandb/                   # Weights & Biases logs (optional)
```

---

## 🚀 Training

To start training, edit `main.py` and set:

```python
training = True
```

This will:

* Load the training, validation, and test sets
* Start training over paired samples
* Save:

  * model weights per epoch (`all_weights`)
  * training checkpoints with optimizer state (`checkpoints`)
  * training/validation losses in CSV format (`metrics`)

---

## 🧪 Testing

To evaluate the model, set in `main.py`:

```python
training = False
```

This will:

* Load the model from a selected epoch
* Evaluate it on the test set
* For complete performance over all epochs, use a separate script (`evaluate_all.py`) to generate a CSV with:

  * Epoch number
  * Accuracy metrics (e.g., Top-1/Top-5/Top-10)

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Dependencies include:

* Python 3.8+
* PyTorch
* `transformers`
* `wandb`
* `numpy`, `pandas`, `scikit-learn`, `torchvision`

---

## 📊 Results

This Siamese ViT-B/32 architecture trained with Contrastive Loss demonstrates robust image retrieval capabilities by learning fine-grained facial similarity.


