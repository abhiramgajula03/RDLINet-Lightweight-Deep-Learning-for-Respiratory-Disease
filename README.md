# RDLINet — Lightweight Deep Learning for Respiratory Disease Classification

Replication and benchmarking of **RDLINet**, a Residual Depthwise-separable Lightweight Inception Network, for multi-class respiratory disease classification from lung sound audio. Evaluated across three progressively harder dataset configurations using 5-fold cross-validation.

Built as a Semester 6 major project, aggregating multiple public lung sound datasets and reproducing the paper's mel-spectrogram + lightweight-CNN pipeline end to end.

---

## Results (5-Fold Cross-Validation)

| Task | Dataset | Classes | Mean Accuracy | Std Dev |
|---|---|---|---|---|
| **Task 2** | D1 only (ICBHI) | 6 | **99.76%** | 0.09% |
| **Task 3** | D1 + D2 combined | 7 | **98.06%** | 0.21% |
| **Task 1** | D1 + D2 + D3 combined | 7 | **97.74%** | 0.23% |

Task 2's per-fold accuracies ranged from 99.61% to 99.89%, consistently exceeding the paper's reported target of 99.6% for the equivalent 6-class D1 task.

<details>
<summary>Per-fold breakdown</summary>

**Task 1 — D1+D2+D3 Combined (7 classes)**
Fold 1: 97.35% · Fold 2: 97.97% · Fold 3: 97.92% · Fold 4: 97.86% · Fold 5: 97.59%

**Task 2 — D1 only (6 classes)**
Fold 1: 99.74% · Fold 2: 99.89% · Fold 3: 99.82% · Fold 4: 99.61% · Fold 5: 99.72%

**Task 3 — D1+D2 Combined (7 classes)**
Fold 1: 97.86% · Fold 2: 97.82% · Fold 3: 98.02% · Fold 4: 98.23% · Fold 5: 98.38%

</details>

---

## What This Project Does

1. **Audio → Mel-Spectrogram pipeline** — Resamples raw lung sound recordings to 4kHz, splits into 5-second snippets (50% overlap), removes baseline wander via DFT (zeroing 0–1Hz components), normalizes amplitude to [-1, 1], then extracts mel spectrograms (64 filters, 1024-point FFT, 512 hop, Hamming window), converted to log-scale 64×38 images.
2. **10-technique augmentation pipeline** — Balances class distribution (including time-stretching at multiple factors) to correct for severe class imbalance across disease categories.
3. **RDLINet architecture** — A residual, depthwise-separable Inception-style CNN: each inception block fuses four parallel branches (1×1 conv, depthwise 3×3, depthwise 5×5, max-pool + pointwise) before concatenation, with a GLU-style gated dense head for final classification.
4. **Multi-dataset evaluation** — Trains and evaluates the same architecture across three increasingly difficult configurations, from single-dataset (D1 only) to a merged 3-dataset combination with unified class handling for overlapping labels (COPD, Healthy, Pneumonia appear across multiple source datasets).
5. **5-fold stratified cross-validation** — Reports per-fold and aggregate accuracy with standard deviation for each configuration.

---

## Task Breakdown

- **Task 1 — D1 + D2 + D3 Combined (7 classes):** The hardest and most diverse configuration — merges all three source datasets under a unified class set (sorted union), pooling snippets for overlapping classes across datasets.
- **Task 2 — D1 only (6 classes):** Bronchiectasis, Bronchiolitis, COPD, Healthy, Pneumonia, URTI — from the ICBHI 2017 Respiratory Sound Database only.
- **Task 3 — D1 + D2 Combined (7 classes):** Asthma, Bronchiectasis, Bronchiolitis, COPD, Healthy, Pneumonia, URTI — merges two datasets, with COPD/Healthy/Pneumonia pooled across both.

---

## Tech Stack

- **Audio processing:** Librosa, NumPy (DFT-based baseline wander removal), OpenCV (spectrogram image resizing)
- **Deep learning:** TensorFlow / Keras (Conv2D, DepthwiseConv2D, BatchNorm, LeakyReLU, GlobalAveragePooling2D)
- **Evaluation:** scikit-learn (StratifiedKFold)
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Google Colab

---

## Datasets

- **D1 — ICBHI 2017 Respiratory Sound Database** — 6 classes: Bronchiectasis, Bronchiolitis, COPD, Healthy, Pneumonia, URTI
- **D2** — 4 classes: Asthma, COPD, Healthy, Pneumonia
- **D3** — merged into Task 1's unified class set

> Datasets are not included in this repo due to size and licensing — source them from their original public repositories and update the path variables at the top of each notebook.

---

## Repository Structure

```
.
├── task1_D1_D2_D3_combined.ipynb   # 7-class, all three datasets merged
├── task2_D1_only.ipynb              # 6-class, ICBHI only
├── task3_D1_D2_combined.ipynb       # 7-class, two datasets merged
└── README.md
```

---

## Notes on Scope

This repository covers the **training and cross-validation pipeline only**. Deployment benchmarking (e.g. edge-device inference latency, parameter count, FLOPs) is not included here — if that work exists as a separate script or notebook, it should be added and documented separately with its own verified numbers rather than carried over from the original RDLINet paper's reported figures.

---

## Author

**Gajula Abhiram** — B.Tech Information Technology, IIIT Allahabad
[LinkedIn](https://www.linkedin.com/in/gajula-abhiram-0a9a462b1) · [GitHub](https://github.com/abhiramgajula03)
