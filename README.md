# The Classification Engine

---

## Overview

This project builds the complete binary classification pipeline from mathematical first principles. Starting with the sigmoid function and cross-entropy loss, it progresses through scikit-learn's LogisticRegression with L1 and L2 regularization, class imbalance handling with SMOTE, threshold optimization, and probability calibration via Platt scaling.

The same pipeline is applied to two independent datasets from entirely different domains, demonstrating that the algorithm generalizes across problems.

---

## Datasets

### Dataset A: Ancient Manuscript Authenticity

A synthetic dataset of 3,000 manuscript specimens. The task is to classify each as authentic or forged based on 8 biochemical and physical measurements including ink iron ratio, parchment density, carbon-14 ratio, scribal pressure variance, pigment layer count, UV fluorescence index, linguistic anachronism score, and vellum thickness.

The dataset is heavily imbalanced (20% authentic, 80% forged), requiring SMOTE oversampling and threshold tuning for meaningful minority class detection.

### Dataset B: Silicon Timing Test Pass/Fail

A synthetic dataset of 3,000 silicon characterization measurements. The task is to predict whether a chip passes or fails a critical timing test based on VDD core voltage, junction temperature, process corner, leakage current, ring oscillator frequency, IR drop, and metal resistance.

This represents a real post-silicon validation use case: predicting timing pass/fail to prioritize physical testing and reduce characterization time.

---

## What This Project Covers

| Concept | Description |
|:---|:---|
| **Sigmoid Function** | Maps log-odds to probabilities; decision boundary at σ(z) = 0.5 |
| **Cross-Entropy Loss** | Convex loss function; penalizes confident wrong predictions more than MSE |
| **Gradient Descent** | Iterative parameter optimization via ∂L/∂w = (1/n)·Xᵀ·(ŷ - y) |
| **L1 Regularization** | Lasso-like penalty; can zero out irrelevant features |
| **L2 Regularization** | Ridge-like penalty; shrinks all coefficients; C = 1/λ in sklearn |
| **SMOTE** | Synthetic Minority Oversampling; generates interpolated minority samples |
| **Threshold Tuning** | Default 0.5 is rarely optimal; tune based on FP/FN cost ratio |
| **AUC-ROC / AUC-PR** | Discrimination across all thresholds; PR is more informative for imbalanced data |
| **MCC** | Matthews Correlation Coefficient; most balanced metric for binary classification |
| **Platt Scaling** | Probability calibration via logistic regression on raw model scores |

---

## Repository Structure

```
002_classification_engine/
├── assets/                                        # All notebook-generated visualizations
│   ├── proj1_manuscript_sigmoid_3d.png            # Manuscript: 3D sigmoid probability surface
│   ├── proj1_manuscript_smote.png                 # Manuscript: SMOTE before/after interpolation
│   ├── proj1_manuscript_roc_pr.png                # Manuscript: ROC and Precision-Recall curves
│   ├── proj1_manuscript_metrics_dashboard.png     # Manuscript: confusion matrix + metrics
│   ├── proj1_manuscript_3d_decision_surface.png   # Manuscript: 3D probability decision surface
│   ├── proj1_manuscript_3d_f1_optimization.png    # Manuscript: F1 score optimization surface
│   ├── proj1_manuscript_flowchart.png             # Manuscript: AI-generated pipeline flowchart
│   ├── proj2_silicon_decision_boundary.png        # Silicon: VDD-Temperature decision boundary
│   ├── proj2_silicon_3d_timing.png                # Silicon: 3D feature space scatter
│   ├── proj2_silicon_multi_model_roc.png          # Silicon: multi-model ROC comparison
│   └── proj2_silicon_flowchart.png                # Silicon: AI-generated pipeline flowchart
├── data/
│   ├── manuscript_authenticity_data.csv           # 3,000 manuscript specimens (8 features + target)
│   └── silicon_timing_test_data.csv               # 3,000 silicon samples (7 features + target)
├── deploy/
│   ├── Dockerfile                                 # Container image for FastAPI server
│   ├── docker-compose.yml                         # Single-command deployment
│   ├── nginx.conf                                 # Reverse proxy configuration
│   └── railway.json                               # Railway.app deployment config
├── docs/
│   ├── Classification_Engine_Report.html          # Report source (HTML with embedded images)
│   └── Classification_Engine_Report.pdf           # Final PDF report (both projects)
├── models/
│   ├── logreg_manuscript.pkl                      # Trained LogisticRegression for manuscript (Acc = 0.84)
│   └── logreg_silicon.pkl                         # Trained LogisticRegression for silicon (Acc = 0.80)
├── notebooks/
│   ├── 01_logistic_regression_manuscript.ipynb    # Full pipeline: EDA → SMOTE → L1/L2 → Metrics
│   └── 02_logistic_regression_silicon.ipynb       # Silicon: decision boundary → calibration → ROC
├── src/
│   ├── train.py                                   # Train LogisticRegression for both datasets
│   ├── predict.py                                 # Load model and run inference
│   ├── api.py                                     # FastAPI serving endpoint (POST /predict)
│   └── data_generator.py                          # Synthetic dataset generation
├── tests/
│   └── test_model.py                              # Model validation tests
├── .gitignore
├── LICENSE                                        # MIT License
└── requirements.txt                               # Python dependencies
```

---

## Key Visualizations

### 3D Sigmoid Probability Surface
The sigmoid function maps linear combinations to probabilities in [0, 1]. This 3D surface shows how two input features jointly determine predicted probabilities, with the decision boundary at P = 0.5.

### SMOTE Oversampling
With only 20% authentic manuscripts, the minority class is underrepresented. SMOTE generates synthetic samples by interpolating between minority-class k-nearest neighbors, rebalancing the training set.

### Decision Boundary in VDD-Temperature Space
For the silicon dataset, the linear decision boundary is visualized in the two most physically meaningful features. Higher VDD and lower temperature consistently improve timing margins.

### ROC and Precision-Recall Curves
Four model variants compared side by side. AUC-PR is more informative than AUC-ROC for the imbalanced manuscript dataset.

### Confusion Matrix and Metrics Dashboard
Raw and normalized confusion matrices with accuracy, precision, recall, F1, and MCC at the optimal threshold.

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/AIML-Engineering-Lab/002_classification_engine.git
cd 002_classification_engine

# Install dependencies
pip install -r requirements.txt

# Generate datasets
python3 src/data_generator.py

# Open notebooks
jupyter notebook notebooks/

# Train models and save artifacts (both datasets)
python3 src/train.py

# Run predictions
python3 src/predict.py

# Start API server
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Run tests
python3 tests/test_model.py
```

---

## Tech Stack

| Tool | Version | Purpose |
|:---|:---|:---|
| Python | 3.11+ | Core language |
| NumPy | 1.24+ | Linear algebra, sigmoid implementation |
| Pandas | 2.0+ | Data manipulation |
| scikit-learn | 1.3+ | LogisticRegression, metrics, calibration, CV |
| imbalanced-learn | 0.11+ | SMOTE oversampling |
| Matplotlib | 3.7+ | All visualizations including 3D |
| Seaborn | 0.12+ | Statistical plots, heatmaps |
| FastAPI | 0.100+ | REST API serving |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
