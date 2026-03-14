# 002 — The Classification Engine

**The AI Engineering Lab** | Post 2 of the Progressive AIML Series

Binary classification from first principles: Sigmoid, Cross-Entropy, Logistic Regression, Class Imbalance, and the full metrics suite — applied to two completely different domains.

---

## Overview

This project builds the complete binary classification pipeline from scratch. Starting with the mathematical foundations (the sigmoid function and cross-entropy loss), it progresses through sklearn's LogisticRegression with L1 and L2 regularization, class imbalance handling with SMOTE, threshold optimization, and probability calibration.

The same pipeline is applied to two independent datasets, demonstrating that the algorithm generalizes across domains.

---

## Datasets

### Dataset 1: Ancient Manuscript Authenticity

| Property | Value |
|:---|:---|
| **Domain** | Archaeometry / Document Forensics |
| **Task** | Predict whether a manuscript is authentic (1) or forged (0) |
| **Rows** | 3,000 |
| **Class Balance** | 20% authentic, 80% forged (imbalanced) |
| **Features** | Ink iron ratio, parchment density, carbon-14 ratio, scribal pressure variance, pigment layer count, UV fluorescence index, linguistic anachronism score, vellum thickness |
| **Why Novel** | Combines archaeometric measurements with ML classification in a domain never used for teaching logistic regression |

### Dataset 2: Silicon Timing Test Pass/Fail

| Property | Value |
|:---|:---|
| **Domain** | Post-Silicon Validation |
| **Task** | Predict whether a chip passes (1) or fails (0) a critical timing test |
| **Rows** | 3,000 |
| **Class Balance** | 71% pass, 29% fail |
| **Features** | VDD core voltage, junction temperature, process corner (slow/typical/fast), leakage current, ring oscillator frequency, IR drop, metal resistance |
| **Engineering Value** | Enables test prioritization by predicting high-confidence PASS outcomes, reducing characterization time by 20-40% |

---

## Key Concepts Covered

| Concept | Description |
|:---|:---|
| **Sigmoid Function** | Maps log-odds to probabilities; decision boundary at σ(z) = 0.5 |
| **Cross-Entropy Loss** | Convex loss function; penalizes confident wrong predictions more than MSE |
| **Gradient Descent** | Iterative parameter optimization via ∂L/∂w = (1/n)·Xᵀ·(ŷ - y) |
| **L1 Regularization** | Lasso-like penalty; can zero out irrelevant features |
| **L2 Regularization** | Ridge-like penalty; shrinks all coefficients; C = 1/λ in sklearn |
| **Stratified Split** | Preserves class ratio in train/test; critical for imbalanced data |
| **SMOTE** | Synthetic Minority Oversampling; generates interpolated minority samples |
| **Threshold Tuning** | Default 0.5 is rarely optimal; tune based on FP/FN cost ratio |
| **Cost-Sensitive Learning** | In silicon validation, FN (missed fail) is 10x more costly than FP |
| **AUC-ROC** | Discrimination ability across all thresholds |
| **AUC-PR** | More informative than ROC for imbalanced datasets |
| **MCC** | Matthews Correlation Coefficient; most balanced metric for binary classification |
| **Platt Scaling** | Probability calibration via logistic regression on raw model scores |

---

## Repository Structure

```
002_classification_engine/
├── data/
│   ├── manuscript_authenticity_data.csv     # 3,000-row archaeometry dataset
│   └── silicon_timing_test_data.csv         # 3,000-row post-silicon dataset
├── notebooks/
│   ├── 01_logistic_regression_manuscript.ipynb   # Full pipeline on manuscript data
│   └── 02_logistic_regression_silicon.ipynb      # Full pipeline on silicon data
├── src/
│   ├── data_generator.py                    # Reproducible synthetic data generation
│   └── generate_visuals.py                  # Standalone publication-quality figures
├── assets/
│   ├── fig1_sigmoid_surface_3d.png          # 3D sigmoid probability surface
│   ├── fig2_decision_boundary.png           # Decision boundary in VDD-Temperature space
│   ├── fig3_roc_pr_curves.png               # ROC and Precision-Recall curves
│   ├── fig4_metrics_dashboard.png           # Full metrics dashboard
│   └── fig5_smote_visualization.png         # SMOTE before/after comparison
├── PRD.md                                   # Product Requirements Document
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Tech Stack

| Tool | Version | Purpose |
|:---|:---|:---|
| Python | 3.11 | Core language |
| pandas | 2.x | Data manipulation |
| numpy | 1.x | Numerical operations |
| scikit-learn | 1.x | LogisticRegression, metrics, preprocessing |
| imbalanced-learn | 0.12 | SMOTE oversampling |
| matplotlib | 3.x | All visualizations |
| seaborn | 0.13 | Heatmaps and statistical plots |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/AIML-Engineering-Lab/002_classification_engine.git
cd 002_classification_engine

# Install dependencies
pip install -r requirements.txt

# Generate datasets
python src/data_generator.py

# Generate all visualizations
python src/generate_visuals.py

# Open notebooks
jupyter notebook notebooks/
```

---

## Visualizations

### Figure 1: 3D Sigmoid Probability Surface
A three-dimensional surface showing how the sigmoid function maps any combination of two features to a probability between 0 and 1. The black contour line marks the decision boundary where P(class=1) = 0.5.

### Figure 2: Decision Boundary in VDD-Temperature Space
The learned linear decision boundary projected onto the two most physically meaningful features: supply voltage and junction temperature. Probability contours show model confidence across the feature space.

### Figure 3: ROC and Precision-Recall Curves
Side-by-side comparison of all four model variants. The PR curve is more informative than ROC for the imbalanced silicon timing dataset.

### Figure 4: Metrics Dashboard
Confusion matrix, all six classification metrics at the optimal threshold, and the precision-recall-F1 tradeoff curve in a single figure.

### Figure 5: SMOTE Visualization
Before-and-after comparison showing how SMOTE generates synthetic minority class samples by interpolating between existing samples in feature space.

---

## Series Context

This is Post 2 of the Progressive AIML Series by The AI Engineering Lab. The series builds from foundational ML algorithms to deep learning, reinforcement learning, generative AI, and agentic AI systems, with every concept demonstrated on both a novel general dataset and a post-silicon validation dataset.

| Post | Topic |
|:---|:---|
| 001 | Linear Regression Engine (OLS, Ridge, Lasso, Elastic Net) |
| **002** | **The Classification Engine (Logistic Regression, Metrics, SMOTE)** |
| 003 | Tree-Based Learning (Decision Trees, Random Forests) |
| 004 | The Boosting Revolution (XGBoost, CatBoost, LightGBM) |
| 005 | Unsupervised Discovery (K-Means, DBSCAN, PCA) |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
