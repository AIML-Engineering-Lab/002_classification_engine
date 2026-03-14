# PRD: Project 002 — The Classification Engine

**Series:** The AI Engineering Lab  
**Post:** 2 of the Progressive AIML Series  
**Builds on:** Project 001 (Linear Regression Engine)

---

## Objective

Build a complete binary and multi-class classification pipeline using Logistic Regression. Cover the sigmoid function, decision boundaries, probability calibration, all classification metrics, and the ROC/PR curve framework. Apply to two independent datasets from different domains.

---

## Datasets

### Dataset A: Ancient Manuscript Authenticity Classification (General)
Predict whether an ancient manuscript is **authentic** or **forged** based on ink chemical composition, parchment fiber density, carbon isotope ratio, scribal pressure variance, pigment layer count, UV fluorescence index, and linguistic anachronism score. Binary classification.

### Dataset B: Silicon Timing Test Pass/Fail (Post-Silicon Validation)
Predict whether a chip **passes** or **fails** a critical timing test based on supply voltage, junction temperature, process corner (fast/typical/slow), leakage current, ring oscillator frequency, IR drop, and metal layer resistance. Binary classification with class imbalance (most chips pass).

---

## Block Diagram

```
Raw Data
   |
   v
[EDA: distributions, class balance, feature-target correlations]
   |
   v
[Preprocessing: StandardScaler + OneHotEncoder + SMOTE for imbalance]
   |
   v
[Logistic Regression from Scratch: Sigmoid + Cross-Entropy + Gradient Descent]
   |
   v
[sklearn LogisticRegression: L1/L2 regularization, C hyperparameter]
   |
   v
[Decision Boundary Visualization: 2D and 3D]
   |
   v
[Threshold Analysis: default 0.5 vs optimal threshold]
   |
   v
[Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, MCC]
   |
   v
[Calibration: Platt Scaling, reliability diagram]
   |
   v
[Multi-class extension: One-vs-Rest, Softmax]
   |
   v
[Final Evaluation on Test Set]
```

---

## Tech Stack

| Tool | Purpose |
|:---|:---|
| Python 3.11 | Core language |
| NumPy | Sigmoid, cross-entropy, gradient descent from scratch |
| Pandas | Data manipulation |
| scikit-learn | LogisticRegression, metrics, calibration |
| imbalanced-learn | SMOTE for class imbalance |
| Matplotlib / Seaborn | All visualizations |
| Plotly | Interactive 3D decision surface |

---

## Repo Structure

```
002_classification_engine/
├── data/
│   ├── manuscript_authenticity_data.csv
│   └── silicon_timing_test_data.csv
├── notebooks/
│   ├── 01_logistic_regression_manuscript.ipynb
│   └── 02_logistic_regression_silicon.ipynb
├── src/
│   ├── data_generator.py
│   └── generate_visuals.py
├── assets/
│   ├── fig1_sigmoid_surface_3d.png
│   ├── fig2_decision_boundary.png
│   ├── fig3_roc_pr_curves.png
│   ├── fig4_confusion_matrix_dashboard.png
│   └── fig5_threshold_analysis.png
├── PRD.md
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Signature Visualizations

1. **3D Sigmoid Probability Surface:** Two features on X/Y axes, predicted probability on Z axis. The decision boundary is the plane at Z=0.5 cutting through the surface.
2. **2D Decision Boundary:** Scatter plot of two features colored by class, with the logistic regression decision boundary drawn as a line, and probability contours shown.
3. **ROC and PR Curves:** Side by side for both datasets, showing AUC values and the effect of threshold choice.
4. **Confusion Matrix Dashboard:** Normalized and raw confusion matrices with all derived metrics (Precision, Recall, F1, MCC) annotated.
5. **Threshold Analysis:** F1, Precision, and Recall plotted as a function of classification threshold, with the optimal threshold highlighted.

---

## Key Concepts Covered

- Sigmoid function and its relationship to the log-odds (logit)
- Cross-entropy loss function and why it is used instead of MSE for classification
- Gradient descent for logistic regression from scratch
- The C hyperparameter (inverse of regularization strength) in sklearn
- Class imbalance: why accuracy is misleading, SMOTE oversampling
- Threshold tuning: when to optimize for precision vs recall
- Probability calibration: Platt Scaling and reliability diagrams
- Multi-class extension: One-vs-Rest (OvR) strategy

---

## Implementation Prompt (for Copilot/Cursor)

Implement a complete logistic regression classification project in Python. Create two synthetic datasets: (1) ancient manuscript authenticity with 8 features and binary target, (2) silicon timing test pass/fail with 7 features, binary target, and 85/15 class imbalance. Build two Jupyter notebooks — one per dataset — each covering: EDA with class balance analysis, preprocessing pipeline with StandardScaler and OneHotEncoder, logistic regression from scratch (sigmoid + cross-entropy + gradient descent), sklearn LogisticRegression with L1/L2 regularization and C tuning via cross-validation, SMOTE for imbalance handling, decision boundary visualization, full metrics suite (accuracy, precision, recall, F1, AUC-ROC, AUC-PR, MCC), threshold analysis, and probability calibration. Generate 5 publication-quality visualizations including a 3D sigmoid probability surface. Use light blue and green color palette throughout.
