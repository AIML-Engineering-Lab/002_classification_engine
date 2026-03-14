"""
Project 002: The Classification Engine
Standalone visualization generator for all publication-quality figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

ASSETS = 'assets'
COLORS = {'pass': '#4CAF50', 'fail': '#F44336', 'primary': '#2196F3', 'secondary': '#FF9800', 'purple': '#9C27B0'}

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: 3D Sigmoid Surface
# ─────────────────────────────────────────────────────────────────────────────
def fig1_sigmoid_surface():
    fig = plt.figure(figsize=(14, 6))

    # 3D sigmoid surface
    ax1 = fig.add_subplot(121, projection='3d')
    x1 = np.linspace(-4, 4, 80)
    x2 = np.linspace(-4, 4, 80)
    X1, X2 = np.meshgrid(x1, x2)
    W1, W2 = 1.5, 1.0
    Z_logit = W1 * X1 + W2 * X2
    Z_prob  = sigmoid(Z_logit)

    surf = ax1.plot_surface(X1, X2, Z_prob, cmap='RdYlGn', alpha=0.85, linewidth=0)
    ax1.contour(X1, X2, Z_prob, levels=[0.5], zdir='z', offset=0.5, colors='black', linewidths=2)
    ax1.set_xlabel('Feature 1 (e.g., VDD)', fontsize=9, labelpad=8)
    ax1.set_ylabel('Feature 2 (e.g., Temperature)', fontsize=9, labelpad=8)
    ax1.set_zlabel('P(PASS)', fontsize=9, labelpad=8)
    ax1.set_title('3D Sigmoid Probability Surface\nBlack contour = decision boundary (P=0.5)', fontsize=10, fontweight='bold')
    ax1.view_init(elev=25, azim=-55)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='P(PASS)')

    # 2D sigmoid slices
    ax2 = fig.add_subplot(122)
    z = np.linspace(-6, 6, 300)
    for w, label, color in [(0.5, 'Weak signal (w=0.5)', COLORS['secondary']),
                             (1.0, 'Medium signal (w=1.0)', COLORS['primary']),
                             (2.0, 'Strong signal (w=2.0)', COLORS['pass']),
                             (3.5, 'Very strong (w=3.5)', COLORS['fail'])]:
        ax2.plot(z, sigmoid(w * z), linewidth=2.5, label=label)
    ax2.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label='Decision boundary')
    ax2.axvline(0.0, color='gray',  linestyle=':',  linewidth=1.0)
    ax2.set_xlabel('Log-odds (z)', fontsize=11)
    ax2.set_ylabel('P(class = 1)', fontsize=11)
    ax2.set_title('Sigmoid Curves for Different Weight Magnitudes\nHigher weights = steeper boundary', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('The Sigmoid Function: From Linear Scores to Probabilities', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{ASSETS}/fig1_sigmoid_surface_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {ASSETS}/fig1_sigmoid_surface_3d.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Decision Boundary in VDD-Temperature Space
# ─────────────────────────────────────────────────────────────────────────────
def fig2_decision_boundary():
    df = pd.read_csv('data/silicon_timing_test_data.csv')
    X = df[['vdd_core', 'junction_temp']].values
    y = df['timing_pass'].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_sm, y_sm)

    x_min, x_max = X_s[:, 0].min() - 0.3, X_s[:, 0].max() + 0.3
    y_min, y_max = X_s[:, 1].min() - 0.3, X_s[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Probability contour map
    cf = axes[0].contourf(xx, yy, Z, levels=25, cmap='RdYlGn', alpha=0.7)
    plt.colorbar(cf, ax=axes[0], label='P(PASS)')
    axes[0].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2.5)
    axes[0].contour(xx, yy, Z, levels=[0.3, 0.7], colors='gray', linewidths=1.0, linestyles='--')
    for label, color, name in [(1, COLORS['pass'], 'PASS'), (0, COLORS['fail'], 'FAIL')]:
        mask = y_test == label
        axes[0].scatter(X_test[mask, 0], X_test[mask, 1], alpha=0.4, s=12, color=color, label=name, edgecolors='white', linewidths=0.3)
    axes[0].set_xlabel('VDD Core (standardized)', fontsize=11)
    axes[0].set_ylabel('Junction Temperature (standardized)', fontsize=11)
    axes[0].set_title('Decision Boundary in VDD-Temperature Space\nSolid line = P=0.5 | Dashed = P=0.3 and P=0.7', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.2)

    # Probability distribution histogram
    y_prob = model.predict_proba(X_test)[:, 1]
    bins = np.linspace(0, 1, 30)
    axes[1].hist(y_prob[y_test == 1], bins=bins, alpha=0.6, color=COLORS['pass'], label='True PASS', density=True)
    axes[1].hist(y_prob[y_test == 0], bins=bins, alpha=0.6, color=COLORS['fail'], label='True FAIL', density=True)
    axes[1].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Default threshold = 0.5')
    axes[1].set_xlabel('Predicted P(PASS)', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Predicted Probability Distribution\nby True Class', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Logistic Regression: Decision Boundary and Probability Distributions', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{ASSETS}/fig2_decision_boundary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {ASSETS}/fig2_decision_boundary.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: ROC and PR Curves Comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig3_roc_pr_curves():
    df = pd.read_csv('data/silicon_timing_test_data.csv')
    FEATURES = ['vdd_core', 'junction_temp', 'process_corner', 'leakage_current',
                'ring_osc_freq', 'ir_drop_mv', 'metal_resistance']
    X = df[FEATURES].values
    y = df['timing_pass'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train_s, y_train)

    from sklearn.metrics import precision_recall_curve, average_precision_score
    models = {
        'Baseline (L2, C=1)': LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42).fit(X_train_s, y_train),
        'Balanced Weights': LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42).fit(X_train_s, y_train),
        'L1 + SMOTE': LogisticRegression(C=0.5, penalty='l1', solver='liblinear', max_iter=1000, random_state=42).fit(X_sm, y_sm),
        'L2 + SMOTE': LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42).fit(X_sm, y_sm),
    }
    model_colors = [COLORS['fail'], COLORS['secondary'], COLORS['purple'], COLORS['primary']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for (name, model), color in zip(models.items(), model_colors):
        y_prob = model.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        axes[0].plot(fpr, tpr, color=color, linewidth=2.5, label=f'{name} (AUC={auc:.3f})')

    axes[0].fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (AUC=0.5)')
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=11)
    axes[0].set_title('ROC Curves — Silicon Timing Test\nHigher AUC = better discrimination', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].grid(True, alpha=0.3)

    for (name, model), color in zip(models.items(), model_colors):
        y_prob = model.predict_proba(X_test_s)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        axes[1].plot(rec, prec, color=color, linewidth=2.5, label=f'{name} (AP={ap:.3f})')

    axes[1].axhline(y_test.mean(), color='gray', linestyle='--', linewidth=1.5, label=f'Baseline (AP={y_test.mean():.3f})')
    axes[1].set_xlabel('Recall', fontsize=11)
    axes[1].set_ylabel('Precision', fontsize=11)
    axes[1].set_title('Precision-Recall Curves — Silicon Timing Test\nMore informative than ROC for imbalanced data', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Model Comparison: ROC and Precision-Recall Curves', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{ASSETS}/fig3_roc_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {ASSETS}/fig3_roc_pr_curves.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Confusion Matrix + Metrics Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def fig4_metrics_dashboard():
    import seaborn as sns
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, matthews_corrcoef)

    df = pd.read_csv('data/silicon_timing_test_data.csv')
    FEATURES = ['vdd_core', 'junction_temp', 'process_corner', 'leakage_current',
                'ring_osc_freq', 'ir_drop_mv', 'metal_resistance']
    X = df[FEATURES].values
    y = df['timing_pass'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train_s, y_train)
    model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42).fit(X_sm, y_sm)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    # Find optimal threshold
    thresholds = np.linspace(0.05, 0.95, 200)
    f1s = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_t = thresholds[np.argmax(f1s)]
    y_pred = (y_prob >= best_t).astype(int)

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Confusion matrix
    ax1 = fig.add_subplot(gs[0])
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    annot = np.array([[f'{cm[i,j]}\n({cm_norm[i,j]:.1%})' for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax1,
                xticklabels=['FAIL', 'PASS'], yticklabels=['FAIL', 'PASS'],
                linewidths=1, linecolor='white', cbar=False)
    ax1.set_title(f'Confusion Matrix\n(threshold = {best_t:.2f})', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=10)
    ax1.set_xlabel('Predicted Label', fontsize=10)

    # Metrics bar chart
    ax2 = fig.add_subplot(gs[1])
    metric_names  = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'MCC']
    metric_values = [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
        (matthews_corrcoef(y_test, y_pred) + 1) / 2,  # normalize MCC to 0-1 for display
    ]
    bar_colors = [COLORS['primary'], COLORS['pass'], COLORS['secondary'], COLORS['fail'], COLORS['purple'], '#795548']
    bars = ax2.barh(metric_names, metric_values, color=bar_colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars, metric_values):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    ax2.set_xlim(0, 1.15)
    ax2.set_title('All Metrics at Optimal Threshold\n(MCC normalized to 0-1 for display)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Score', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x')

    # Threshold analysis
    ax3 = fig.add_subplot(gs[2])
    precisions = [precision_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    recalls    = [recall_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    ax3.plot(thresholds, precisions, color=COLORS['primary'],    linewidth=2.5, label='Precision')
    ax3.plot(thresholds, recalls,    color=COLORS['pass'],       linewidth=2.5, label='Recall')
    ax3.plot(thresholds, f1s,        color=COLORS['secondary'],  linewidth=2.5, label='F1 Score')
    ax3.axvline(best_t, color='red', linestyle='--', linewidth=2, label=f'Optimal = {best_t:.2f}')
    ax3.axvline(0.5,    color='gray', linestyle=':', linewidth=1.5, label='Default = 0.50')
    ax3.set_xlabel('Classification Threshold', fontsize=10)
    ax3.set_ylabel('Score', fontsize=10)
    ax3.set_title('Threshold Analysis\nPrecision-Recall-F1 Tradeoff', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Classification Metrics Dashboard — Silicon Timing Test', fontsize=13, fontweight='bold', y=1.02)
    plt.savefig(f'{ASSETS}/fig4_metrics_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {ASSETS}/fig4_metrics_dashboard.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: SMOTE Visualization
# ─────────────────────────────────────────────────────────────────────────────
def fig5_smote_visualization():
    df = pd.read_csv('data/manuscript_authenticity_data.csv')
    X = df[['ink_iron_ratio', 'carbon_14_ratio']].values
    y = df['is_authentic'].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(X_s, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for label, color, name in [(1, COLORS['pass'], 'Authentic'), (0, COLORS['fail'], 'Forged')]:
        mask = y_train == label
        axes[0].scatter(X_train[mask, 0], X_train[mask, 1], alpha=0.4, s=12, color=color, label=f'{name} ({mask.sum()})')
    axes[0].set_title(f'Before SMOTE\nClass ratio: {y_train.mean():.2f} authentic', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Ink Iron Ratio (standardized)')
    axes[0].set_ylabel('Carbon-14 Ratio (standardized)')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    n_original = len(y_train)
    for label, color, name in [(1, COLORS['pass'], 'Authentic (original)'), (0, COLORS['fail'], 'Forged (original)')]:
        mask = y_sm[:n_original] == label
        axes[1].scatter(X_sm[:n_original][mask, 0], X_sm[:n_original][mask, 1],
                       alpha=0.4, s=12, color=color, label=f'{name} ({mask.sum()})')

    synthetic_mask = y_sm[n_original:] == 1
    axes[1].scatter(X_sm[n_original:][synthetic_mask, 0], X_sm[n_original:][synthetic_mask, 1],
                   alpha=0.7, s=20, color='#81C784', marker='^', label=f'Synthetic Authentic ({synthetic_mask.sum()})')
    axes[1].set_title(f'After SMOTE\nClass ratio: {y_sm.mean():.2f} authentic (balanced)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Ink Iron Ratio (standardized)')
    axes[1].set_ylabel('Carbon-14 Ratio (standardized)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('SMOTE: Synthetic Minority Oversampling Technique\nManuscript Authenticity Dataset', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{ASSETS}/fig5_smote_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {ASSETS}/fig5_smote_visualization.png')


if __name__ == '__main__':
    import os
    os.makedirs(ASSETS, exist_ok=True)
    print('Generating all visualizations for Project 002...')
    fig1_sigmoid_surface()
    fig2_decision_boundary()
    fig3_roc_pr_curves()
    fig4_metrics_dashboard()
    fig5_smote_visualization()
    print('\nAll 5 figures generated successfully.')
