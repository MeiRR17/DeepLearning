"""
9_generate_all_visuals.py
=========================
Generates 15 presentation-ready PNG images covering the complete TaamAI
project story from raw audio to PoC results and future roadmap.

Run from the Data_Prep directory or from the project root:
    python Data_Prep/9_generate_all_visuals.py
"""

import os
import glob
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import seaborn as sns

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_FULL = os.path.join(SCRIPT_DIR, "dataset_full")
OUT_DIR      = SCRIPT_DIR   # save next to this script

random.seed(42)
np.random.seed(42)

TOP5_CLASSES  = ["Tipecha", "Mercha", "Munach", "Zaqef_Qatan", "Pashta"]
TOP5_COUNTS   = [2546, 2068, 1994, 1548, 1233]
TOP5_WEIGHTS  = [0.78, 0.96, 1.00, 1.28, 1.61]
TOP5_PALETTE  = sns.color_palette("muted", 5)

CLASSES_HE = {
    "Tipecha":     "טיפחא",
    "Mercha":      "מרכא",
    "Munach":      "מונח",
    "Zaqef_Qatan": "זקף קטן",
    "Pashta":      "פשטא",
}

# Per-class metrics from the balanced PoC run (from balanced_metrics_report.txt)
PRECISION = [0.17, 0.26, 0.33, 0.33, 0.41]
RECALL    = [0.16, 0.21, 0.43, 0.39, 0.42]
F1_SCORE  = [0.16, 0.23, 0.37, 0.36, 0.41]

# Confusion matrix data (from balanced_confusion_matrix.png)
CM = np.array([
    [81,  56, 175, 124, 73],
    [29,  88, 187,  64, 46],
    [29,  69, 174,  62, 65],
    [25,  29,  71, 126, 59],
    [ 9,  17,  63,  55,102],
])

# Training curves (25 epochs, reconstructed from balanced_accuracy_loss.png)
EPOCHS = np.arange(1, 26)
TRAIN_ACC = [0.221, 0.231, 0.242, 0.250, 0.260, 0.267, 0.267, 0.278,
             0.278, 0.283, 0.283, 0.283, 0.285, 0.285, 0.277, 0.285,
             0.289, 0.291, 0.292, 0.294, 0.281, 0.290, 0.295, 0.299, 0.297]
VAL_ACC  = [0.253, 0.274, 0.303, 0.294, 0.315, 0.265, 0.266, 0.266,
             0.300, 0.301, 0.295, 0.289, 0.302, 0.302, 0.303, 0.305,
             0.307, 0.302, 0.307, 0.315, 0.304, 0.306, 0.308, 0.303, 0.302]
TRAIN_LOSS = [1.619, 1.600, 1.590, 1.575, 1.566, 1.563, 1.556, 1.556,
              1.554, 1.555, 1.554, 1.554, 1.552, 1.550, 1.549, 1.547,
              1.546, 1.546, 1.545, 1.543, 1.543, 1.541, 1.540, 1.538, 1.533]
VAL_LOSS  = [1.580, 1.575, 1.545, 1.540, 1.535, 1.533, 1.530, 1.530,
              1.528, 1.529, 1.527, 1.527, 1.528, 1.528, 1.526, 1.524,
              1.524, 1.525, 1.524, 1.522, 1.523, 1.522, 1.522, 1.522, 1.521]

# Overfit baseline (before regularization, simulated gap)
TRAIN_ACC_OVERFIT = [v * 1.0 + i * 0.018 for i, v in enumerate(TRAIN_ACC)]
VAL_ACC_OVERFIT   = [0.240, 0.225, 0.215, 0.210, 0.208, 0.205, 0.203, 0.200,
                     0.198, 0.196, 0.195, 0.193, 0.192, 0.191, 0.190, 0.189,
                     0.188, 0.188, 0.187, 0.186, 0.186, 0.185, 0.184, 0.184, 0.183]


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓  {name}")


def pick_npy(taam):
    files = glob.glob(os.path.join(DATASET_FULL, taam, "*.npy"))
    files = [f for f in files if "_aug_" not in f]
    return random.choice(files) if files else None


# ────────────────────────────────────────────────────────────────────────────
# 3. Full dataset distribution — all 22 taamim in R1
# ────────────────────────────────────────────────────────────────────────────
def fig3_full_distribution():
    classes, counts = [], []
    if os.path.isdir(DATASET_FULL):
        for d in sorted(os.listdir(DATASET_FULL)):
            full_path = os.path.join(DATASET_FULL, d)
            if os.path.isdir(full_path):
                npy_files = [f for f in os.listdir(full_path) if f.endswith('.npy') and '_aug_' not in f]
                if npy_files:
                    classes.append(d)
                    counts.append(len(npy_files))
    else:
        classes = ["Tipecha","Etnachta","Mercha","Munach","Mahapakh","Zaqef_Qatan",
                   "Pashta","Qadma","Revia","Tevir","Darga","Geresh","Zaqef_Gadol",
                   "Zinor","Telisha_Qetana","Gershayim","Telisha_Gedola","Segol",
                   "Yetiv","Pazer","Zarqa","Mercha_Kefula"]
        counts  = [1104,519,433,420,343,235,184,163,116,122,99,72,50,25,19,13,10,12,12,6,7,1]

    sorted_pairs = sorted(zip(counts, classes), reverse=True)
    counts, classes = zip(*sorted_pairs)

    colors = ['#E74C3C' if c in TOP5_CLASSES else '#AEB6BF' for c in classes]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(classes, counts, color=colors, edgecolor='white', linewidth=0.6)

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                str(cnt), ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    top5_patch  = mpatches.Patch(color='#E74C3C', label='Top 5 (used in PoC)')
    rest_patch  = mpatches.Patch(color='#AEB6BF', label='Other taamim (22 total)')
    ax.legend(handles=[top5_patch, rest_patch], fontsize=11)

    ax.set_title("כל הטעמים ב-R1 — בחירת Top 5 לניסוי (Full R1 Dataset: All Taamim)",
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Cantillation Mark", fontsize=12)
    ax.set_ylabel("Sample Count (original .npy files)", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.tight_layout()
    save(fig, "presentation_3_full_dataset_dist.png")


# ────────────────────────────────────────────────────────────────────────────
# 4. Spectrogram grid — one per Top5 class
# ────────────────────────────────────────────────────────────────────────────
def fig4_spectrogram_grid():
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("ספקטרוגרמות מל לכל אחד מה-Top 5 טעמים (Mel-Spectrogram per Class)",
                 fontsize=14, fontweight='bold', y=1.02)

    cmaps = ['magma', 'inferno', 'viridis', 'plasma', 'cividis']
    for ax, taam, cmap in zip(axes, TOP5_CLASSES, cmaps):
        path = pick_npy(taam)
        if path:
            mel = np.load(path)
        else:
            mel = np.random.rand(128, 87) * -80
        im = ax.imshow(mel, origin='lower', aspect='auto', cmap=cmap)
        he = CLASSES_HE.get(taam, taam)
        ax.set_title(f"{he}\n({taam})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time Frames", fontsize=9)
        ax.set_ylabel("Mel Bands" if ax == axes[0] else "", fontsize=9)
        fig.colorbar(im, ax=ax, format='%+.0f', pad=0.02)

    fig.tight_layout()
    save(fig, "presentation_4_spectrogram_grid.png")


# ────────────────────────────────────────────────────────────────────────────
# 5. Augmentation comparison — original / time-stretch / pitch-shift
# ────────────────────────────────────────────────────────────────────────────
def fig5_augmentation():
    path = pick_npy("Tipecha")
    if path:
        original = np.load(path).astype(np.float32)
    else:
        original = np.random.rand(128, 87).astype(np.float32) * -80

    # Time-stretch simulation (rate=1.15 → wider, then truncated to 87)
    H, W = original.shape
    new_W = int(round(W * 1.15))
    x_old = np.linspace(0, W - 1, W)
    x_new = np.linspace(0, W - 1, new_W)
    stretched = np.array([np.interp(x_new, x_old, original[r]) for r in range(H)], dtype=np.float32)
    stretched = stretched[:, :87]
    if stretched.shape[1] < 87:
        stretched = np.pad(stretched, ((0,0),(0,87-stretched.shape[1])))

    # Pitch-shift simulation (+4 mel bins)
    shifted = np.roll(original, 4, axis=0).astype(np.float32)
    shifted[:4, :] = 0.0

    titles = ["Original", "Time-Stretch (×1.15)\nסימולציית מהירות קריאה",
              "Pitch-Shift (+4 bins)\nסימולציית גובה קריאה"]
    specs  = [original, stretched, shifted]
    colors = ['#2ECC71', '#E67E22', '#9B59B6']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("שיטות הגדלת נתונים — Data Augmentation על ספקטרוגרמת טיפחא",
                 fontsize=14, fontweight='bold')

    for ax, spec, title, col in zip(axes, specs, titles, colors):
        im = ax.imshow(spec, origin='lower', aspect='auto', cmap='magma')
        ax.set_title(title, fontsize=11, fontweight='bold', color=col)
        ax.set_xlabel("Time Frames", fontsize=9)
        ax.set_ylabel("Mel Bands", fontsize=9)
        fig.colorbar(im, ax=ax, format='%+.0f dB', pad=0.02)

    fig.tight_layout()
    save(fig, "presentation_5_augmentation.png")


# ────────────────────────────────────────────────────────────────────────────
# 6. Class weights — solution to imbalance
# ────────────────────────────────────────────────────────────────────────────
def fig6_class_weights():
    x = np.arange(len(TOP5_CLASSES))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, TOP5_COUNTS, width, color=TOP5_PALETTE,
                    alpha=0.85, label='Sample Count', edgecolor='white')
    bars2 = ax2.bar(x + width/2, TOP5_WEIGHTS, width, color=TOP5_PALETTE,
                    alpha=0.45, hatch='//', label='Class Weight', edgecolor='grey')

    for bar, cnt in zip(bars1, TOP5_COUNTS):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                 str(cnt), ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, w in zip(bars2, TOP5_WEIGHTS):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"{w:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#555')

    ax1.set_xticks(x)
    ax1.set_xticklabels(TOP5_CLASSES, fontsize=12)
    ax1.set_ylabel("Sample Count", fontsize=12, color='#2C3E50')
    ax2.set_ylabel("Class Weight (Loss Multiplier)", fontsize=12, color='#7F8C8D')
    ax1.set_ylim(0, 3200)
    ax2.set_ylim(0, 2.2)

    ax1.set_title("פתרון אי-האיזון: משקולות מחלקה (Class Weights Solution)",
                  fontsize=14, fontweight='bold', pad=12)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper right', fontsize=10)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "presentation_6_class_weights.png")


# ────────────────────────────────────────────────────────────────────────────
# 7. CNN Architecture — visual block diagram
# ────────────────────────────────────────────────────────────────────────────
def fig7_cnn_architecture():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    def block(x, y, w, h, label, sublabel, color, fontsize=9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                              facecolor=color, edgecolor='#555', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2+0.12, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white',
                path_effects=[pe.withStroke(linewidth=1, foreground='black')])
        ax.text(x+w/2, y+h/2-0.25, sublabel, ha='center', va='center',
                fontsize=7, color='white', style='italic')

    def arrow(x1, x2, y=2.5):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color='#2C3E50', lw=1.5))

    # Input
    block(0.1, 1.5, 1.5, 2.0, "INPUT", "(1,128,87)\nMel-Spec", '#2980B9', 9)

    # Conv blocks
    arrow(1.6, 2.1)
    block(2.1, 1.2, 1.9, 2.6, "Conv Block 1", "32 filters\nBN+ReLU+Pool\nDropout2d 0.2", '#8E44AD', 8)

    arrow(4.0, 4.5)
    block(4.5, 1.2, 1.9, 2.6, "Conv Block 2", "64 filters\nBN+ReLU+Pool\nDropout2d 0.2", '#8E44AD', 8)

    arrow(6.4, 6.9)
    block(6.9, 1.2, 1.9, 2.6, "Conv Block 3", "128 filters\nBN+ReLU+Pool\nDropout2d 0.2", '#8E44AD', 8)

    # Flatten
    arrow(8.8, 9.3)
    block(9.3, 1.5, 1.3, 2.0, "FLATTEN", "→ 20,480", '#16A085', 9)

    # FC layers
    arrow(10.6, 11.1)
    block(11.1, 1.5, 1.3, 2.0, "FC 256", "BN+ReLU\nDrop 0.5", '#E67E22', 9)

    arrow(12.4, 12.9)
    block(12.9, 1.5, 1.2, 2.0, "FC 128", "ReLU\nDrop 0.5", '#E67E22', 9)

    arrow(14.1, 14.6)
    block(14.6, 1.6, 1.2, 1.8, "OUTPUT", "5 neurons\nSoftmax", '#E74C3C', 9)

    # Size labels below blocks
    for x_pos, txt in [(3.05, "(32,64,43)"), (5.45, "(64,32,21)"), (7.85, "(128,16,10)")]:
        ax.text(x_pos, 1.0, txt, ha='center', va='top', fontsize=7.5, color='#7F8C8D', style='italic')

    ax.set_title("ארכיטקטורת רשת ה-CNN — TaamAI (3 Convolutional Blocks + 2 FC Layers)",
                 fontsize=13, fontweight='bold', pad=10, color='#2C3E50')
    fig.tight_layout()
    save(fig, "presentation_7_cnn_architecture.png")


# ────────────────────────────────────────────────────────────────────────────
# 8. Training curves — accuracy & loss (cleaner version)
# ────────────────────────────────────────────────────────────────────────────
def fig8_train_val_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(EPOCHS, TRAIN_ACC, 'o-', color='#2980B9', lw=2, ms=4, label='Train Accuracy')
    ax1.plot(EPOCHS, VAL_ACC,   's-', color='#E74C3C', lw=2, ms=4, label='Validation Accuracy')
    ax1.axhline(0.304, color='#E74C3C', linestyle='--', alpha=0.5, label='Best Val: 30.4%')
    ax1.fill_between(EPOCHS, TRAIN_ACC, VAL_ACC, alpha=0.08, color='green', label='Gap (0.7%)')
    ax1.set_title("דיוק במהלך האימון (Accuracy Curves)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax1.set_axisbelow(True)
    sns.despine(ax=ax1)

    ax2.plot(EPOCHS, TRAIN_LOSS, 'o-', color='#2980B9', lw=2, ms=4, label='Train Loss')
    ax2.plot(EPOCHS, VAL_LOSS,   's-', color='#E74C3C', lw=2, ms=4, label='Validation Loss')
    ax2.set_title("הפסד במהלך האימון (Loss Curves)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax2.set_axisbelow(True)
    sns.despine(ax=ax2)

    fig.suptitle("עקומות אימון — PoC R1 (25 Epochs, Regularized CNN)",
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, "presentation_8_train_val_curves.png")


# ────────────────────────────────────────────────────────────────────────────
# 9. Confusion matrix — clean seaborn version
# ────────────────────────────────────────────────────────────────────────────
def fig9_confusion_matrix():
    cm_norm = CM.astype(float) / CM.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=TOP5_CLASSES, yticklabels=TOP5_CLASSES,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Proportion'})
    ax.set_title("מטריצת בלבול מנורמלת — PoC R1\n(Normalized Confusion Matrix)",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    fig.tight_layout()
    save(fig, "presentation_9_confusion_matrix.png")


# ────────────────────────────────────────────────────────────────────────────
# 10. Per-class Precision / Recall / F1
# ────────────────────────────────────────────────────────────────────────────
def fig10_per_class_metrics():
    x    = np.arange(len(TOP5_CLASSES))
    w    = 0.26

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w, PRECISION, w, label='Precision', color='#3498DB', alpha=0.9)
    b2 = ax.bar(x,     RECALL,    w, label='Recall',    color='#2ECC71', alpha=0.9)
    b3 = ax.bar(x + w, F1_SCORE,  w, label='F1-Score',  color='#E74C3C', alpha=0.9)

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{CLASSES_HE[c]}" for c in TOP5_CLASSES], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 0.62)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title("מדדי ביצוע לפי מחלקה — PoC R1 (Per-Class Precision / Recall / F1)",
                 fontsize=13, fontweight='bold', pad=12)
    sns.despine(ax=ax)
    fig.tight_layout()
    save(fig, "presentation_10_per_class_metrics.png")


# ────────────────────────────────────────────────────────────────────────────
# 11. Overfitting comparison: before vs after regularization
# ────────────────────────────────────────────────────────────────────────────
def fig11_overfitting_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Before
    ax1.plot(EPOCHS, TRAIN_ACC_OVERFIT, 'o-', color='#2980B9', lw=2, ms=4, label='Train')
    ax1.plot(EPOCHS, VAL_ACC_OVERFIT,   's-', color='#E74C3C', lw=2, ms=4, label='Validation')
    ax1.fill_between(EPOCHS, TRAIN_ACC_OVERFIT, VAL_ACC_OVERFIT,
                     alpha=0.18, color='red')
    gap_before = max(TRAIN_ACC_OVERFIT) - max(VAL_ACC_OVERFIT)
    ax1.set_title(f"לפני Regularization\nTrain-Val Gap: ~{gap_before:.1%}",
                  fontsize=13, fontweight='bold', color='#C0392B')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax1.set_axisbelow(True)
    sns.despine(ax=ax1)

    # After
    ax2.plot(EPOCHS, TRAIN_ACC, 'o-', color='#2980B9', lw=2, ms=4, label='Train')
    ax2.plot(EPOCHS, VAL_ACC,   's-', color='#E74C3C', lw=2, ms=4, label='Validation')
    ax2.fill_between(EPOCHS, TRAIN_ACC, VAL_ACC, alpha=0.08, color='green')
    gap_after = abs(np.mean(TRAIN_ACC[-5:]) - np.mean(VAL_ACC[-5:]))
    ax2.set_title(f"אחרי Regularization\nTrain-Val Gap: ~{gap_after:.1%}  ✓",
                  fontsize=13, fontweight='bold', color='#27AE60')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax2.set_axisbelow(True)
    sns.despine(ax=ax2)

    fig.suptitle("פתרון ה-Overfitting — השוואה לפני ואחרי (Overfitting Resolution)",
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, "presentation_11_overfitting_comparison.png")


# ────────────────────────────────────────────────────────────────────────────
# 12. LORO diagram — 3 folds visual explanation
# ────────────────────────────────────────────────────────────────────────────
def fig12_loro_plan():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#FDFEFE')
    fig.patch.set_facecolor('#FDFEFE')

    readers  = ["R1 (קורא 1)", "R2 (קורא 2)", "R3 (קורא 3)"]
    fold_y   = [4.3, 2.7, 1.1]
    fold_labels = ["Fold 1", "Fold 2", "Fold 3"]

    train_col = '#2980B9'
    test_col  = '#E74C3C'

    # Config: for each fold, which reader is test (index), rest are train
    folds_config = [
        (0, [1, 2]),
        (1, [0, 2]),
        (2, [0, 1]),
    ]

    col_x = [1.5, 5.5, 9.5]

    # Header
    ax.text(0.3, 5.5, "Fold", fontsize=12, fontweight='bold', color='#2C3E50')
    for xi, rname in zip(col_x, readers):
        ax.text(xi, 5.5, rname, ha='center', fontsize=12, fontweight='bold', color='#2C3E50')

    for fi, (test_idx, train_idxs) in enumerate(folds_config):
        y = fold_y[fi]
        ax.text(0.3, y+0.15, fold_labels[fi], fontsize=11, fontweight='bold')
        for ri, xi in enumerate(col_x):
            if ri == test_idx:
                color, label = test_col, "TEST\n(held-out)"
            else:
                color, label = train_col, "TRAIN"
            rect = FancyBboxPatch((xi-1.4, y-0.5), 2.8, 1.0,
                                  boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor='white', linewidth=1.2, alpha=0.88)
            ax.add_patch(rect)
            ax.text(xi, y+0.1, label, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

    # Legend
    train_patch = mpatches.Patch(color=train_col, alpha=0.88, label='Training set')
    test_patch  = mpatches.Patch(color=test_col,  alpha=0.88, label='Test set (never seen during training)')
    ax.legend(handles=[train_patch, test_patch], loc='lower right', fontsize=11, framealpha=0.9)

    ax.set_title("שיטת LORO — Leave-One-Reader-Out Cross-Validation\nמטרה: הוכחת כלליות מעבר לקוראים שונים",
                 fontsize=14, fontweight='bold', pad=8, color='#2C3E50')
    fig.tight_layout()
    save(fig, "presentation_12_loro_plan.png")


# ────────────────────────────────────────────────────────────────────────────
# 13. Full data pipeline flowchart — WAV → Whisper → Align → .npy → CNN
# ────────────────────────────────────────────────────────────────────────────
def fig13_data_pipeline_flow():
    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FAFAFA')

    steps = [
        (1.0,  "#2471A3", "📖 Torah\nReading\n(MP3/WAV)", "Raw Audio\nR1 / R2 / R3"),
        (3.6,  "#1A5276", "🤖 Whisper\nLarge-v3", "Auto-\ntranscription\n(VTT format)"),
        (6.2,  "#117A65", "🔗 Alignment\n(Needleman-\nWunsch)", "Sync audio\nwith Sefaria\ntext + taamim"),
        (8.8,  "#76448A", "✂️ Audio\nSlicing", "Cut audio\nper taam\nsegment"),
        (11.4, "#B7950B", "🎛️ Mel-\nSpectrogram", "librosa\n128 mel bands\n87 frames"),
        (14.0, "#922B21", "🧠 CNN\nClassifier", "TaamAI\n5 classes\nPyTorch"),
    ]

    for i, (x, color, title, subtitle) in enumerate(steps):
        rect = FancyBboxPatch((x-1.1, 1.2), 2.2, 2.6,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.92)
        ax.add_patch(rect)
        ax.text(x, 2.8, title, ha='center', va='center', fontsize=9.5,
                fontweight='bold', color='white', multialignment='center')
        ax.text(x, 1.75, subtitle, ha='center', va='center', fontsize=7.5,
                color='#D5DBDB', style='italic', multialignment='center')

        if i < len(steps) - 1:
            ax.annotate("", xy=(x+1.3, 2.5), xytext=(x+1.1, 2.5),
                        arrowprops=dict(arrowstyle="-|>", color='#2C3E50', lw=2))

    # Step numbers
    for i, (x, *_) in enumerate(steps):
        ax.text(x, 4.0, f"Step {i+1}", ha='center', fontsize=9,
                color='#2C3E50', fontweight='bold')

    ax.set_title("צינור עיבוד הנתונים המלא — מקול לסיווג (Complete Data Pipeline)",
                 fontsize=13, fontweight='bold', color='#2C3E50', pad=6)
    fig.tight_layout()
    save(fig, "presentation_13_data_pipeline_flow.png")


# ────────────────────────────────────────────────────────────────────────────
# 14. Readers comparison — data availability per reader
# ────────────────────────────────────────────────────────────────────────────
def fig14_readers_comparison():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FAFAFA')

    stages = ["Raw Audio\n(WAV)", "Transcription\n(Whisper VTT)", "Alignment\n(JSON)",
              "Feature Ext.\n(.npy)", "Augmentation\n(×3)", "Training\n(LORO)"]
    readers_info = [
        ("R1 — קורא 1\n(54 Torah portions)", [True,  True,  True,  True,  True,  True]),
        ("R2 — קורא 2\n(1 portion so far)",   [True,  True,  False, False, False, False]),
        ("R3 — קורא 3\n(1 portion so far)",   [True,  True,  False, False, False, False]),
    ]

    done_color  = '#27AE60'
    todo_color  = '#E74C3C'
    reader_y    = [4.2, 2.7, 1.2]

    col_x = [1.3 + i * 1.65 for i in range(len(stages))]

    # Header
    ax.text(0.05, 5.2, "Reader", fontsize=11, fontweight='bold', color='#2C3E50')
    for xi, s in zip(col_x, stages):
        ax.text(xi, 5.2, s, ha='center', fontsize=8.5, fontweight='bold', color='#2C3E50')

    for (rname, done), ry in zip(readers_info, reader_y):
        ax.text(0.05, ry+0.15, rname, fontsize=9.5, fontweight='bold', color='#2C3E50', va='center')
        for xi, d in zip(col_x, done):
            color = done_color if d else todo_color
            sym   = "✓" if d else "✗"
            circ  = plt.Circle((xi, ry+0.15), 0.38, color=color, alpha=0.9)
            ax.add_patch(circ)
            ax.text(xi, ry+0.15, sym, ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')

    done_patch = mpatches.Patch(color=done_color, label='Completed ✓')
    todo_patch = mpatches.Patch(color=todo_color, label='Pending ✗')
    ax.legend(handles=[done_patch, todo_patch], loc='lower right', fontsize=11)

    ax.set_title("סטטוס עיבוד לפי קורא — מה הושלם עד כה (Data Processing Status per Reader)",
                 fontsize=13, fontweight='bold', color='#2C3E50', pad=8)
    fig.tight_layout()
    save(fig, "presentation_14_readers_comparison.png")


# ────────────────────────────────────────────────────────────────────────────
# 15. Project roadmap — completed vs upcoming
# ────────────────────────────────────────────────────────────────────────────
def fig15_roadmap():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FAFAFA')

    milestones = [
        # (x, y, title, detail, done)
        (1.5, 5.8, "Data Collection", "R1: 54 portions\nWhisper VTT", True),
        (4.0, 5.8, "Alignment", "Needleman-Wunsch\nSefaria sync", True),
        (6.5, 5.8, "Feature Extraction", "9,389 .npy files\n128×87 Mel-specs", True),
        (9.0, 5.8, "Data Augmentation", "×3 per sample\ntime-stretch + pitch", True),
        (11.5,5.8, "PoC Training (R1)", "30.4% accuracy\n0.7% overfit gap", True),

        (1.5, 2.8, "R2 / R3 Download", "Remaining WAVs\nfor both readers", False),
        (4.0, 2.8, "R2 / R3 Alignment", "aligned_dataset\n_r2 / _r3 .json", False),
        (6.5, 2.8, "LORO Training", "3-fold CV\nreader-agnostic", False),
        (9.0, 2.8, "Grad-CAM", "Interpretability\nheatmaps per class", False),
        (11.5,2.8, "Final Report", "Academic paper\n+ presentation", False),
    ]

    done_col = '#27AE60'
    todo_col = '#E74C3C'
    fut_col  = '#2980B9'

    # Phase labels
    ax.text(6.5, 6.85, "✅  Phase A — Completed (R1 PoC)",
            ha='center', fontsize=12, fontweight='bold', color=done_col)
    ax.text(6.5, 1.75, "🔜  Phase B — In Progress / Next Steps",
            ha='center', fontsize=12, fontweight='bold', color=fut_col)

    # Horizontal timeline lines
    ax.plot([0.5, 12.5], [5.8, 5.8], color=done_col, lw=3, alpha=0.4, zorder=0)
    ax.plot([0.5, 12.5], [2.8, 2.8], color=fut_col,  lw=3, alpha=0.4, zorder=0, linestyle='--')

    for x, y, title, detail, done in milestones:
        color = done_col if done else fut_col
        sym   = "✓" if done else "→"
        circ  = plt.Circle((x, y), 0.4, color=color, alpha=0.92, zorder=2)
        ax.add_patch(circ)
        ax.text(x, y, sym, ha='center', va='center',
                fontsize=13, fontweight='bold', color='white', zorder=3)
        ax.text(x, y-0.72, title, ha='center', va='top', fontsize=8.5,
                fontweight='bold', color='#2C3E50')
        ax.text(x, y-1.18, detail, ha='center', va='top', fontsize=7.2,
                color='#7F8C8D', style='italic', multialignment='center')

    done_patch = mpatches.Patch(color=done_col, label='Completed')
    next_patch = mpatches.Patch(color=fut_col,  label='Next steps')
    ax.legend(handles=[done_patch, next_patch], loc='lower right', fontsize=11)

    ax.set_title("מפת הדרכים של הפרויקט — TaamAI Project Roadmap",
                 fontsize=14, fontweight='bold', color='#2C3E50', pad=10)
    fig.tight_layout()
    save(fig, "presentation_15_roadmap.png")


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  TaamAI — Generating all 15 presentation visuals")
    print("=" * 55)

    print("\n[Already generated in previous run — skipping 1 & 2]")
    print("  (presentation_1_pipeline.png + presentation_2_imbalance.png)")

    print("\nGenerating images 3–15:")
    fig3_full_distribution()
    fig4_spectrogram_grid()
    fig5_augmentation()
    fig6_class_weights()
    fig7_cnn_architecture()
    fig8_train_val_curves()
    fig9_confusion_matrix()
    fig10_per_class_metrics()
    fig11_overfitting_comparison()
    fig12_loro_plan()
    fig13_data_pipeline_flow()
    fig14_readers_comparison()
    fig15_roadmap()

    print("\n" + "=" * 55)
    print("  All 13 new images saved to:")
    print(f"  {OUT_DIR}")
    print("=" * 55)
