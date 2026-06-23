import os
import glob
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['font.family'] = 'Arial'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TIPECHA_DIR = os.path.join(SCRIPT_DIR, "dataset_full", "Tipecha")
OUT_1 = os.path.join(SCRIPT_DIR, "presentation_1_pipeline.png")
OUT_2 = os.path.join(SCRIPT_DIR, "presentation_2_imbalance.png")


def pick_random_npy(directory):
    files = glob.glob(os.path.join(directory, "*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {directory}")
    return random.choice(files)


def generate_spectrogram_image():
    npy_path = pick_random_npy(TIPECHA_DIR)
    mel = np.load(npy_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    img = ax.imshow(mel, origin='lower', aspect='auto', cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Amplitude (dB)')

    ax.set_title(
        "תצוגת ספקטרוגרמה - טעם טיפחא (Mel-Spectrogram Representation)",
        fontsize=16, fontweight='bold', pad=14
    )
    ax.set_xlabel("Time Frames", fontsize=13)
    ax.set_ylabel("Mel Bands", fontsize=13)

    fig.tight_layout()
    fig.savefig(OUT_1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[1/2] Saved spectrogram image → {OUT_1}")
    print(f"      (source file: {os.path.basename(npy_path)})")


def generate_imbalance_chart():
    classes = ["Tipecha", "Mercha", "Munach", "Zaqef_Qatan", "Pashta"]
    counts  = [2546,     2068,    1994,    1548,         1233]

    palette = sns.color_palette("muted", n_colors=len(classes))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, counts, color=palette, edgecolor='white', linewidth=0.8, width=0.6)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{count:,}",
            ha='center', va='bottom',
            fontsize=13, fontweight='bold', color='#222222'
        )

    ax.set_title(
        "התפלגות הטעמים - חוסר איזון בנתונים (Class Imbalance)",
        fontsize=16, fontweight='bold', pad=14
    )
    ax.set_xlabel("Taam (Cantillation Mark)", fontsize=13)
    ax.set_ylabel("Sample Count", fontsize=13)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=False, bottom=False)

    fig.tight_layout()
    fig.savefig(OUT_2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[2/2] Saved imbalance chart  → {OUT_2}")


if __name__ == "__main__":
    generate_spectrogram_image()
    generate_imbalance_chart()
    print("\nDone! Both presentation images are ready.")
