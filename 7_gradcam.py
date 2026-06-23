"""
7_gradcam.py
============
Grad-CAM interpretability visualizations for the TaamAI CNN.

For each class in the model, this script:
  1. Picks one correctly-classified sample from the test/held-out set.
  2. Runs Grad-CAM on the last convolutional layer (conv3).
  3. Overlays the heatmap on the original Mel-Spectrogram.
  4. Saves the figure as:
       grad_cam_<class_name_lower>.png
     e.g.  grad_cam_tipecha.png, grad_cam_mercha.png, ...

What Grad-CAM shows
-------------------
  The heatmap highlights the time-frequency regions of the spectrogram that
  most influenced the model's prediction.  Hot regions (red/yellow) indicate
  where the CNN "focuses" to recognise a particular cantillation mark.

Usage
-----
  # Use the LORO fold-3 model (held-out R3) and R3 test samples
  python 7_gradcam.py

  # Explicit model and dataset
  python 7_gradcam.py \\
      --model loro_fold3_model.pth \\
      --classes loro_classes.json \\
      --dataset Data_Prep/dataset_r3 \\
      --out gradcam_outputs

  # Use the original R1 model with R1 top-5 classes
  python 7_gradcam.py \\
      --model model_r1_top5_balanced_pytorch.pth \\
      --classes classes_top5.json \\
      --dataset Data_Prep/dataset_full
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_PREP   = PROJECT_DIR / "Data_Prep"

# ── Spectrogram constants ──────────────────────────────────────────────────────
N_MELS     = 128
MAX_FRAMES = 87


# ── Model (must match architecture used during training) ──────────────────────

class TaamCNN(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 10, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ── Grad-CAM implementation ────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for a CNN.

    Hooks into the target layer to capture:
      - forward  activations (feature maps)
      - backward gradients  (gradient of the score w.r.t. feature maps)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.activations  = None
        self.gradients    = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for `class_idx`.

        Parameters
        ----------
        input_tensor : (1, 1, H, W) float tensor on the same device as model
        class_idx    : integer class index to explain

        Returns
        -------
        heatmap : (H, W) float32 array in [0, 1]
        """
        self.model.eval()

        # Forward pass
        logits = self.model(input_tensor)                    # (1, num_classes)
        score  = logits[0, class_idx]

        # Backward pass — gradient of the class score w.r.t. feature maps
        self.model.zero_grad()
        score.backward()

        # Global-average-pool the gradients over spatial dims  → (C,)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activations             → (H', W')
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)                                             # only positive influence

        # Upsample to original input resolution
        cam = F.interpolate(
            cam,
            size=(N_MELS, MAX_FRAMES),
            mode='bilinear',
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()  # (H, W)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_class_samples(
    dataset_dir: Path,
    class_names: list,
    n_per_class: int = 30,
    include_augmented: bool = False,
) -> dict:
    """
    Load up to `n_per_class` spectrogram arrays per class.

    Returns
    -------
    dict mapping class_name -> list of np.ndarray (N_MELS, MAX_FRAMES)
    """
    samples = {}
    for taam in class_names:
        taam_dir = dataset_dir / taam
        if not taam_dir.is_dir():
            samples[taam] = []
            continue

        paths = sorted(taam_dir.glob("*.npy"))
        if not include_augmented:
            paths = [p for p in paths
                     if not (p.stem.endswith("_aug_ts") or p.stem.endswith("_aug_ps"))]

        chosen = paths[:n_per_class]
        arrays = []
        for p in chosen:
            arr = np.load(str(p)).astype(np.float32)
            # Canonical shape
            h, w = arr.shape
            if h != N_MELS:
                arr = arr[:N_MELS] if h > N_MELS else np.pad(
                    arr, ((0, N_MELS - h), (0, 0)), mode='constant')
            if w != MAX_FRAMES:
                arr = arr[:, :MAX_FRAMES] if w > MAX_FRAMES else np.pad(
                    arr, ((0, 0), (0, MAX_FRAMES - w)), mode='constant')
            arrays.append(arr)

        samples[taam] = arrays

    return samples


def pick_correct_sample(
    model: nn.Module,
    device: torch.device,
    class_idx: int,
    candidates: list,
) -> np.ndarray | None:
    """
    Return the first candidate that the model correctly classifies.
    Falls back to the first candidate if none are correct.
    """
    model.eval()
    for arr in candidates:
        t = torch.FloatTensor(arr[None, None, :, :]).to(device)
        with torch.no_grad():
            pred = model(t).argmax(dim=1).item()
        if pred == class_idx:
            return arr
    # Fallback — return first regardless
    return candidates[0] if candidates else None


# ── Visualisation ──────────────────────────────────────────────────────────────

def save_gradcam_figure(
    spec: np.ndarray,
    heatmap: np.ndarray,
    class_name: str,
    class_idx: int,
    out_path: Path,
    confidence: float,
):
    """
    Save a 3-panel figure:
      Left  : original Mel-Spectrogram
      Middle: Grad-CAM heatmap (greyscale)
      Right : overlay (spectrogram + heatmap)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Colour-map limits from the spectrogram itself
    vmin, vmax = spec.min(), spec.max()

    # Panel 1 — original spectrogram
    im0 = axes[0].imshow(spec, aspect='auto', origin='lower',
                          cmap='magma', vmin=vmin, vmax=vmax)
    axes[0].set_title('Mel-Spectrogram', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time frame'); axes[0].set_ylabel('Mel bin')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='dB')

    # Panel 2 — Grad-CAM heatmap alone
    im1 = axes[1].imshow(heatmap, aspect='auto', origin='lower', cmap='jet',
                          vmin=0, vmax=1)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time frame'); axes[1].set_ylabel('Mel bin')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Activation')

    # Panel 3 — overlay
    # Normalise spectrogram to [0,1] for blending
    spec_norm = (spec - vmin) / max(vmax - vmin, 1e-8)
    axes[2].imshow(spec_norm, aspect='auto', origin='lower',
                   cmap='magma', alpha=1.0)
    axes[2].imshow(heatmap,   aspect='auto', origin='lower',
                   cmap='jet',   alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time frame'); axes[2].set_ylabel('Mel bin')

    class_label = class_name.replace('_', ' ')
    fig.suptitle(
        f'Grad-CAM — Class: {class_label}  |  Model confidence: {confidence*100:.1f}%',
        fontsize=14, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM for TaamAI CNN")
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Path to .pth model file. "
             "Auto-detected: loro_fold3_model.pth → model_r1_top5_balanced_pytorch.pth.",
    )
    parser.add_argument(
        "--classes", type=Path, default=None,
        help="Path to classes JSON (keys: class_names). "
             "Auto-detected alongside the model file.",
    )
    parser.add_argument(
        "--dataset", type=Path, default=None,
        help="Dataset directory with class sub-dirs. "
             "Auto-detected: dataset_r3/ → dataset_full/.",
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_DIR,
        help="Output directory for PNG files (default: project root).",
    )
    parser.add_argument(
        "--candidates", type=int, default=30,
        help="Number of candidate samples per class to search for a correct one.",
    )
    return parser.parse_args()


def resolve_defaults(args):
    """Fill in None arguments with sensible auto-detected defaults."""
    # Model
    if args.model is None:
        for candidate in [
            PROJECT_DIR / "loro_fold3_model.pth",
            PROJECT_DIR / "loro_fold2_model.pth",
            PROJECT_DIR / "loro_fold1_model.pth",
            PROJECT_DIR / "model_r1_top5_balanced_pytorch.pth",
        ]:
            if candidate.exists():
                args.model = candidate
                break

    if args.model is None or not args.model.exists():
        raise FileNotFoundError(
            "No model .pth file found. Train a model first or pass --model."
        )

    # Classes JSON — look alongside the model, then fallback
    if args.classes is None:
        model_stem = args.model.stem  # e.g. "loro_fold3_model"
        for candidate in [
            PROJECT_DIR / "loro_classes.json",
            PROJECT_DIR / "classes_top5.json",
            PROJECT_DIR / "classes.json",
        ]:
            if candidate.exists():
                args.classes = candidate
                break

    if args.classes is None or not args.classes.exists():
        raise FileNotFoundError(
            "No classes JSON found. Pass --classes explicitly."
        )

    # Dataset directory
    if args.dataset is None:
        for candidate in [
            DATA_PREP / "dataset_r3",
            DATA_PREP / "dataset_r2",
            DATA_PREP / "dataset_full",
            DATA_PREP / "dataset",
        ]:
            if candidate.is_dir() and any(candidate.iterdir()):
                args.dataset = candidate
                break

    if args.dataset is None or not args.dataset.is_dir():
        raise FileNotFoundError(
            "No dataset directory found. Pass --dataset explicitly."
        )

    return args


def main():
    args = parse_args()

    print("=" * 70)
    print("STEP 7: GRAD-CAM INTERPRETABILITY VISUALISATIONS")
    print("=" * 70)

    try:
        args = resolve_defaults(args)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print(f"Model   : {args.model}")
    print(f"Classes : {args.classes}")
    print(f"Dataset : {args.dataset}")
    print(f"Output  : {args.out}")

    # ── Load class names ─────────────────────────────────────────────────
    with open(str(args.classes), 'r', encoding='utf-8') as f:
        cls_data = json.load(f)

    # Support both {"class_names": [...]} and {"name": idx, ...} formats
    if "class_names" in cls_data:
        class_names = cls_data["class_names"]
    else:
        class_names = [k for k, v in sorted(cls_data.items(), key=lambda x: x[1])]

    print(f"Classes ({len(class_names)}): {class_names}")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    model = TaamCNN(num_classes=len(class_names)).to(device)
    state = torch.load(str(args.model), map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.")

    # ── Attach Grad-CAM to the ReLU inside conv3 ──────────────────────────
    # conv3 is a Sequential; index [2] is the ReLU after BN
    target_layer = model.conv3[2]   # ReLU
    gradcam = GradCAM(model, target_layer)

    # ── Load candidate samples ────────────────────────────────────────────
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\nLoading up to {args.candidates} candidates per class...")
    all_samples = load_class_samples(
        dataset_dir=args.dataset,
        class_names=class_names,
        n_per_class=args.candidates,
        include_augmented=False,
    )

    # ── Generate one Grad-CAM per class ───────────────────────────────────
    print()
    saved_files = []

    for class_idx, class_name in enumerate(class_names):
        candidates = all_samples.get(class_name, [])
        if not candidates:
            print(f"  [{class_name}] No samples available — skipping.")
            continue

        spec = pick_correct_sample(model, device, class_idx, candidates)
        if spec is None:
            print(f"  [{class_name}] Could not pick a sample — skipping.")
            continue

        # Grad-CAM forward + backward
        input_t = torch.FloatTensor(spec[None, None, :, :]).to(device)
        input_t.requires_grad_(False)

        heatmap = gradcam(input_t, class_idx)

        # Confidence of the prediction
        with torch.no_grad():
            logits = model(input_t)
            probs  = torch.softmax(logits, dim=1)
            confidence = probs[0, class_idx].item()

        # Save figure
        safe_name = class_name.lower().replace(' ', '_')
        out_path  = args.out / f"grad_cam_{safe_name}.png"
        save_gradcam_figure(
            spec=spec,
            heatmap=heatmap,
            class_name=class_name,
            class_idx=class_idx,
            out_path=out_path,
            confidence=confidence,
        )
        saved_files.append(out_path.name)
        print(f"  [{class_name}]  confidence={confidence*100:.1f}%  -> {out_path.name}")

    gradcam.remove_hooks()

    print()
    print("=" * 70)
    print("GRAD-CAM COMPLETE")
    print("=" * 70)
    print(f"Saved {len(saved_files)} figures in {args.out}:")
    for fname in saved_files:
        print(f"  {fname}")
    print()
    print("Interpretation guide:")
    print("  Red/yellow areas  = regions with HIGH influence on the prediction")
    print("  Blue areas        = regions with LOW  influence on the prediction")
    print("  Frequency axis (Y): low mel bins = low pitch, high bins = high pitch")
    print("  Time axis      (X): left = onset, right = offset of the cantillation")


if __name__ == "__main__":
    main()
