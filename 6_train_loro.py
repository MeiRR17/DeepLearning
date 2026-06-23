"""
6_train_loro.py
===============
Leave-One-Reader-Out (LORO) training and evaluation.

Strategy
--------
  We have 3 readers (R1, R2, R3).  For each fold k in {1, 2, 3}:
    - Train  on the spectrograms from the other 2 readers.
    - Validate during training on 10 % of the training pool (random).
    - Test   (held-out) on ALL spectrograms from reader k.

  This demonstrates that the model learns the *melody* of the cantillation
  marks rather than the *voice* of a specific reader.

Dataset layout expected
-----------------------
  Data_Prep/
    dataset_full/          <- R1 spectrograms  (class sub-dirs, *.npy)
    dataset_r2/            <- R2 spectrograms  (same layout)
    dataset_r3/            <- R3 spectrograms  (same layout)

  Augmented files (*_aug_ts.npy, *_aug_ps.npy) inside each directory are
  automatically included in the training pool (but never in the test set).

Outputs (saved to project root)
---------------------------------------------------------------------------
  loro_fold<k>_model.pth            best model for fold k
  loro_fold<k>_loss_curve.png       training / validation loss+accuracy
  loro_fold<k>_confusion_matrix.png test confusion matrix
  loro_fold<k>_metrics.txt          classification report
  loro_summary.txt                  cross-fold summary

Usage
-----
  python 6_train_loro.py [--folds 1 2 3] [--epochs 40] [--batch 32]
"""

import argparse
import copy
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_PREP   = PROJECT_DIR / "Data_Prep"

READER_DIRS = {
    "R1": DATA_PREP / "dataset_full",
    "R2": DATA_PREP / "dataset_r2",
    "R3": DATA_PREP / "dataset_r3",
}

# ── Spectrogram dimensions ───────────────────────────────────────────────────
N_MELS     = 128
MAX_FRAMES = 87

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)


# ── Model architecture (identical to 5_train_regularized.py) ────────────────

class TaamCNN(nn.Module):
    """3-block CNN with BatchNorm, Dropout2d, and fully-connected head."""

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
        # After 3× MaxPool(2,2): H=128→16, W=87→10
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


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7):
        self.patience   = patience
        self.counter    = 0
        self.best_loss  = None
        self.best_epoch = 0
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss: float, epoch: int, model: nn.Module):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter    = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_reader_dataset(reader_dir: Path, class_names: list[str],
                        include_augmented: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all .npy spectrograms from reader_dir/<taam>/*.npy.

    Parameters
    ----------
    reader_dir       : Root directory for this reader's spectrograms.
    class_names      : Ordered list of class names (defines label indices).
    include_augmented: If False, skip files ending in _aug_ts.npy / _aug_ps.npy.

    Returns
    -------
    X : float32 array of shape (N, N_MELS, MAX_FRAMES)
    y : int32   array of shape (N,)
    """
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    X_list, y_list = [], []

    for taam in class_names:
        taam_dir = reader_dir / taam
        if not taam_dir.is_dir():
            continue

        for npy_path in sorted(taam_dir.glob("*.npy")):
            stem = npy_path.stem
            if not include_augmented and (
                stem.endswith("_aug_ts") or stem.endswith("_aug_ps")
            ):
                continue

            arr = np.load(str(npy_path)).astype(np.float32)

            # Pad / truncate to canonical shape
            h, w = arr.shape
            if h != N_MELS:
                arr = arr[:N_MELS] if h > N_MELS else np.pad(
                    arr, ((0, N_MELS - h), (0, 0)), mode='constant')
            if w != MAX_FRAMES:
                arr = arr[:, :MAX_FRAMES] if w > MAX_FRAMES else np.pad(
                    arr, ((0, 0), (0, MAX_FRAMES - w)), mode='constant')

            X_list.append(arr)
            y_list.append(name_to_idx[taam])

    if not X_list:
        return np.empty((0, N_MELS, MAX_FRAMES), dtype=np.float32), np.empty(0, dtype=np.int32)

    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int32)


def discover_classes(reader_dirs: list[Path]) -> list[str]:
    """Collect the union of class names across all available reader directories."""
    classes = set()
    for rd in reader_dirs:
        if rd.is_dir():
            for d in rd.iterdir():
                if d.is_dir():
                    classes.add(d.name)
    return sorted(classes)


# ── Training one fold ─────────────────────────────────────────────────────────

def train_fold(
    fold_idx: int,
    held_out_reader: str,
    train_readers: list[str],
    class_names: list[str],
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    val_fraction: float = 0.10,
) -> dict:
    """
    Train and evaluate a single LORO fold.

    Returns a dict with metrics for the summary report.
    """
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx} | Held-out: {held_out_reader} | Train: {train_readers}")
    print(f"{'='*70}")

    # ── Load training data (with augmentation) ────────────────────────────
    X_parts, y_parts = [], []
    for rdr in train_readers:
        rdir = READER_DIRS[rdr]
        if not rdir.is_dir():
            print(f"  WARNING: {rdir} not found — skipping {rdr}.")
            continue
        X_r, y_r = load_reader_dataset(rdir, class_names, include_augmented=True)
        print(f"  {rdr}: {X_r.shape[0]} samples (incl. augmented)")
        X_parts.append(X_r)
        y_parts.append(y_r)

    if not X_parts:
        print("  ERROR: No training data found — skipping fold.")
        return {}

    X_train_all = np.concatenate(X_parts, axis=0)
    y_train_all = np.concatenate(y_parts, axis=0)

    # ── Shuffle + split validation set ────────────────────────────────────
    rng = np.random.default_rng(42 + fold_idx)
    perm = rng.permutation(len(X_train_all))
    X_train_all = X_train_all[perm]
    y_train_all = y_train_all[perm]

    val_n        = max(1, int(len(X_train_all) * val_fraction))
    X_val        = X_train_all[:val_n]
    y_val        = y_train_all[:val_n]
    X_train      = X_train_all[val_n:]
    y_train      = y_train_all[val_n:]

    print(f"  Training samples   : {len(X_train)}")
    print(f"  Validation samples : {len(X_val)}")

    # ── Load held-out test set (originals only — no augmentation) ─────────
    test_dir = READER_DIRS[held_out_reader]
    if not test_dir.is_dir():
        print(f"  ERROR: Held-out reader directory {test_dir} not found.")
        return {}

    X_test, y_test = load_reader_dataset(test_dir, class_names, include_augmented=False)
    print(f"  Test samples (held-out {held_out_reader}) : {X_test.shape[0]}")

    if X_test.shape[0] == 0:
        print(f"  ERROR: No test samples for {held_out_reader} — skipping fold.")
        return {}

    # ── DataLoaders ────────────────────────────────────────────────────────
    def make_loader(X, y, shuffle):
        t = TensorDataset(
            torch.FloatTensor(X[:, None, :, :]),   # (N,1,H,W)
            torch.LongTensor(y),
        )
        return DataLoader(t, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    # ── Class weights ──────────────────────────────────────────────────────
    unique_classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    # Expand to full num_classes (some classes may be missing)
    weights = np.ones(len(class_names), dtype=np.float32)
    for cls, w in zip(unique_classes, cw):
        weights[cls] = w
    cw_tensor = torch.FloatTensor(weights).to(device)

    # ── Model ──────────────────────────────────────────────────────────────
    model     = TaamCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=False
    )
    stopper   = EarlyStopping(patience=7)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # ── Epoch loop ─────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        # Training
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tl += loss.item() * inputs.size(0)
            _, pred = torch.max(out, 1)
            tc += (pred == labels).sum().item()
            tt += labels.size(0)

        # Validation
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                out  = model(inputs)
                loss = criterion(out, labels)
                vl += loss.item() * inputs.size(0)
                _, pred = torch.max(out, 1)
                vc += (pred == labels).sum().item()
                vt += labels.size(0)

        tr_loss = tl / tt;  tr_acc = tc / tt
        v_loss  = vl / vt;  v_acc  = vc / vt

        train_losses.append(tr_loss);  train_accs.append(tr_acc)
        val_losses.append(v_loss);     val_accs.append(v_acc)

        scheduler.step(v_loss)
        stopper(v_loss, epoch + 1, model)

        print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Val {v_loss:.4f}/{v_acc:.4f}"
              + (" [BEST]" if stopper.counter == 0 else ""))

        if stopper.early_stop:
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(best epoch={stopper.best_epoch})")
            break

    # Restore best weights
    model.load_state_dict(stopper.best_state)

    # ── Save model ─────────────────────────────────────────────────────────
    model_path = PROJECT_DIR / f"loro_fold{fold_idx}_model.pth"
    torch.save(model.state_dict(), str(model_path))
    print(f"  Model saved: {model_path.name}")

    # ── Evaluate on held-out test set ──────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            out    = model(inputs)
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc   = (all_preds == all_labels).mean()

    print(f"\n  TEST ACCURACY (held-out {held_out_reader}): {test_acc*100:.2f}%")

    # ── Loss/Accuracy curves ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    epochs_ran = range(1, len(train_accs) + 1)

    axes[0].plot(epochs_ran, train_accs, label='Train Acc', linewidth=2)
    axes[0].plot(epochs_ran, val_accs,   label='Val Acc',   linewidth=2)
    axes[0].axvline(stopper.best_epoch, color='r', linestyle='--', alpha=0.6,
                    label=f'Best epoch ({stopper.best_epoch})')
    axes[0].set_title(f'Fold {fold_idx}: Accuracy  |  Test={test_acc*100:.1f}%',
                      fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_ran, train_losses, label='Train Loss', linewidth=2)
    axes[1].plot(epochs_ran, val_losses,   label='Val Loss',   linewidth=2)
    axes[1].axvline(stopper.best_epoch, color='r', linestyle='--', alpha=0.6)
    axes[1].set_title(f'Fold {fold_idx}: Loss  |  Held-out: {held_out_reader}',
                      fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'LORO Fold {fold_idx}  |  Train: {train_readers}  |  Test: {held_out_reader}',
                 fontsize=11)
    plt.tight_layout()
    curve_path = PROJECT_DIR / f"loro_fold{fold_idx}_loss_curve.png"
    plt.savefig(str(curve_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {curve_path.name}")

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names) - 1)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix  |  Fold {fold_idx}  |  Held-out: {held_out_reader}  '
              f'|  Acc={test_acc*100:.1f}%',
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    cm_path = PROJECT_DIR / f"loro_fold{fold_idx}_confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {cm_path.name}")

    # ── Classification report ──────────────────────────────────────────────
    # Only include labels that actually appear in test set
    present_labels = sorted(np.unique(all_labels).tolist())
    present_names  = [class_names[i] for i in present_labels]
    report = classification_report(
        all_labels, all_preds,
        labels=present_labels, target_names=present_names, digits=4,
        zero_division=0,
    )

    metrics_path = PROJECT_DIR / f"loro_fold{fold_idx}_metrics.txt"
    with open(str(metrics_path), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"LORO FOLD {fold_idx} — Held-out: {held_out_reader}\n")
        f.write(f"Trained on: {train_readers}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test Accuracy: {test_acc*100:.4f}%\n")
        f.write(f"Best Epoch   : {stopper.best_epoch}\n")
        f.write(f"Best Val Loss: {stopper.best_loss:.4f}\n\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 70 + "\n")
        f.write(report)
        f.write("\n\nPER-CLASS TEST ACCURACY\n")
        f.write("-" * 70 + "\n")
        for i, cn in enumerate(class_names):
            row = cm[i]
            total = row.sum()
            acc_i = cm[i, i] / total if total > 0 else 0.0
            f.write(f"  {cn:22s}: {acc_i*100:6.2f}%  ({cm[i,i]}/{total})\n")

    print(f"  Saved: {metrics_path.name}")

    return {
        "fold":         fold_idx,
        "held_out":     held_out_reader,
        "train":        train_readers,
        "test_acc":     test_acc,
        "best_epoch":   stopper.best_epoch,
        "epochs_ran":   len(train_accs),
        "cm":           cm,
        "class_names":  class_names,
    }


# ── Summary report ────────────────────────────────────────────────────────────

def write_summary(results: list[dict]):
    summary_path = PROJECT_DIR / "loro_summary.txt"
    with open(str(summary_path), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("LEAVE-ONE-READER-OUT (LORO) CROSS-VALIDATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        accs = []
        for r in results:
            acc = r['test_acc']
            accs.append(acc)
            f.write(f"Fold {r['fold']}  |  Held-out: {r['held_out']:3s}  "
                    f"|  Train: {r['train']}  "
                    f"|  Test Acc: {acc*100:.2f}%  "
                    f"|  Best Epoch: {r['best_epoch']}\n")
        f.write("\n" + "-" * 70 + "\n")
        if accs:
            f.write(f"Mean Test Accuracy : {np.mean(accs)*100:.2f}%\n")
            f.write(f"Std  Test Accuracy : {np.std(accs)*100:.2f}%\n")
        f.write("=" * 70 + "\n")

    print(f"\nSummary saved: {summary_path.name}")
    if results:
        print(f"Mean LORO Accuracy: {np.mean([r['test_acc'] for r in results])*100:.2f}%")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LORO training for TaamAI")
    parser.add_argument(
        "--folds", nargs="+", type=int, default=[1, 2, 3],
        metavar="K",
        help="Which fold indices to run (1=hold R1, 2=hold R2, 3=hold R3). "
             "Default: all three.",
    )
    parser.add_argument("--epochs",  type=int, default=40, help="Max training epochs per fold.")
    parser.add_argument("--batch",   type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Explicit list of class names. If omitted, auto-detected from dataset directories.",
    )
    return parser.parse_args()


FOLD_CONFIG = {
    1: {"held_out": "R1", "train": ["R2", "R3"]},
    2: {"held_out": "R2", "train": ["R1", "R3"]},
    3: {"held_out": "R3", "train": ["R1", "R2"]},
}


def main():
    args = parse_args()

    print("=" * 70)
    print("LEAVE-ONE-READER-OUT (LORO) TRAINING — TaamAI")
    print("=" * 70)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("WARNING: No GPU detected, running on CPU (will be slow).")

    os.chdir(str(PROJECT_DIR))

    # Class names
    if args.classes:
        class_names = args.classes
    else:
        class_names = discover_classes(list(READER_DIRS.values()))
        if not class_names:
            print("ERROR: No class directories found in any reader dataset.")
            sys.exit(1)

    print(f"\nClasses ({len(class_names)}): {class_names}")

    # Save classes JSON for downstream use
    classes_out = PROJECT_DIR / "loro_classes.json"
    with open(str(classes_out), 'w', encoding='utf-8') as f:
        json.dump({"class_names": class_names,
                   "label_to_idx": {n: i for i, n in enumerate(class_names)}}, f, indent=2)
    print(f"Classes saved: {classes_out.name}")

    # Validate requested fold indices
    invalid = [k for k in args.folds if k not in FOLD_CONFIG]
    if invalid:
        print(f"ERROR: Invalid fold indices {invalid}. Valid: {list(FOLD_CONFIG.keys())}")
        sys.exit(1)

    # Check which reader directories actually exist
    for rdr, rdir in READER_DIRS.items():
        status = "OK" if rdir.is_dir() and any(rdir.iterdir()) else "MISSING/EMPTY"
        print(f"  {rdr}: {rdir}  [{status}]")

    print()

    all_results = []
    for fold_k in sorted(args.folds):
        cfg    = FOLD_CONFIG[fold_k]
        result = train_fold(
            fold_idx       = fold_k,
            held_out_reader= cfg["held_out"],
            train_readers  = cfg["train"],
            class_names    = class_names,
            device         = device,
            num_epochs     = args.epochs,
            batch_size     = args.batch,
        )
        if result:
            all_results.append(result)

    write_summary(all_results)

    print("\n" + "=" * 70)
    print("LORO TRAINING COMPLETE")
    print("=" * 70)
    print("Outputs in project root:")
    for fold_k in sorted(args.folds):
        for suffix in ["_model.pth", "_loss_curve.png",
                       "_confusion_matrix.png", "_metrics.txt"]:
            print(f"  loro_fold{fold_k}{suffix}")
    print("  loro_summary.txt")
    print("  loro_classes.json")
    print("\nNext step: run 7_gradcam.py for interpretability visualizations.")


if __name__ == "__main__":
    main()
