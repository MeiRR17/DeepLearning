"""
3_augment_dataset.py
====================
Data augmentation for Mel-Spectrogram .npy files.

For each existing spectrogram in the source dataset directory, generates two
augmented copies and saves them into the same class subdirectory:

  1. Time-stretch  — resizes spectrogram width by a random factor in [0.8, 1.2].
     This simulates a reader reciting the same note faster or slower.

  2. Pitch-shift   — vertically shifts (rolls) the mel-frequency axis by a random
     amount in [-6, +6] mel bins, wrapping edges with zero-padding (not circular).
     This simulates a reader singing in a slightly higher or lower register.

Augmented files are suffixed:
  <original_name>_aug_ts.npy   (time-stretch)
  <original_name>_aug_ps.npy   (pitch-shift)

Usage
-----
  python Data_Prep/3_augment_dataset.py [--source DATASET_DIR] [--readers R1 R2 R3] [--seed 42]

By default, operates on Data_Prep/dataset_full/ (R1 data already there) and
Data_Prep/dataset_r2/ / Data_Prep/dataset_r3/ when they exist.
Pass --source to override.
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2  # for resize (time-stretch); falls back to scipy if not available


MAX_FRAMES = 87   # fixed width  (time axis)
N_MELS     = 128  # fixed height (frequency axis)


# ── Augmentation helpers ────────────────────────────────────────────────────

def time_stretch_spec(spec: np.ndarray, rate: float) -> np.ndarray:
    """
    Resize the time axis of a spectrogram (shape H x W) by `rate`.
    rate > 1 → slower (wider)  |  rate < 1 → faster (narrower)
    The result is zero-padded or truncated to MAX_FRAMES.
    """
    H, W = spec.shape
    new_W = max(1, int(round(W * rate)))

    try:
        # cv2 expects (width, height) and uint8/float32
        resized = cv2.resize(spec.astype(np.float32), (new_W, H),
                             interpolation=cv2.INTER_LINEAR)
    except Exception:
        # Fallback: numpy-only linear interpolation
        x_old = np.linspace(0, W - 1, W)
        x_new = np.linspace(0, W - 1, new_W)
        resized = np.array(
            [np.interp(x_new, x_old, spec[row]) for row in range(H)],
            dtype=np.float32,
        )

    # Pad or truncate to MAX_FRAMES
    if new_W < MAX_FRAMES:
        pad = MAX_FRAMES - new_W
        out = np.pad(resized, ((0, 0), (0, pad)), mode='constant', constant_values=0)
    else:
        out = resized[:, :MAX_FRAMES]

    return out.astype(np.float32)


def pitch_shift_spec(spec: np.ndarray, shift_bins: int) -> np.ndarray:
    """
    Vertically shift (roll) the mel-frequency axis of a spectrogram by
    `shift_bins` mel bins.  Rows that roll off the edge are zeroed out
    (no circular wrap) so as not to inject spurious energy.
    """
    out = np.roll(spec, shift_bins, axis=0).astype(np.float32)

    if shift_bins > 0:
        out[:shift_bins, :] = 0.0          # zero out top rows that wrapped
    elif shift_bins < 0:
        out[shift_bins:, :] = 0.0          # zero out bottom rows that wrapped

    return out


# ── Main augmentation loop ──────────────────────────────────────────────────

def augment_directory(source_dir: Path, rng: np.random.Generator, skip_existing: bool = True):
    """
    Walk all class subdirectories in `source_dir`, generate augmented copies,
    and save them in-place (same subdirectory as the original).
    """
    taam_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])

    if not taam_dirs:
        print(f"  No class subdirectories found in {source_dir}")
        return

    total_new = 0
    total_skip = 0

    for taam_dir in tqdm(taam_dirs, desc=f"Augmenting {source_dir.name}"):
        originals = sorted(taam_dir.glob("*.npy"))
        # Only process originals — not already-augmented files
        originals = [f for f in originals
                     if not (f.stem.endswith("_aug_ts") or f.stem.endswith("_aug_ps"))]

        for npy_path in originals:
            ts_path = taam_dir / (npy_path.stem + "_aug_ts.npy")
            ps_path = taam_dir / (npy_path.stem + "_aug_ps.npy")

            spec = np.load(str(npy_path)).astype(np.float32)

            # Ensure shape is (N_MELS, MAX_FRAMES)
            if spec.shape != (N_MELS, MAX_FRAMES):
                # Try to fix — pad/truncate silently
                h, w = spec.shape
                if h != N_MELS:
                    spec = spec[:N_MELS] if h > N_MELS else np.pad(
                        spec, ((0, N_MELS - h), (0, 0)), mode='constant')
                if w != MAX_FRAMES:
                    spec = spec[:, :MAX_FRAMES] if w > MAX_FRAMES else np.pad(
                        spec, ((0, 0), (0, MAX_FRAMES - w)), mode='constant')

            # ── 1. Time-stretch ────────────────────────────────────────────
            if skip_existing and ts_path.exists():
                total_skip += 1
            else:
                rate = rng.uniform(0.8, 1.2)
                aug_ts = time_stretch_spec(spec, rate)
                np.save(str(ts_path), aug_ts)
                total_new += 1

            # ── 2. Pitch-shift ─────────────────────────────────────────────
            if skip_existing and ps_path.exists():
                total_skip += 1
            else:
                shift = int(rng.integers(-6, 7))   # -6 to +6 inclusive
                aug_ps = pitch_shift_spec(spec, shift)
                np.save(str(ps_path), aug_ps)
                total_new += 1

    print(f"  -> {total_new} augmented files saved, {total_skip} already existed (skipped).")


# ── Entry point ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Mel-Spectrogram data augmentation")
    parser.add_argument(
        "--source",
        nargs="+",
        default=None,
        help="One or more dataset directories to augment. "
             "Defaults to dataset_full/ and any dataset_r*/ siblings.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-generate even if augmented file already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    skip_existing = not args.no_skip

    base = Path(__file__).parent

    if args.source:
        sources = [Path(s) for s in args.source]
    else:
        # Auto-detect: dataset_full + any dataset_r*/ directories
        sources = [base / "dataset_full"]
        for candidate in sorted(base.glob("dataset_r*/")):
            if candidate.is_dir():
                sources.append(candidate)

    print("=" * 60)
    print("STEP 3: MEL-SPECTROGRAM DATA AUGMENTATION")
    print("=" * 60)
    print(f"Seed: {args.seed}  |  Skip existing: {skip_existing}")
    print(f"Augmentation strategies:")
    print("  1. Time-stretch : random rate in [0.80, 1.20]")
    print("  2. Pitch-shift  : random shift in [-6, +6] mel bins")
    print()

    for src in sources:
        if not src.exists():
            print(f"[SKIP] {src} does not exist — skipping.")
            continue
        print(f"[DIR] {src}")
        augment_directory(src, rng, skip_existing=skip_existing)
        print()

    print("Done. Each original .npy now has two augmented siblings.")
    print("  *_aug_ts.npy  — time-stretched")
    print("  *_aug_ps.npy  — pitch-shifted")


if __name__ == "__main__":
    main()
