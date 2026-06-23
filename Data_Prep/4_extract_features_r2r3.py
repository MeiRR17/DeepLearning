"""
4_extract_features_r2r3.py
==========================
Extract Mel-Spectrogram features from R2 and R3 reader audio files.

This script mirrors 2_extract_features.py but:
  - Accepts a per-reader aligned JSON and WAV directory as inputs.
  - Tags each output .npy filename with the reader prefix so all readers'
    datasets can coexist under Data_Prep/dataset_r<N>/ without filename
    collisions.
  - Output directories are Data_Prep/dataset_r2/ and Data_Prep/dataset_r3/
    (one directory per reader, same class-subdirectory layout as dataset_full/).

Prerequisites
-------------
  Each reader requires an aligned_dataset_<reader>.json produced by the
  alignment pipeline (1_align_vtt.py) for that reader's VTT + WAV files.
  Place those JSON files alongside this script, or pass --json explicitly.

Usage
-----
  # Process both R2 and R3 (default)
  python Data_Prep/4_extract_features_r2r3.py

  # Process only R2 with explicit paths
  python Data_Prep/4_extract_features_r2r3.py \\
      --reader R2 \\
      --json  Data_Prep/aligned_dataset_r2.json \\
      --wav   data/R2/wav \\
      --out   Data_Prep/dataset_r2

After running this script, execute 3_augment_dataset.py to generate augmented
copies, then proceed to 6_train_loro.py for Leave-One-Reader-Out training.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ── Audio / spectrogram parameters (must match 2_extract_features.py) ──────
SR         = 22050
MAX_FRAMES = 87    # ~2 seconds
N_MELS     = 128
FMAX       = 8000
PAD_SEC    = 0.05
MIN_DUR    = 0.1   # seconds

BASE_DIR = Path(__file__).parent.parent   # project root


# ── Core extraction function ────────────────────────────────────────────────

def extract_features_for_reader(
    reader_name: str,
    json_path: Path,
    wav_dir: Path,
    out_dir: Path,
) -> dict:
    """
    Extract Mel-Spectrograms for one reader and save to out_dir/<taam>/.

    Parameters
    ----------
    reader_name : str  e.g. "R2"
    json_path   : Path to aligned_dataset_<reader>.json
    wav_dir     : Path to the directory containing .wav files for this reader
    out_dir     : Output root directory (class subdirs created automatically)

    Returns
    -------
    dict with keys: saved, skipped, taam_counts
    """
    if not json_path.exists():
        print(f"[{reader_name}] ERROR: JSON not found at {json_path}")
        return {}

    if not wav_dir.is_dir():
        print(f"[{reader_name}] ERROR: WAV directory not found at {wav_dir}")
        return {}

    print(f"\n{'='*60}")
    print(f"[{reader_name}] Loading aligned dataset: {json_path.name}")
    with open(str(json_path), 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"[{reader_name}] {len(data)} segments loaded.")

    # Group by WAV file — load each file once
    file_to_items = defaultdict(list)
    for item in data:
        file_to_items[item['audio_file']].append(item)

    out_dir.mkdir(parents=True, exist_ok=True)

    saved       = 0
    skipped     = 0
    taam_counts = defaultdict(int)
    reader_tag  = reader_name.lower()  # e.g. "r2"

    for wav_file, items in tqdm(file_to_items.items(),
                                desc=f"[{reader_name}] Processing WAVs"):
        wav_path = wav_dir / wav_file

        if not wav_path.exists():
            print(f"\n  WARNING: {wav_file} not found in {wav_dir} — skipping.")
            skipped += len(items)
            continue

        # Load full WAV once
        try:
            sf_info   = sf.info(str(wav_path))
            native_sr = sf_info.samplerate
            y_full, _ = sf.read(str(wav_path), dtype='float32', always_2d=True)
            y_full    = y_full.mean(axis=1)  # stereo → mono
            if native_sr != SR:
                y_full = librosa.resample(y_full, orig_sr=native_sr, target_sr=SR)
        except Exception as e:
            print(f"\n  ERROR loading {wav_file}: {e}")
            skipped += len(items)
            continue

        for idx, item in enumerate(items):
            taam       = item['taam']
            start_time = float(item['start'])
            end_time   = float(item['end'])

            start_sample = max(0, int((start_time - PAD_SEC) * SR))
            end_sample   = min(len(y_full), int((end_time + PAD_SEC) * SR))

            if end_sample <= start_sample:
                skipped += 1
                continue

            y_slice = y_full[start_sample:end_sample]

            if len(y_slice) < SR * MIN_DUR:
                skipped += 1
                continue

            # Mel-Spectrogram
            mel_spec    = librosa.feature.melspectrogram(
                y=y_slice, sr=SR, n_mels=N_MELS, fmax=FMAX
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Pad / truncate to MAX_FRAMES
            if mel_spec_db.shape[1] < MAX_FRAMES:
                pad_w       = MAX_FRAMES - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_w)), mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :MAX_FRAMES]

            # Save: out_dir/<taam>/<reader>_<wav_stem>_<idx:04d>.npy
            taam_dir = out_dir / taam
            taam_dir.mkdir(exist_ok=True)

            wav_stem = wav_file.replace('.wav', '').replace('.WAV', '')
            out_name  = f"{reader_tag}_{wav_stem}_{idx:04d}.npy"
            out_path  = taam_dir / out_name

            # Long-path prefix for Windows
            long_path = "\\\\?\\" + str(out_path.resolve())
            with open(long_path, 'wb') as fh:
                np.save(fh, mel_spec_db.astype(np.float32))

            saved += 1
            taam_counts[taam] += 1

    print(f"\n[{reader_name}] Saved: {saved}  |  Skipped: {skipped}")
    print(f"[{reader_name}] Output directory: {out_dir}")
    print(f"[{reader_name}] Class distribution:")
    for taam, count in sorted(taam_counts.items(), key=lambda x: -x[1]):
        print(f"     {taam:22s}: {count:4d}")

    return {"reader": reader_name, "saved": saved, "skipped": skipped,
            "taam_counts": dict(taam_counts)}


# ── Default reader configurations ───────────────────────────────────────────

DEFAULT_READERS = {
    "R2": {
        "json": BASE_DIR / "Data_Prep" / "aligned_dataset_r2.json",
        "wav":  BASE_DIR / "data" / "R2" / "wav",
        "out":  BASE_DIR / "Data_Prep" / "dataset_r2",
    },
    "R3": {
        "json": BASE_DIR / "Data_Prep" / "aligned_dataset_r3.json",
        "wav":  BASE_DIR / "data" / "R3" / "wav",
        "out":  BASE_DIR / "Data_Prep" / "dataset_r3",
    },
}


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Mel-Spectrogram features for R2 and/or R3 readers."
    )
    parser.add_argument(
        "--reader", choices=["R2", "R3"], default=None,
        help="Process only this reader (default: both R2 and R3)."
    )
    parser.add_argument("--json", type=Path, default=None,
                        help="Explicit path to aligned JSON (only with --reader).")
    parser.add_argument("--wav",  type=Path, default=None,
                        help="Explicit WAV directory    (only with --reader).")
    parser.add_argument("--out",  type=Path, default=None,
                        help="Explicit output directory  (only with --reader).")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("STEP 4: MEL-SPECTROGRAM EXTRACTION FOR R2 / R3")
    print("=" * 60)

    if args.reader:
        # Single reader, explicit paths allowed
        cfg = DEFAULT_READERS[args.reader].copy()
        if args.json: cfg["json"] = args.json
        if args.wav:  cfg["wav"]  = args.wav
        if args.out:  cfg["out"]  = args.out

        readers_to_process = {args.reader: cfg}
    else:
        readers_to_process = DEFAULT_READERS

    all_results = []
    for reader_name, cfg in readers_to_process.items():
        result = extract_features_for_reader(
            reader_name=reader_name,
            json_path=cfg["json"],
            wav_dir=cfg["wav"],
            out_dir=cfg["out"],
        )
        if result:
            all_results.append(result)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    total_saved   = sum(r.get("saved", 0)   for r in all_results)
    total_skipped = sum(r.get("skipped", 0) for r in all_results)
    print(f"Total spectrograms saved  : {total_saved}")
    print(f"Total segments skipped    : {total_skipped}")
    print()
    print("Next steps:")
    print("  1. Run 3_augment_dataset.py to augment the new reader datasets.")
    print("  2. Run 6_train_loro.py for Leave-One-Reader-Out training.")


if __name__ == "__main__":
    main()
