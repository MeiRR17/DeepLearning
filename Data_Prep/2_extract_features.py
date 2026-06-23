import json
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# --- הגדרות נתיבים ---
BASE_DIR  = Path(__file__).parent.parent
JSON_PATH = Path(__file__).parent / "aligned_dataset.json"
WAV_DIR   = BASE_DIR / "data" / "R1" / "wav"
OUT_DIR   = Path(__file__).parent / "dataset_full"

# --- הגדרות עיבוד אודיו ---
SR         = 22050
MAX_FRAMES = 87   # ~2 שניות

def process_audio_files():
    print("📖 קורא את קובץ הנתונים...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # קיבוץ לפי קובץ WAV — כל קובץ נטען פעם אחת בלבד לזיכרון
    file_to_taamim = defaultdict(list)
    for item in data:
        file_to_taamim[item['audio_file']].append(item)

    OUT_DIR.mkdir(exist_ok=True)

    total_saved  = 0
    total_skip   = 0
    taam_counts  = defaultdict(int)

    print(f"🚀 מתחיל לעבד {len(file_to_taamim)} קבצי אודיו...")

    for wav_file, items in tqdm(file_to_taamim.items(), desc="Processing WAVs"):
        wav_path = WAV_DIR / wav_file

        if not wav_path.exists():
            print(f"\n⚠️  {wav_file} חסר. מדלג.")
            continue

        # טען את כל ה-WAV פעם אחת כ-float32 mono — מהיר, זיכרון סביר
        try:
            sf_info   = sf.info(str(wav_path))
            native_sr = sf_info.samplerate
            y_full, _ = sf.read(str(wav_path), dtype='float32', always_2d=True)
            y_full    = y_full.mean(axis=1)          # stereo → mono
            if native_sr != SR:
                y_full = librosa.resample(y_full, orig_sr=native_sr, target_sr=SR)
            sr = SR
        except Exception as e:
            print(f"\n❌  שגיאה בטעינת {wav_file}: {e}")
            continue

        for idx, item in enumerate(items):
            taam       = item['taam']
            start_time = float(item['start'])
            end_time   = float(item['end'])

            pad_sec      = 0.05
            start_sample = max(0, int((start_time - pad_sec) * sr))
            end_sample   = min(len(y_full), int((end_time + pad_sec) * sr))

            if end_sample <= start_sample:
                total_skip += 1
                continue

            y_slice = y_full[start_sample:end_sample]

            # התעלמות מחיתוכים קצרים מדי (< 0.1 שנייה)
            if len(y_slice) < sr * 0.1:
                total_skip += 1
                continue

            # --- ספקטרוגרמת-מל ---
            mel_spec    = librosa.feature.melspectrogram(y=y_slice, sr=sr, n_mels=128, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # נרמול גודל ל-MAX_FRAMES (padding / truncating)
            if mel_spec_db.shape[1] < MAX_FRAMES:
                pad_width   = MAX_FRAMES - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :MAX_FRAMES]

            # שמירה: dataset/<taam>/<parasha>_<idx>.npy
            taam_dir = OUT_DIR / taam
            taam_dir.mkdir(exist_ok=True)

            out_name  = f"{wav_file.replace('.wav', '')}_{idx:04d}.npy"
            out_path  = taam_dir / out_name
            # \\?\ prefix מאפשר נתיבים ארוכים ב-Windows
            long_path = "\\\\?\\" + str(out_path.resolve())
            with open(long_path, 'wb') as fh:
                np.save(fh, mel_spec_db.astype(np.float32))

            total_saved += 1
            taam_counts[taam] += 1

    print(f"\n{'='*60}")
    print(f"✅ סיום! נשמרו {total_saved} ספקטרוגרמות")
    print(f"⏭️  דולגו {total_skip} חיתוכים קצרים מדי")
    print(f"📂 תיקיית פלט: {OUT_DIR}")
    print(f"\n📊 דוגמאות לפי טעם:")
    for taam, count in sorted(taam_counts.items(), key=lambda x: -x[1]):
        print(f"   {taam:20s}: {count:4d}  →  {OUT_DIR / taam}")


if __name__ == "__main__":
    process_audio_files()
