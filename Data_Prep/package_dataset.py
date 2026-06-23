import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# --- נתיבים ---
DATASET_DIR = Path(__file__).parent / "dataset_full"
OUT_DIR     = Path(__file__).parent.parent   # workspace root

def package_dataset():
    print("📦 אורז dataset ל-3 קבצים...")
    
    # מציאת כל הטעמים וקביעת סדר
    taam_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    classes = {name: idx for idx, name in enumerate([d.name for d in taam_dirs])}
    num_classes = len(classes)
    
    print(f"✓ {num_classes} טעמים שונים")
    
    # ספירת סה"כ קבצים
    total_files = sum(len(list(d.glob("*.npy"))) for d in taam_dirs)
    print(f"✓ {total_files} ספקטרוגרמות סה\"כ")
    
    # אלוקציה מראש
    X_data = np.zeros((total_files, 128, 87), dtype=np.float32)
    y_labels = np.zeros(total_files, dtype=np.int32)
    
    idx = 0
    for taam_dir in taam_dirs:
        taam_name = taam_dir.name
        label = classes[taam_name]
        
        npy_files = sorted(taam_dir.glob("*.npy"))
        print(f"  [{label:2d}] {taam_name:20s}: {len(npy_files):4d} קבצים")
        
        for npy_file in npy_files:
            arr = np.load(str(npy_file))
            X_data[idx] = arr
            y_labels[idx] = label
            idx += 1
    
    # שמירה ב-root של הworkspace
    out_x = OUT_DIR / "X_data.npy"
    out_y = OUT_DIR / "y_labels.npy"
    out_c = OUT_DIR / "classes.json"
    
    print(f"\n💾 שומר...")
    np.save(str(out_x), X_data)
    np.save(str(out_y), y_labels)
    
    with open(out_c, 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ סיום!")
    print(f"   X_data.npy   : {out_x}  shape={X_data.shape}  size={out_x.stat().st_size/1024/1024:.1f}MB")
    print(f"   y_labels.npy : {out_y}  shape={y_labels.shape}  size={out_y.stat().st_size/1024/1024:.1f}MB")
    print(f"   classes.json : {out_c}")
    print(f"\n📂 כל הקבצים ב-root של הworkspace, מוכנים להעלאה לקולאב")

if __name__ == "__main__":
    package_dataset()
