import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set paths
BASE_DIR = Path("g:/My Drive/02_Academia/DataMining/S2/Deep Learning/Project")
DATA_DIR = BASE_DIR / "data" / "R1" / "dataset"
OUTPUT_DIR = BASE_DIR

# Count files in each class directory
class_counts = defaultdict(int)
for class_dir in DATA_DIR.iterdir():
    if class_dir.is_dir() and not class_dir.name.startswith('.'):
        npy_files = list(class_dir.glob("*.npy"))
        class_counts[class_dir.name] = len(npy_files)

# Sort by count and get top 5
top5_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 classes by sample count:")
for class_name, count in top5_classes:
    print(f"  {class_name}: {count} samples")

# Extract class names and create label mapping
class_names = [cls[0] for cls in top5_classes]
label_to_idx = {name: idx for idx, name in enumerate(class_names)}

# Load all .npy files from top 5 classes
X_data = []
y_labels = []

for class_name in class_names:
    class_dir = DATA_DIR / class_name
    npy_files = list(class_dir.glob("*.npy"))
    print(f"Loading {len(npy_files)} files from {class_name}...")
    
    for npy_file in npy_files:
        try:
            mel_spec = np.load(npy_file)
            X_data.append(mel_spec)
            y_labels.append(label_to_idx[class_name])
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")

# Convert to numpy arrays
X_data = np.array(X_data)
y_labels = np.array(y_labels)

print(f"\nFinal data shape: {X_data.shape}")
print(f"Labels shape: {y_labels.shape}")
print(f"Expected shape: (num_samples, 128, 87)")

# Verify shape
if X_data.shape[1:] != (128, 87):
    print(f"WARNING: Shape is {X_data.shape[1:]}, expected (128, 87)")

# Save the packaged data
print("\nSaving packaged data...")
np.save(OUTPUT_DIR / "X_data_top5.npy", X_data)
np.save(OUTPUT_DIR / "y_labels_top5.npy", y_labels)

# Save class mapping
classes_info = {
    "class_names": class_names,
    "label_to_idx": label_to_idx,
    "class_counts": {name: count for name, count in top5_classes}
}

with open(OUTPUT_DIR / "classes_top5.json", 'w', encoding='utf-8') as f:
    json.dump(classes_info, f, indent=2, ensure_ascii=False)

print("\nData packaging complete!")
print(f"Saved files:")
print(f"  - X_data_top5.npy (shape: {X_data.shape})")
print(f"  - y_labels_top5.npy (shape: {y_labels.shape})")
print(f"  - classes_top5.json")
