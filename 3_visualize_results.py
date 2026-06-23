import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras

# Set paths
BASE_DIR = Path("g:/My Drive/02_Academia/DataMining/S2/Deep Learning/Project")

# Load training history
with open(BASE_DIR / "training_history.json", 'r') as f:
    history = json.load(f)

# Load data and model for confusion matrix
X_data = np.load(BASE_DIR / "X_data_top5.npy")
y_labels = np.load(BASE_DIR / "y_labels_top5.npy")

with open(BASE_DIR / "classes_top5.json", 'r', encoding='utf-8') as f:
    classes_info = json.load(f)

class_names = classes_info['class_names']

# Add channel dimension
X_data = X_data[..., np.newaxis]

# Load trained model
model = keras.models.load_model(BASE_DIR / "model_r1_top5.h5")

# Get predictions
y_pred_probs = model.predict(X_data, batch_size=32, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Create accuracy/loss plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_DIR / "accuracy_loss_plot.png", dpi=300, bbox_inches='tight')
print("Saved: accuracy_loss_plot.png")
plt.close()

# Create confusion matrix
cm = confusion_matrix(y_labels, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix - Top 5 Trope Classes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix.png")
plt.close()

# Print confusion matrix statistics
print("\nConfusion Matrix Statistics:")
print(f"Total samples: {len(y_labels)}")
print(f"Correct predictions: {np.sum(y_pred == y_labels)}")
print(f"Overall accuracy: {np.sum(y_pred == y_labels) / len(y_labels):.4f}")

print("\nPer-class accuracy:")
for i, class_name in enumerate(class_names):
    class_mask = y_labels == i
    class_correct = np.sum((y_pred == i) & class_mask)
    class_total = np.sum(class_mask)
    class_acc = class_correct / class_total if class_total > 0 else 0
    print(f"  {class_name}: {class_acc:.4f} ({class_correct}/{class_total})")

print("\nVisualization complete!")
