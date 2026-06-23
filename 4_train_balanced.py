import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("PHASE 1: BALANCED TRAINING WITH CLASS WEIGHTS")
print("=" * 60)

# 1. Data Loading
print("\n[1] Loading data...")
X = np.load('X_data_top5.npy')
y = np.load('y_labels_top5.npy')

# Load class names
with open('classes_top5.json', 'r') as f:
    class_data = json.load(f)
    class_names = class_data['class_names']

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Classes: {class_names}")

# Split into 80/20 train/validation (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")

# 2. Compute Class Weights
print("\n[2] Computing class weights...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Create a dictionary mapping class indices to weights
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print("\nClass distribution in training set:")
for i, class_name in enumerate(class_names):
    count = np.sum(y_train == i)
    print(f"  {class_name}: {count} samples (weight: {class_weight_dict[i]:.4f})")

print("\nComputed class weights:")
for i, (class_name, weight) in enumerate(zip(class_names, class_weights)):
    print(f"  Class {i} ({class_name}): {weight:.4f}")

# 3. Model Architecture with BatchNormalization
print("\n[3] Building model with BatchNormalization...")
model = keras.Sequential([
    layers.Input(shape=(X.shape[1], X.shape[2], 1)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4. Training with Class Weights
print("\n[4] Training model with class weights for 25 epochs...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=1
)

# 5. Visualizations & Metrics
print("\n[5] Generating visualizations and metrics...")

# 5a. Accuracy/Loss Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy (Balanced Training)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss (Balanced Training)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('balanced_accuracy_loss.png', dpi=300, bbox_inches='tight')
print("Saved: balanced_accuracy_loss.png")

# 5b. Confusion Matrix
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_val, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix (Balanced Training)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('balanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: balanced_confusion_matrix.png")

# 5c. Classification Report
report = classification_report(y_val, y_pred_classes, target_names=class_names, digits=4)

with open('balanced_metrics_report.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("BALANCED TRAINING METRICS REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model Architecture: CNN with BatchNormalization\n")
    f.write(f"Optimizer: Adam (lr=0.0005)\n")
    f.write(f"Epochs: 25\n")
    f.write(f"Class Weights Applied: Yes\n\n")
    f.write("Class Weights Used:\n")
    for i, (class_name, weight) in enumerate(zip(class_names, class_weights)):
        f.write(f"  {class_name}: {weight:.4f}\n")
    f.write("\n" + "=" * 60 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)
    f.write("\n" + "=" * 60 + "\n")
    f.write("PER-CLASS ACCURACY\n")
    f.write("=" * 60 + "\n")
    for i, class_name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        f.write(f"{class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)\n")

print("Saved: balanced_metrics_report.txt")

# Print summary
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print("\nPer-Class Validation Accuracy:")
for i, class_name in enumerate(class_names):
    class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"  {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")

# Save the model
model.save('model_r1_top5_balanced.h5')
print("\nModel saved as: model_r1_top5_balanced.h5")
