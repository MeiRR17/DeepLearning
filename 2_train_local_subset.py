import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set paths
BASE_DIR = Path("g:/My Drive/02_Academia/DataMining/S2/Deep Learning/Project")

# Load packaged data
print("Loading packaged data...")
X_data = np.load(BASE_DIR / "X_data_top5.npy")
y_labels = np.load(BASE_DIR / "y_labels_top5.npy")

with open(BASE_DIR / "classes_top5.json", 'r', encoding='utf-8') as f:
    classes_info = json.load(f)

class_names = classes_info['class_names']
num_classes = len(class_names)

print(f"Data shape: {X_data.shape}")
print(f"Labels shape: {y_labels.shape}")
print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")

# Add channel dimension for CNN (required for Conv2D)
X_data = X_data[..., np.newaxis]
print(f"Data shape with channel: {X_data.shape}")

# Split data: 80% train, 20% validation (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X_data, y_labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_labels
)

print(f"\nTrain set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# Build lightweight CNN model (optimized for 4GB VRAM)
def build_model(input_shape, num_classes):
    model = keras.Sequential([
        # First Conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = X_train.shape[1:]
model = build_model(input_shape, num_classes)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model
print("\nStarting training...")
batch_size = 16  # Small batch size for 4GB VRAM
epochs = 20

history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    verbose=1
)

# Save the trained model
model_path = BASE_DIR / "model_r1_top5.h5"
model.save(model_path)
print(f"\nModel saved to: {model_path}")

# Save training history for visualization
history_dict = {
    'loss': [float(x) for x in history.history['loss']],
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']]
}

with open(BASE_DIR / "training_history.json", 'w') as f:
    json.dump(history_dict, f, indent=2)

print("Training history saved to training_history.json")

# Print final metrics
print(f"\nFinal training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
