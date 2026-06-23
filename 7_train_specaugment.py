import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("=" * 60)
print("PHASE 1: FINAL SPEC-AUGMENT + CLASS-WEIGHTS + REGULARIZATION")
print("=" * 60)

PROJECT_DIR = r'G:\My Drive\02_Academia\DataMining\S2\Deep Learning\Project'
os.chdir(PROJECT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected, using CPU")

print("\n[1] Loading data...")
X = np.load(os.path.join(PROJECT_DIR, 'X_data_top5.npy'))
y = np.load(os.path.join(PROJECT_DIR, 'y_labels_top5.npy'))

with open(os.path.join(PROJECT_DIR, 'classes_top5.json'), 'r') as f:
    class_data = json.load(f)
    class_names = class_data['class_names']

print(f"Original X shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Classes: {class_names}")

X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
print(f"PyTorch X shape: {X.shape}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("\n[2] Computing class weights...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
print("Class distribution and weights:")
for i, class_name in enumerate(class_names):
    count = np.sum(y_train == i)
    print(f"  {class_name}: {count} samples, weight: {class_weights[i]:.4f}")

print("\n[3] SpecAugment helper...")

def spec_augment(x, freq_mask_param=15, time_mask_param=15,
                 num_freq_masks=None, num_time_masks=None, replace_value=0.0):
    """
    Apply SpecAugment to a batch of spectrograms.
    x: (B, 1, H, W) tensor. H = frequency bins, W = time frames.
    Masks contiguous vertical (time) and horizontal (frequency) bands.
    """
    B, _, H, W = x.shape
    y = x.clone()

    for i in range(B):
        n_freq = num_freq_masks if num_freq_masks is not None else np.random.randint(1, 3)
        n_time = num_time_masks if num_time_masks is not None else np.random.randint(1, 3)
        # Frequency masking (horizontal bands)
        for _ in range(n_freq):
            f = np.random.randint(0, freq_mask_param + 1)
            f0 = np.random.randint(0, H - f + 1) if H > f else 0
            if f > 0 and H > f:
                y[i, :, f0:f0 + f, :] = replace_value

        # Time masking (vertical bands)
        for _ in range(n_time):
            t = np.random.randint(0, time_mask_param + 1)
            t0 = np.random.randint(0, W - t + 1) if W > t else 0
            if t > 0 and W > t:
                y[i, :, :, t0:t0 + t] = replace_value
    return y

print("\n[4] Building regularized CNN model...")
class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 10, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CNN(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)

print(model)
print(f"Model moved to: {device}")
print(f"Optimizer: Adam (lr=0.0005, weight_decay=1e-3)")
print(f"SpecAugment: freq_mask=15, time_mask=15, 1-2 random masks each, applied per batch")

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, val_loss, epoch, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True

early_stopping = EarlyStopping(patience=7)

print("\n[5] Training for up to 40 epochs with early stopping...")
num_epochs = 40
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = spec_augment(inputs, freq_mask_param=15, time_mask_param=15)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss /= train_total
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    early_stopping(val_loss, epoch + 1, model)
    if early_stopping.early_stop:
        print(f"\nEarly stopping triggered at epoch {epoch+1}. Best validation loss at epoch {early_stopping.best_epoch}.")
        break

print("\n[6] Restoring best model weights...")
model.load_state_dict(early_stopping.best_model_state)
best_epoch = early_stopping.best_epoch
print(f"Best model restored from epoch {best_epoch} with val_loss={early_stopping.best_loss:.4f}")

print("\n[7] Generating visualizations and metrics...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_accs) + 1), train_accs, label='Training Accuracy', linewidth=2, marker='o')
plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy', linewidth=2, marker='o')
plt.title('Final Model Accuracy (SpecAugment + Regularization)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', linewidth=2, marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', linewidth=2, marker='o')
plt.title('Final Model Loss (SpecAugment + Regularization)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('specaugment_accuracy_loss.png', dpi=300, bbox_inches='tight')
print("Saved: specaugment_accuracy_loss.png")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Final Confusion Matrix (SpecAugment + Regularization)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('specaugment_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: specaugment_confusion_matrix.png")

report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
with open('specaugment_metrics_report.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("SPEC-AUGMENT PYTORCH TRAINING METRICS REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Device: {device}\n")
    if device.type == 'cuda':
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
    f.write(f"Model Architecture: CNN with BatchNorm + Dropout2d (PyTorch)\n")
    f.write(f"Optimizer: Adam (lr=0.0005, weight_decay=1e-3)\n")
    f.write(f"Early Stopping: patience=7 epochs\n")
    f.write(f"Epochs Run: {len(train_accs)}\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Class Weights Applied: Yes\n")
    f.write(f"SpecAugment Applied: Yes\n\n")
    f.write("Class Weights Used:\n")
    for i, (class_name, weight) in enumerate(zip(class_names, class_weights)):
        f.write(f"  {class_name}: {weight:.4f}\n")
    f.write("\n" + "=" * 60 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)
    f.write("\n" + "=" * 60 + "\n")
    f.write("PER-CLASS VALIDATION ACCURACY\n")
    f.write("=" * 60 + "\n")
    for i, class_name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        f.write(f"{class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)\n")
print("Saved: specaugment_metrics_report.txt")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Epochs run: {len(train_accs)}")
print(f"Best epoch (restored): {best_epoch}")
print(f"Final Training Accuracy (best model): {train_accs[best_epoch - 1]:.4f}")
print(f"Final Validation Accuracy (best model): {val_accs[best_epoch - 1]:.4f}")
print(f"Final Training Loss (best model): {train_losses[best_epoch - 1]:.4f}")
print(f"Final Validation Loss (best model): {val_losses[best_epoch - 1]:.4f}")
print("\nPer-Class Validation Accuracy:")
for i, class_name in enumerate(class_names):
    class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"  {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")

torch.save(model.state_dict(), 'model_r1_final_presentation.pth')
print("\nModel saved as: model_r1_final_presentation.pth")
