# Large Files - Data & Models

This document describes the large files that are excluded from Git due to their size. These files must be generated locally using the provided scripts.

## 📁 Dataset Files (Excluded from Git)

### Mel-Spectrogram Datasets
- **Location**: `Data_Prep/dataset/`, `Data_Prep/dataset_full/`, `Data_Prep/dataset_test/`
- **Format**: `.npy` files (Mel-spectrograms)
- **Size**: ~400MB total
- **Generation**: Run `Data_Prep/2_extract_features.py`
- **Description**: Mel-spectrogram representations of audio segments for each cantillation mark

### Processed Training Data
- **Location**: `X_data_top5.npy`, `y_labels_top5.npy`
- **Format**: NumPy arrays
- **Size**: ~400MB
- **Generation**: Run `Data_Prep/package_dataset.py`
- **Description**: Packaged training data for top 5 cantillation classes

### Aligned Dataset JSON
- **Location**: `Data_Prep/aligned_dataset.json`
- **Format**: JSON
- **Size**: ~3MB
- **Generation**: Run alignment script (part of data pipeline)
- **Description**: Aligned audio-text segments with timestamps and cantillation labels

## 🎵 Audio Files (Excluded from Git)

### Raw Audio Recordings
- **Location**: `data/R1/wav/`, `data/R2/wav/`, `data/R3/wav/`
- **Format**: `.wav` files
- **Size**: Several GB
- **Source**: Torah reading recordings
- **Description**: Original audio files for each reader (R1, R2, R3)

### Transcription Files
- **Location**: `data/R1/vtt/`, `data/R2/vtt/`, `data/R3/vtt/`
- **Format**: `.vtt` files
- **Size**: ~50MB total
- **Generation**: Run `Data_Prep/6_transcribe_new_readers.py`
- **Description**: Whisper transcriptions with timestamps

## 🤖 Model Files (Excluded from Git)

### Trained PyTorch Model
- **Location**: `model_r1_top5_balanced_pytorch.pth`
- **Format**: PyTorch state dict
- **Size**: ~20MB
- **Generation**: Run `5_train_regularized.py`
- **Description**: Trained CNN model for top 5 cantillation classification

### Keras Model (Legacy)
- **Location**: `model_r1_top5.h5`
- **Format**: Keras HDF5
- **Size**: ~60MB
- **Generation**: Run `4_train_balanced.py`
- **Description**: Legacy Keras model (superseded by PyTorch version)

## 🔄 How to Generate These Files

### Step 1: Transcribe Audio
```bash
cd Data_Prep
python 6_transcribe_new_readers.py
```
This generates `.vtt` transcription files for R2 and R3 readers.

### Step 2: Extract Features
```bash
cd Data_Prep
python 2_extract_features.py
```
This converts audio segments to Mel-spectrograms and saves them as `.npy` files.

### Step 3: Package Dataset
```bash
cd Data_Prep
python package_dataset.py
```
This packages the spectrograms into training arrays.

### Step 4: Train Model
```bash
python 5_train_regularized.py
```
This trains the CNN model and saves it as `.pth`.

## 📊 File Sizes Summary

| File Type | Approximate Size | Count |
|-----------|-----------------|-------|
| Audio WAV | ~50MB each | 100+ |
| VTT Transcriptions | ~500KB each | 100+ |
| Mel-Spectrograms (.npy) | ~50KB each | 7,000+ |
| Training Arrays | ~400MB total | 2 |
| Model (.pth) | ~20MB | 1 |

## 💡 Why These Files Are Excluded

1. **Git LIM**: Git has a 100MB file size limit
2. **Repository Size**: Including these would make the repository several GB
3. **Reproducibility**: These files can be regenerated from source code
4. **Privacy**: Audio recordings may contain personal voice data

## 📥 Alternative Distribution

For sharing the actual data and models, consider:
- **Google Drive**: Share audio and dataset files
- **Hugging Face**: Upload model and dataset
- **Zenodo**: Academic data repository with DOI
- **GitHub Releases**: Upload trained models as release assets
