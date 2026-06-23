import os
import sys
import time
from huggingface_hub import snapshot_download

# Disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add ctranslate2 package dir to PATH so its bundled cuDNN 8 DLLs are found
import ctranslate2 as _ct2
_ct2_dir = os.path.dirname(_ct2.__file__)
os.add_dll_directory(_ct2_dir)
os.environ["PATH"] = _ct2_dir + os.pathsep + os.environ.get("PATH", "")
print(f"ctranslate2 {_ct2.__version__} | DLL dir: {_ct2_dir}")

from faster_whisper import WhisperModel

# Paths
wav_dir = r"G:\My Drive\02_Academia\DataMining\S2\Deep Learning\Project\data\R3\wav"
vtt_dir = r"G:\My Drive\02_Academia\DataMining\S2\Deep Learning\Project\data\R3\vtt"
model_local_dir = r"C:\whisper_models\large-v3"

# Create vtt directory if it doesn't exist
os.makedirs(vtt_dir, exist_ok=True)
os.makedirs(model_local_dir, exist_ok=True)

# Download model directly into a flat local folder (no symlinks needed)
print("Downloading/loading Whisper large-v3 model to local directory...")
try:
    snapshot_download(
        repo_id="Systran/faster-whisper-large-v3",
        local_dir=model_local_dir,
    )
    model_path = model_local_dir
    print(f"Model ready at: {model_path}")
except Exception as e:
    print(f"Failed to download large-v3: {e}")
    model_local_dir = r"C:\whisper_models\large-v2"
    os.makedirs(model_local_dir, exist_ok=True)
    snapshot_download(
        repo_id="Systran/faster-whisper-large-v2",
        local_dir=model_local_dir,
    )
    model_path = model_local_dir
    print(f"Fallback model ready at: {model_path}")

# Initialize model with INT8 quantization for 4GB VRAM
print("\nLoading Whisper model with INT8 quantization on CUDA...")
for compute_type in ["int8_float16", "int8", "float16", "auto"]:
    try:
        model = WhisperModel(model_path, device="cuda", compute_type=compute_type)
        print(f"Model loaded on GPU with compute_type='{compute_type}'!")
        break
    except Exception as e:
        print(f"  compute_type='{compute_type}' failed: {e}")
else:
    print("All GPU options failed, falling back to CPU with int8...")
    model = WhisperModel(model_path, device="cpu", compute_type="int8")
    print("Model loaded on CPU with int8.")

# Get all wav files
wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
wav_files.sort()

print(f"\nFound {len(wav_files)} WAV files to transcribe")
print("=" * 60)

total_start_time = time.time()

for i, wav_file in enumerate(wav_files, 1):
    wav_path = os.path.join(wav_dir, wav_file)
    vtt_file = wav_file.replace('.wav', '.vtt')
    vtt_path = os.path.join(vtt_dir, vtt_file)
    
    print(f"\n[{i}/{len(wav_files)}] Processing: {wav_file}")
    print(f"File size: {os.path.getsize(wav_path) / (1024*1024):.2f} MB")
    
    start_time = time.time()
    
    try:
        # Transcribe with Hebrew language
        segments, info = model.transcribe(
            wav_path, 
            language="he", 
            word_timestamps=False
        )
        
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        print(f"Duration: {info.duration:.2f} seconds")
        
        # Write VTT file
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for segment in segments:
                # Format timestamps as VTT: 00:00:01.000 --> 00:00:05.000
                start = segment.start
                end = segment.end
                
                start_time_str = time.strftime('%H:%M:%S', time.gmtime(start)) + f".{int((start % 1) * 1000):03d}"
                end_time_str = time.strftime('%H:%M:%S', time.gmtime(end)) + f".{int((end % 1) * 1000):03d}"
                
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{segment.text.strip()}\n\n")
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds")
        print(f"VTT saved to: {vtt_path}")
        
    except Exception as e:
        print(f"Error transcribing {wav_file}: {e}")
        continue

total_elapsed = time.time() - total_start_time
print("\n" + "=" * 60)
print(f"All transcriptions completed in {total_elapsed:.2f} seconds")
print(f"Average time per file: {total_elapsed / len(wav_files):.2f} seconds")
