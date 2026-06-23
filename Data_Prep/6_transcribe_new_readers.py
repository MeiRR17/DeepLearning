import os
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download

# Disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from faster_whisper import WhisperModel

# Project and data paths
PROJECT_DIR = r"G:\My Drive\02_Academia\DataMining\S2\Deep Learning\Project"
DATA_DIR = os.path.join(PROJECT_DIR, "data")

READERS = {
    "R3": os.path.join(DATA_DIR, "R3"),
    "R2": os.path.join(DATA_DIR, "R2"),
}

MODEL_LOCAL_DIR = r"C:\whisper_models\large-v3"


def format_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to VTT timestamp HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    # Handle rounding edge case where millis == 1000
    if millis == 1000:
        secs += 1
        millis = 0
    if secs == 60:
        minutes += 1
        secs = 0
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def transcribe_reader(reader_name: str, reader_dir: str, model: WhisperModel) -> dict:
    """Transcribe all WAV files for a single reader and write VTT files."""
    wav_dir = os.path.join(reader_dir, "wav")
    vtt_dir = os.path.join(reader_dir, "vtt")
    
    os.makedirs(vtt_dir, exist_ok=True)
    
    wav_files = sorted([f for f in os.listdir(wav_dir) if f.lower().endswith('.wav')])
    
    print(f"\n{'='*60}")
    print(f"Processing {reader_name}: {len(wav_files)} WAV files found")
    print(f"WAV source: {wav_dir}")
    print(f"VTT target: {vtt_dir}")
    print(f"{'='*60}")
    
    results = {
        "reader": reader_name,
        "total": len(wav_files),
        "success": 0,
        "failed": 0,
        "skipped": 0,
    }
    
    for i, wav_file in enumerate(wav_files, 1):
        wav_path = os.path.join(wav_dir, wav_file)
        vtt_file = wav_file.replace('.wav', '.vtt').replace('.WAV', '.vtt')
        vtt_path = os.path.join(vtt_dir, vtt_file)
        
        file_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
        
        print(f"\n[{i}/{len(wav_files)}] {reader_name} - {wav_file} ({file_size_mb:.2f} MB)")
        
        # Skip empty files
        if file_size_mb < 0.01:
            print(f"  WARNING: File is nearly empty, skipping")
            results["skipped"] += 1
            continue
        
        # Skip already-transcribed files (resume capability)
        if os.path.exists(vtt_path) and os.path.getsize(vtt_path) > 0:
            print(f"  VTT already exists, skipping: {vtt_path}")
            results["skipped"] += 1
            continue
        
        start_time = time.time()
        
        try:
            # Transcribe with Hebrew language and word timestamps enabled
            segments, info = model.transcribe(
                wav_path,
                language="he",
                word_timestamps=True,
                vad_filter=True,
                condition_on_previous_text=True,
            )
            
            print(f"  Detected language: {info.language} (p={info.language_probability:.2f}), duration: {info.duration:.1f}s")
            
            # Collect all segments
            segment_list = list(segments)
            
            if not segment_list:
                print(f"  WARNING: No speech detected, writing empty VTT")
                with open(vtt_path, 'w', encoding='utf-8') as f:
                    f.write("WEBVTT\n\n")
                results["success"] += 1
                continue
            
            # Write VTT file with segment-level cues (word timestamps improve boundary accuracy)
            with open(vtt_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                
                for segment in segment_list:
                    start_str = format_vtt_timestamp(segment.start)
                    end_str = format_vtt_timestamp(segment.end)
                    text = segment.text.strip()
                    
                    if not text:
                        continue
                    
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{text}\n\n")
            
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.2f}s -> {vtt_path}")
            results["success"] += 1
            
        except KeyboardInterrupt:
            print(f"\n  Interrupted by user")
            raise
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            results["failed"] += 1
            # Continue with next file
            continue
    
    return results


def main():
    print("=" * 60)
    print("FASTER-WHISPER TRANSCRIPTION FOR NEW READERS (R2 + R3)")
    print("=" * 60)
    
    # Ensure model directory exists
    os.makedirs(MODEL_LOCAL_DIR, exist_ok=True)
    
    # Download/load model
    print("\nLoading/downloading Whisper large-v3 model...")
    try:
        snapshot_download(
            repo_id="Systran/faster-whisper-large-v3",
            local_dir=MODEL_LOCAL_DIR,
        )
        model_path = MODEL_LOCAL_DIR
        print(f"Model ready at: {model_path}")
    except Exception as e:
        print(f"Failed to download large-v3: {e}")
        fallback_dir = r"C:\whisper_models\large-v2"
        os.makedirs(fallback_dir, exist_ok=True)
        snapshot_download(
            repo_id="Systran/faster-whisper-large-v2",
            local_dir=fallback_dir,
        )
        model_path = fallback_dir
        print(f"Fallback model ready at: {model_path}")
    
    # Force CPU mode for stability (avoids NVIDIA driver / CUDA / cuDNN issues)
    print("\nInitializing model on CPU with INT8 quantization...")
    print("CPU mode selected for reliable transcription across all readers.")
    model = WhisperModel(model_path, device="cpu", compute_type="int8")
    print("Model loaded successfully on CPU with int8.")
    
    # Process each reader
    total_start = time.time()
    all_results = []
    
    for reader_name, reader_dir in READERS.items():
        if not os.path.isdir(os.path.join(reader_dir, "wav")):
            print(f"\nWARNING: {reader_dir}\\wav does not exist, skipping {reader_name}")
            continue
        
        results = transcribe_reader(reader_name, reader_dir, model)
        all_results.append(results)
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 60)
    print("TRANSCRIPTION SUMMARY")
    print("=" * 60)
    total_files = sum(r["total"] for r in all_results)
    total_success = sum(r["success"] for r in all_results)
    total_failed = sum(r["failed"] for r in all_results)
    total_skipped = sum(r["skipped"] for r in all_results)
    
    for r in all_results:
        print(f"{r['reader']}: {r['success']}/{r['total']} succeeded, {r['failed']} failed, {r['skipped']} skipped")
    
    print(f"\nTotal time: {total_elapsed:.2f}s for {total_files} files")
    if total_files > 0:
        print(f"Average per file: {total_elapsed / total_files:.2f}s")
    print(f"Successful: {total_success}, Failed: {total_failed}, Skipped: {total_skipped}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTranscription interrupted by user.")
        sys.exit(1)
