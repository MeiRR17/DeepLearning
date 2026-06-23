import yt_dlp
import os

# Configuration for audio extraction
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
    'postprocessor_args': [
        '-ar', '22050',  # Set sampling rate to 22050 Hz
        '-ac', '1',      # Convert to mono
    ],
    'outtmpl': 'G:/My Drive/02_Academia/DataMining/S2/Deep Learning/Project/data/R3/wav_files/temp_%(autonumber)s.%(ext)s',
    'quiet': False,
    'ffmpeg_location': r'C:\Users\Avnet\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin',
    'concurrent_fragment_downloads': 1,
}

# YouTube URLs to download
urls = [
    'https://www.youtube.com/watch?v=XFaoCR0heXU',
]

# Output filenames
output_filenames = [
    '01_bereshit_02.wav',
]

# Download each video
for i, url in enumerate(urls):
    print(f"\nDownloading video {i+1}/3: {url}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Rename the downloaded file to the desired name
        downloaded_files = os.listdir('G:/My Drive/02_Academia/DataMining/S2/Deep Learning/Project/data/R3/wav_files')
        wav_files = [f for f in downloaded_files if f.endswith('.wav')]
        
        if wav_files:
            latest_file = wav_files[-1]  # Get the most recently downloaded file
            old_path = os.path.join('G:/My Drive/02_Academia/DataMining/S2/Deep Learning/Project/data/R3/wav_files', latest_file)
            new_path = os.path.join('G:/My Drive/02_Academia/DataMining/S2/Deep Learning/Project/data/R3/wav_files', output_filenames[i])
            
            if os.path.exists(new_path):
                os.remove(new_path)
            
            os.rename(old_path, new_path)
            print(f"Renamed to: {output_filenames[i]}")
        else:
            print(f"Error: No WAV file found for {url}")
            
    except Exception as e:
        print(f"Error downloading {url}: {e}")

print("\nAll downloads completed!")
