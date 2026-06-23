import wave
import os

path = r'G:\My Drive\02_Academia\DataMining\S2\Deep Learning\Project\data\R3\wav_files'
files = ['01_bereshit_01.wav', '01_bereshit_02.wav', '01_bereshit_03.wav']

for f in files:
    filepath = os.path.join(path, f)
    if os.path.exists(filepath):
        with wave.open(filepath, 'rb') as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            print(f'{f}:')
            print(f'  Channels: {channels}')
            print(f'  Sample Rate: {sample_rate} Hz')
            print(f'  Size: {os.path.getsize(filepath)} bytes')
            print()
