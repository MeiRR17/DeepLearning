import os
import shutil

src = r'G:\My Drive\02_Academia\DataMining\S2\Deep Learning\Project\data\R3\wav_files\temp_00001.wav'
dst = r'G:\My Drive\02_Academia\DataMining\S2\Deep Learning\Project\data\R3\wav_files\01_bereshit_03.wav'

if os.path.exists(src):
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(src, dst)
    print(f"Renamed to: {dst}")
else:
    print(f"Source file not found: {src}")
