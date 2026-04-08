import librosa
import numpy as np
import soundfile as sf
import os

INPUT_DIR = "dataset/positive_raw"
OUTPUT_DIR = "dataset/positive"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 16000
WINDOW_SIZE = 16000  # 1 sec

count = 0

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".wav"):
        continue

    path = os.path.join(INPUT_DIR, file)
    audio, sr = librosa.load(path, sr=TARGET_SR)

    best_energy = 0
    best_chunk = None

    # Slide window
    for i in range(0, len(audio) - WINDOW_SIZE, 4000):  # step = 0.25 sec
        chunk = audio[i:i+WINDOW_SIZE]

        energy = np.sum(chunk**2)

        if energy > best_energy:
            best_energy = energy
            best_chunk = chunk

    if best_chunk is None:
        continue

    # Normalize
    best_chunk = librosa.util.normalize(best_chunk)

    sf.write(f"{OUTPUT_DIR}/p_{count:04d}.wav", best_chunk, TARGET_SR)
    count += 1

print("Processed:", count)