import os
import librosa
import numpy as np
import soundfile as sf

INPUT_DIR = "dataset/positive"
OUTPUT_DIR = "dataset/positive_aug"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_LEN = 16000

def add_noise(audio):
    noise = np.random.randn(len(audio))
    return audio + 0.002 * noise

def volume_scale(audio):
    return audio * np.random.uniform(0.8, 1.2)

def slight_pitch(audio, sr):
    return librosa.effects.pitch_shift(
        audio, sr=sr, n_steps=np.random.uniform(-1, 1)
    )

count = 0

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".wav"):
        continue

    path = os.path.join(INPUT_DIR, file)
    audio, sr = librosa.load(path, sr=TARGET_SR)

    # Save original
    audio = librosa.util.fix_length(audio, size=TARGET_LEN)
    sf.write(f"{OUTPUT_DIR}/p_{count:05d}.wav", audio, sr)
    count += 1

    # Generate 3 safe variants
    for _ in range(3):
        aug = audio.copy()

        if np.random.rand() < 0.7:
            aug = add_noise(aug)

        if np.random.rand() < 0.5:
            aug = volume_scale(aug)

        if np.random.rand() < 0.3:
            aug = slight_pitch(aug, sr)

        aug = librosa.util.normalize(aug)
        aug = librosa.util.fix_length(aug, size=TARGET_LEN)

        sf.write(f"{OUTPUT_DIR}/p_{count:05d}.wav", aug, sr)
        count += 1

print("Total augmented samples:", count)