




import os
import librosa
import soundfile as sf

INPUT_DIR = "dataset/negative_raw"
OUTPUT_DIR = "dataset/negative"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_LEN = 16000  # 1 sec

count = 0

for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if not file.lower().endswith(".wav"):
            continue

        path = os.path.join(root, file)

        try:
            audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)

            # split into 1-sec chunks
            for i in range(0, len(audio), TARGET_LEN):
                chunk = audio[i:i+TARGET_LEN]

                if len(chunk) < TARGET_LEN:
                    continue

                # normalize
                chunk = librosa.util.normalize(chunk)

                out_path = os.path.join(OUTPUT_DIR, f"n_{count:05d}.wav")
                sf.write(out_path, chunk, TARGET_SR)

                count += 1

        except Exception as e:
            print(f"Skipped {file}: {e}")

print("Total negatives created:", count)