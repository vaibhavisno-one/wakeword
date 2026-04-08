import os
import random
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

class WakeWordDataset(Dataset):
    def __init__(self, pos_dir, neg_dir, max_negatives=10000):
        self.files = []

        # Load positive files (only .wav files, skip directories)
        for f in os.listdir(pos_dir):
            path = os.path.join(pos_dir, f)
            if os.path.isfile(path) and f.endswith('.wav'):
                self.files.append((path, 1))

        # Load negative files (only .wav files, skip directories)
        neg_files = [f for f in os.listdir(neg_dir) 
                     if os.path.isfile(os.path.join(neg_dir, f)) and f.endswith('.wav')]
        random.shuffle(neg_files)
        neg_files = neg_files[:max_negatives]  # limit negatives to 10k

        for f in neg_files:
            self.files.append((os.path.join(neg_dir, f), 0))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]

        audio, sr = librosa.load(path, sr=16000)
        
        # Ensure audio is exactly 1 second (16000 samples)
        # Pad if too short, truncate if too long
        target_length = 16000
        if len(audio) < target_length:
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            # Truncate
            audio = audio[:target_length]

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=400,
            hop_length=160,
            n_mels=40
        )

        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        mel = torch.tensor(mel).unsqueeze(0).float()
        label = torch.tensor(label).float().unsqueeze(0)

        return mel, label