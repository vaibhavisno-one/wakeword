import torch
import librosa
import numpy as np
import sounddevice as sd
from collections import deque
import time
from model import WakeWordCNN


# Configuration
SAMPLE_RATE = 16000
WINDOW_SIZE = 16000  # 1 second
CHUNK_SIZE = 8000    # 0.5 seconds
N_MELS = 40
N_FFT = 400
HOP_LENGTH = 160
THRESHOLD = 0.50  # Lowered to make wake-word detection easier
SMOOTHING_SIZE = 3
COOLDOWN_TIME = 2.0


class WakeWordDetector:
    def __init__(self, model_path):
        # Load model
        self.model = WakeWordCNN()
        
        # Handle both checkpoint dict and state_dict formats
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded best model (Recall: {checkpoint['recall']:.2%})")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"✓ Loaded model from {model_path}")
        
        self.model.eval()
        
        # Audio buffer (sliding window)
        self.audio_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=SMOOTHING_SIZE)
        
        # Cooldown tracking
        self.last_trigger_time = 0
        self.prediction_count = 0
        self.max_confidence = 0.0
        self.trigger_count = 0
        self.last_candidate_time = 0
        
        print("Wake word detector initialized")
        print(f"Model loaded from: {model_path}")
        print(f"Threshold: {THRESHOLD}")
        print("Listening... say 'atlas' and watch the confidence value.")
    
    def preprocess_audio(self, audio):
        """Preprocess audio exactly like training"""
        # Normalize audio
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()
        
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        
        # Convert to dB
        mel = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize mean/std
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        
        return mel
    
    def predict(self, audio):
        """Run inference on audio"""
        mel = self.preprocess_audio(audio)
        
        # Convert to tensor (add batch and channel dimensions)
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()
        
        # Run inference
        with torch.no_grad():
            logits = self.model(mel_tensor)
            confidence = torch.sigmoid(logits).item()
        
        return confidence
    
    def update_buffer(self, new_audio):
        """Update sliding window with new audio chunk"""
        chunk_len = len(new_audio)
        
        # Shift old audio left and append new audio
        self.audio_buffer = np.roll(self.audio_buffer, -chunk_len)
        self.audio_buffer[-chunk_len:] = new_audio
    
    def should_trigger(self, confidence):
        """Determine if wake word should trigger"""
        # Add to smoothing buffer
        self.prediction_buffer.append(confidence)
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_trigger_time < COOLDOWN_TIME:
            return False
        
        # Need enough predictions for smoothing
        if len(self.prediction_buffer) < SMOOTHING_SIZE:
            return False
        
        # Simple trigger: just check if current confidence is high
        # The high precision (99.05%) means we can trust individual predictions
        if confidence > THRESHOLD:
            self.last_trigger_time = current_time
            self.prediction_buffer.clear()  # Clear buffer after trigger
            return True
        
        return False

    def format_status(self, confidence):
        smoothed = sum(self.prediction_buffer) / len(self.prediction_buffer) if self.prediction_buffer else 0.0
        peak = max(self.prediction_buffer) if self.prediction_buffer else 0.0

        if confidence > THRESHOLD and len(self.prediction_buffer) >= SMOOTHING_SIZE:
            state = "DETECTED"
        elif confidence > THRESHOLD:
            state = "candidate"
        else:
            state = "listening"

        return f"conf={confidence:.4f} avg={smoothed:.4f} peak={peak:.4f} threshold={THRESHOLD:.2f} state={state}"
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice stream"""
        if status:
            print(f"Status: {status}")
        
        # Convert to mono and flatten
        audio_chunk = indata[:, 0].copy().flatten()
        
        # Update sliding window
        self.update_buffer(audio_chunk)
        
        # Run prediction on current window
        confidence = self.predict(self.audio_buffer)
        self.prediction_count += 1
        self.max_confidence = max(self.max_confidence, confidence)

        # Show live confidence so it is obvious whether the model is responding
        print(f"\r{self.format_status(confidence)}", end="", flush=True)
        
        # Check if wake word detected
        if self.should_trigger(confidence):
            self.trigger_count += 1
            print(f"\n🔊 WAKE WORD DETECTED! (confidence: {confidence:.3f})")
        elif confidence > THRESHOLD:
            current_time = time.time()
            if current_time - self.last_candidate_time > 0.5:
                self.last_candidate_time = current_time
                recent = ", ".join(f"{c:.2f}" for c in list(self.prediction_buffer)[-5:])
                print(f"\nCANDIDATE: confidence crossed threshold. recent=[{recent}]")
    
    def start(self):
        """Start continuous listening"""
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            callback=self.audio_callback,
            dtype=np.float32
        ):
            print("Press Ctrl+C to stop")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\nStopped listening")
                print(f"Total predictions: {self.prediction_count}")
                print(f"Max confidence seen: {self.max_confidence:.4f}")
                print(f"Triggers: {self.trigger_count}")


if __name__ == "__main__":
    # Use best_model.pth by default (highest accuracy)
    import os
    model_path = "saved_models/best_model.pth"
    if not os.path.exists(model_path):
        model_path = "saved_models/model.pth"  # Fallback to regular model
    
    detector = WakeWordDetector(model_path)
    detector.start()