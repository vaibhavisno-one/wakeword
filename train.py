import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
import os
from datetime import datetime

from data_loader import WakeWordDataset
from model import WakeWordCNN

# Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
PATIENCE = 15  # Increased from 10 - allow more time to improve
MAX_NEGATIVES = 20000  # Increased from 15000 - more diverse negatives

print("=" * 60)
print("Wake Word Detection Training - 'Atlas'")
print("=" * 60)

# Load dataset
print("\n[1/6] Loading dataset...")
dataset = WakeWordDataset(
    "dataset/positive_aug",
    "dataset/negative",
    max_negatives=MAX_NEGATIVES
)

print(f"Total samples: {len(dataset)}")
pos_count = sum(1 for _, label in dataset.files if label == 1)
neg_count = sum(1 for _, label in dataset.files if label == 0)
print(f"  Positive samples: {pos_count}")
print(f"  Negative samples: {neg_count}")
print(f"  Class ratio: 1:{neg_count/pos_count:.1f}")

# 80/20 train-val split
print("\n[2/6] Splitting dataset (80% train, 20% validation)...")
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"  Training samples: {train_size}")
print(f"  Validation samples: {val_size}")

# Calculate class weights for balanced training
train_labels = [dataset.files[i][1] for i in train_dataset.indices]
train_pos = sum(train_labels)
train_neg = len(train_labels) - train_pos
weight_ratio = train_neg / train_pos

print(f"\n[3/6] Setting up weighted sampling...")
print(f"  Weight ratio (neg/pos): {weight_ratio:.1f}")

# Create weighted sampler for balanced batches
class_weights = [weight_ratio if l == 1 else 1.0 for l in train_labels]
sampler = WeightedRandomSampler(class_weights, num_samples=len(class_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
print("\n[4/6] Initializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

model = WakeWordCNN().to(device)

# Loss function with pos_weight for class imbalance
pos_weight = torch.tensor([weight_ratio]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# Training setup
best_val_recall = 0.0
best_val_f1 = 0.0
patience_counter = 0
best_model_path = "saved_models/best_model.pth"
os.makedirs("saved_models", exist_ok=True)

print("\n[5/6] Starting training...")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Early stopping patience: {PATIENCE}")
print("=" * 60)

for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train()
    total_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        preds = model(x).squeeze(1)
        y = y.squeeze(1)

        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Calculate training accuracy
        with torch.no_grad():
            predicted = (torch.sigmoid(preds) > 0.5).float()
            train_correct += (predicted == y).sum().item()
            train_total += y.size(0)

    avg_loss = total_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    # --- Validation Phase ---
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze(1)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.squeeze(1).cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    val_acc = 100 * np.mean(all_preds == all_labels)
    
    # Calculate per-class metrics
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        wake_word_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        wake_word_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    else:
        wake_word_recall = 0
        wake_word_precision = 0

    # Print epoch summary
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Acc: {val_acc:.2f}% | Val F1: {f1:.4f}")
    print(f"  Wake Word Recall: {wake_word_recall:.2%} | Precision: {wake_word_precision:.2%}")
    
    # Learning rate scheduling based on F1 score (balances recall and precision)
    scheduler.step(f1)
    
    # Save best model based on F1 score (not just recall)
    # F1 balances both detecting wake words AND avoiding false positives
    if f1 > best_val_f1 and wake_word_recall >= 0.80:  # Must have 80%+ recall
        best_val_recall = wake_word_recall
        best_val_f1 = f1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'recall': wake_word_recall,
            'f1': f1,
            'precision': wake_word_precision
        }, best_model_path)
        print(f"  ✓ New best model saved! (F1: {f1:.4f}, Recall: {wake_word_recall:.2%})")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{PATIENCE})")
    
    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
        break

print("\n" + "=" * 60)
print("[6/6] Training Complete!")
print("=" * 60)

# Load best model for final evaluation
checkpoint = torch.load(best_model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\nBest Model Performance:")
print(f"  Wake Word Recall: {checkpoint['recall']:.2%}")
print(f"  Wake Word Precision: {checkpoint['precision']:.2%}")
print(f"  F1 Score: {checkpoint['f1']:.4f}")

# Final validation evaluation
print("\n" + "=" * 60)
print("Final Validation Report:")
print("=" * 60)

all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        out = torch.sigmoid(model(x).squeeze(1))
        preds = (out > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.squeeze(1).cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=["negative", "positive"], digits=4))

# Save final model
final_model_path = "saved_models/model.pth"
torch.save(model.state_dict(), final_model_path)
print(f"\n✓ Models saved:")
print(f"  - {best_model_path}")
print(f"  - {final_model_path}")
print("\n✓ Training completed successfully!")