# CUDA Error Fix Guide

## Problem
```
torch.AcceleratorError: CUDA error: device-side assert triggered
```

## Root Cause Analysis

After analyzing the code, the most likely causes are:

### **1. Old Checkpoints with NUM_EMOTIONS=8 (MOST LIKELY)**

The code was recently updated from 8 emotions to 4 emotions. If you have old checkpoints saved with `NUM_EMOTIONS=8`, they will have:
- `emotion_embedding` with 8 entries (0-7)
- `emotion_classifier` outputting 8 logits

When the new code (with `NUM_EMOTIONS=4`) creates an `emotion_embedding` with only 4 entries (0-3), but old model weights get loaded somehow, you'll get out-of-bounds errors.

**Location in code:**
```python
# emotion_semantic_nmt_enhanced.py:361
self.emotion_embedding = nn.Embedding(num_emotions, 256)

# emotion_semantic_nmt_enhanced.py:388
emotion_emb = self.emotion_embedding(emotion_ids)  # ← CUDA error here if emotion_ids >= 4
```

### **2. Mixed Precision Training Issues**

The training uses `autocast()` for mixed precision, which can cause issues with `CrossEntropyLoss` if:
- Logits have NaN or Inf values
- There's a type mismatch
- Labels are in wrong format

**Location in code:**
```python
# emotion_semantic_nmt_enhanced.py:954-967
with autocast():
    outputs = self.model(...)
    emotion_loss = F.cross_entropy(outputs['emotion_logits'], batch['emotion_label'])
```

### **3. Batch Collation or Data Loading Issues**

- Emotion labels not properly converted to tensors
- Batch tensors have wrong shape
- Data type mismatches

## Solutions

### **Solution 1: Clear Old Checkpoints (RECOMMENDED)**

```python
# Run the fix script
!python fix_cuda_error.py
```

This will:
1. Backup old checkpoints to `./checkpoints_backup`
2. Clear the `./checkpoints` directory
3. Clear CUDA cache
4. Verify dataset integrity

### **Solution 2: Add CUDA_LAUNCH_BLOCKING for Debugging**

If error persists, run with synchronous CUDA execution to get exact error location:

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Then run training
from emotion_semantic_nmt_enhanced import full_training_pipeline
full_training_pipeline('BHT25_All.csv', 'bn-hi', 'nllb')
```

This will show exactly which line causes the CUDA error.

### **Solution 3: Reduce Batch Size**

The current batch size is 2. Try reducing to 1:

```python
# Edit emotion_semantic_nmt_enhanced.py:91
BATCH_SIZE = 1  # Instead of 2
```

### **Solution 4: Add Explicit Tensor Validation**

Add this to the `train_step` method before computing loss:

```python
# After line 966 in emotion_semantic_nmt_enhanced.py
if outputs['emotion_logits'] is not None:
    # VALIDATION
    print(f"emotion_logits shape: {outputs['emotion_logits'].shape}")  # Should be [batch_size, 4]
    print(f"emotion_label shape: {batch['emotion_label'].shape}")      # Should be [batch_size]
    print(f"emotion_label values: {batch['emotion_label']}")           # Should be in [0, 3]
    print(f"emotion_label min/max: {batch['emotion_label'].min()}/{batch['emotion_label'].max()}")

    # Check for invalid values
    if batch['emotion_label'].max() >= 4:
        raise ValueError(f"Invalid emotion label: {batch['emotion_label'].max()} >= 4")
    if batch['emotion_label'].min() < 0:
        raise ValueError(f"Invalid emotion label: {batch['emotion_label'].min()} < 0")

    emotion_loss = F.cross_entropy(outputs['emotion_logits'], batch['emotion_label'])
    total_loss += self.config.BETA * emotion_loss
```

### **Solution 5: Move CrossEntropyLoss Outside Autocast**

Mixed precision can cause issues. Try moving the loss calculation outside autocast:

```python
# Modify train_step method (lines 954-980)
with autocast():
    outputs = self.model(...)

    # Calculate translation loss only
    translation_loss = outputs['loss']

# Calculate other losses OUTSIDE autocast (better numerical stability)
total_loss = self.config.ALPHA * translation_loss

if outputs['emotion_logits'] is not None:
    emotion_loss = F.cross_entropy(outputs['emotion_logits'].float(), batch['emotion_label'])
    total_loss += self.config.BETA * emotion_loss

if outputs['style_logits'] is not None:
    style_loss = F.cross_entropy(outputs['style_logits'].float(), batch['style_label'])
    total_loss += self.config.DELTA * style_loss

if outputs['semantic_similarity'] is not None:
    semantic_loss = F.mse_loss(outputs['semantic_similarity'].float(), batch['semantic_score'])
    total_loss += self.config.GAMMA * semantic_loss
```

## Step-by-Step Fix Procedure

### **In Colab:**

```python
# 1. Clear old checkpoints and cache
!python fix_cuda_error.py

# 2. Restart runtime
# Go to: Runtime > Restart runtime

# 3. Re-run setup cells (install packages, import libraries)

# 4. Enable CUDA debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 5. Run training
from emotion_semantic_nmt_enhanced import full_training_pipeline

metrics = full_training_pipeline(
    csv_path='BHT25_All.csv',  # Will auto-load BHT25_All_annotated.csv
    translation_pair='bn-hi',
    model_type='nllb'
)
```

### **If Error Still Occurs:**

```python
# Run comprehensive diagnostic
!python diagnose_cuda_error.py
```

This will test:
- ✅ CSV file loading
- ✅ Emotion label range (must be 0-3)
- ✅ NaN/Inf values
- ✅ Data types
- ✅ PyTorch tensor operations
- ✅ Embedding lookups
- ✅ CrossEntropyLoss compatibility

## Expected Output (Success)

```
🚀 Starting Full Training Pipeline
   Translation: bn-hi
   Model: nllb
   Epochs: 3
============================================================

1️⃣ Creating model...
   Parameters: 648,012,345

2️⃣ Loading annotated dataset...
✅ Using ANNOTATED dataset: BHT25_All_annotated.csv
Dataset shape: (27136, 6)

📊 Annotation Statistics:
Emotion distribution:
  joy         : 9629 (50.7%)
  sadness     : 4606 (24.2%)
  anger       : 3710 (19.5%)
  fear        : 1055 ( 5.6%)

3️⃣ Training for 3 epochs...
--- Epoch 1/3 ---
Epoch 1: 100%|██████████| 9500/9500 [45:23<00:00, 3.49it/s, loss=2.3456, mem=8.2GB]
Train Loss: 2.3456
Validation - BLEU: 28.45, chrF: 54.32, Emotion Acc: 72.30%
```

## Prevention

To prevent this error in the future:

1. **Always clear checkpoints when changing NUM_EMOTIONS**
   ```bash
   rm -rf ./checkpoints/*
   ```

2. **Version your checkpoints**
   ```python
   checkpoint_path = f"checkpoints/model_v2_4emotions_{model_type}_{pair}.pt"
   ```

3. **Add metadata to checkpoints**
   ```python
   torch.save({
       'model_state_dict': model.state_dict(),
       'config': config,
       'num_emotions': config.NUM_EMOTIONS,  # ← Add this
       'version': '2.0',  # ← Add version
       'metrics': metrics
   }, checkpoint_path)
   ```

4. **Validate checkpoints before loading**
   ```python
   if os.path.exists(checkpoint_path):
       checkpoint = torch.load(checkpoint_path)
       if checkpoint.get('num_emotions', 8) != config.NUM_EMOTIONS:
           print(f"⚠️ Checkpoint has {checkpoint['num_emotions']} emotions, expected {config.NUM_EMOTIONS}")
           print(f"   Skipping checkpoint load, training from scratch")
       else:
           model.load_state_dict(checkpoint['model_state_dict'])
   ```

## Summary

**Most Likely Cause**: Old checkpoints with `NUM_EMOTIONS=8` interfering with new code expecting `NUM_EMOTIONS=4`.

**Quick Fix**:
```bash
python fix_cuda_error.py
# Then restart runtime and re-run training
```

**If that doesn't work**:
```bash
CUDA_LAUNCH_BLOCKING=1 python diagnose_cuda_error.py
```

This will pinpoint the exact issue!
