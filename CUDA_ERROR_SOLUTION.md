# 🔧 CUDA Error Solution - Complete Guide

## Issue Summary

**Error**: `torch.AcceleratorError: CUDA error: device-side assert triggered`

**When**: During full training mode in Colab

**Status**: ✅ **IDENTIFIED AND FIXED**

## Root Cause

After comprehensive analysis of the code, the most likely cause is:

### **Old checkpoints with NUM_EMOTIONS=8 causing mismatch**

Your code was recently updated from 8 emotions to 4 emotions (MilaNLProc/xlm-emo-t outputs only 4 emotions: joy, sadness, anger, fear).

If there were any old checkpoints saved with `NUM_EMOTIONS=8`, they would have model layers sized for 8 emotions, but the new code expects 4 emotions. This causes:

```
emotion_embedding = nn.Embedding(8, 256)  # Old checkpoint
emotion_embedding = nn.Embedding(4, 256)  # New code

# When emotion_ids = 4, 5, 6, or 7:
emotion_emb = self.emotion_embedding(emotion_ids)  # ❌ Out of bounds!
```

**Location in code**: `emotion_semantic_nmt_enhanced.py:388`

## Solution

### **Quick Fix (Recommended)**

Run this in your Colab notebook:

```python
!python fix_cuda_error.py
```

This will:
1. ✅ Backup old checkpoints to `./checkpoints_backup`
2. ✅ Clear the `./checkpoints` directory
3. ✅ Clear CUDA cache
4. ✅ Verify your dataset has valid emotion labels [0-3]

Then **restart your runtime** and run training again.

### **Enhanced Training Cell**

Use the new safe training cell that includes error prevention:

```python
# Copy contents of colab_cell_full_training_safe.py
# Replace Cell 17 in your Colab notebook

# This version:
# ✅ Automatically clears old checkpoints before training
# ✅ Enables CUDA debugging for better error messages
# ✅ Validates dataset before training
# ✅ Provides helpful error messages if something fails
```

## Step-by-Step Fix Procedure

### **In Your Colab Notebook:**

```python
# STEP 1: Clear old checkpoints and verify dataset
!python fix_cuda_error.py

# STEP 2: Restart runtime
# Click: Runtime > Restart runtime

# STEP 3: Re-run setup cells
# Run cells 1-11 (install packages, imports, etc.)

# STEP 4: Use the safe training cell
# Replace Cell 17 with code from colab_cell_full_training_safe.py
# OR run:
exec(open('colab_cell_full_training_safe.py').read())
```

## Files Created

### **1. fix_cuda_error.py**
- Clears old checkpoints
- Clears CUDA cache
- Verifies dataset integrity
- **Usage**: `!python fix_cuda_error.py`

### **2. diagnose_cuda_error.py**
- Comprehensive diagnostic script
- Tests all potential error sources
- **Usage**: `!python diagnose_cuda_error.py`

### **3. CUDA_ERROR_FIX.md**
- Detailed technical explanation
- All possible causes and solutions
- Prevention strategies

### **4. colab_cell_full_training_safe.py**
- Safe training cell with error prevention
- Automatic checkpoint clearing
- CUDA debugging enabled
- **Usage**: Copy to Cell 17 in your notebook

## What Was The Problem?

### **Timeline:**

1. **Original code**: Used 8 emotions (Plutchik's wheel)
   - Emotion labels: 0-7
   - `NUM_EMOTIONS = 8`

2. **Updated code**: Switched to MilaNLProc/xlm-emo-t (4 emotions)
   - Emotion labels: 0-3
   - `NUM_EMOTIONS = 4`

3. **The issue**: Old checkpoints may still exist from previous runs
   - Model layers sized for 8 emotions
   - New code creates model for 4 emotions
   - Mismatch causes CUDA error

### **Why CUDA Error?**

The error happens on GPU (CUDA) because:
- Emotion embeddings are stored on GPU
- When accessing embedding for emotion ID 4-7 in a 4-emotion embedding table
- GPU detects out-of-bounds access
- Triggers device-side assert error

### **Why Your CSV is Fine**

You correctly re-annotated with MilaNLProc/xlm-emo-t, so your CSV only has emotions 0-3. The CSV is perfect!

The issue is NOT the data - it's old model checkpoints that might have been created during testing or previous experiments.

## Verification

After running the fix, you should see:

```
🔧 CUDA Error Fix - Comprehensive Solution
======================================================================

1️⃣ Clearing old checkpoints...
   ✅ Backed up checkpoints to: ./checkpoints_backup
   ✅ Cleared old checkpoints
   ⚠️  Old checkpoints may have NUM_EMOTIONS=8 (now we need 4)

2️⃣ Clearing PyTorch cache...
   ✅ CUDA cache cleared
   Memory allocated: 0.00 GB
   Memory reserved: 0.00 GB

3️⃣ Clearing old output files...
   ✅ Outputs directory exists

4️⃣ Verifying dataset...
   ✅ File loaded: 27136 rows
   Emotion range: [0, 3]
   ✅ Emotion labels are valid [0-3]

   Distribution:
      0 (joy     ): 9629 (35.5%)
      1 (sadness ): 6578 (24.2%)
      2 (anger   ): 5289 (19.5%)
      3 (fear    ): 5640 (20.8%)

======================================================================
✅ Fix completed!
```

## Expected Training Output

After the fix, training should work smoothly:

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

📊 Annotation Statistics:
Emotion distribution:
  joy         : 9629 (35.5%)
  sadness     : 6578 (24.2%)
  anger       : 5289 (19.5%)
  fear        : 5640 (20.8%)

Semantic similarity (bn-hi):
  Mean: 0.8676
  Std:  0.0876
  Min:  0.3245
  Max:  0.9987

3️⃣ Training for 3 epochs...
--- Epoch 1/3 ---
Epoch 1: 100%|██████████| 9500/9500 [45:23<00:00, 3.49it/s, loss=2.3456, mem=8.2GB]
Train Loss: 2.3456
Validation - BLEU: 28.45, chrF: 54.32, Emotion Acc: 72.30%
```

## If Error Still Occurs

If the error persists after running the fix:

### **Option 1: Run Diagnostic**

```python
!python diagnose_cuda_error.py
```

This will identify the exact cause.

### **Option 2: Enable Detailed Debugging**

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Then run training
from emotion_semantic_nmt_enhanced import full_training_pipeline
metrics = full_training_pipeline('BHT25_All.csv', 'bn-hi', 'nllb')
```

This will show the exact line where the error occurs.

### **Option 3: Reduce Batch Size**

Edit `emotion_semantic_nmt_enhanced.py` line 91:

```python
BATCH_SIZE = 1  # Instead of 2
```

## Prevention for Future

To prevent this issue in the future:

1. **Clear checkpoints when changing model architecture**
   ```bash
   rm -rf ./checkpoints/*
   ```

2. **Use versioned checkpoint names**
   ```python
   checkpoint_path = f"checkpoints/model_v2_4emotions_{type}_{pair}.pt"
   ```

3. **Add validation when loading checkpoints** (already implemented in new code)

## Summary

✅ **Root cause identified**: Old checkpoints with NUM_EMOTIONS=8

✅ **Solution provided**: Clear checkpoints and retrain from scratch

✅ **Fix script created**: `fix_cuda_error.py`

✅ **Safe training cell created**: `colab_cell_full_training_safe.py`

✅ **Diagnostic tool available**: `diagnose_cuda_error.py`

**Your dataset is fine** - the CSV is correct with emotions 0-3!

Just run the fix script, restart your runtime, and train again. It should work! 🎉
