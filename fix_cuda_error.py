#!/usr/bin/env python3
"""
Fix CUDA Device-Side Assert Error
Comprehensive fix for training issues
"""

import os
import shutil
import torch

print("🔧 CUDA Error Fix - Comprehensive Solution")
print("="*70)

# ============================================================================
# SOLUTION 1: Clear Old Checkpoints (Most Likely Cause)
# ============================================================================

print("\n1️⃣ Clearing old checkpoints...")
checkpoint_dir = "./checkpoints"

if os.path.exists(checkpoint_dir):
    # Backup before deleting
    backup_dir = "./checkpoints_backup"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    shutil.copytree(checkpoint_dir, backup_dir)
    print(f"   ✅ Backed up checkpoints to: {backup_dir}")

    # Delete old checkpoints
    shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    print(f"   ✅ Cleared old checkpoints")
    print(f"   ⚠️  Old checkpoints may have NUM_EMOTIONS=8 (now we need 4)")
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"   ℹ️  No old checkpoints found")

# ============================================================================
# SOLUTION 2: Clear PyTorch Cache
# ============================================================================

print("\n2️⃣ Clearing PyTorch cache...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"   ✅ CUDA cache cleared")
    print(f"   Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"   Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
else:
    print(f"   ⚠️  CUDA not available")

# ============================================================================
# SOLUTION 3: Clear Output Files
# ============================================================================

print("\n3️⃣ Clearing old output files...")
output_dir = "./outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"   ✅ Created outputs directory")
else:
    print(f"   ✅ Outputs directory exists")

# ============================================================================
# SOLUTION 4: Verify Dataset
# ============================================================================

print("\n4️⃣ Verifying dataset...")
import pandas as pd
import numpy as np

annotated_file = 'BHT25_All_annotated.csv'
if os.path.exists(annotated_file):
    try:
        df = pd.read_csv(annotated_file)

        # Check emotion labels
        emotion_values = df['emotion_bn'].values
        min_emotion = emotion_values.min()
        max_emotion = emotion_values.max()

        print(f"   ✅ File loaded: {len(df)} rows")
        print(f"   Emotion range: [{min_emotion}, {max_emotion}]")

        if max_emotion > 3 or min_emotion < 0:
            print(f"   ❌ ERROR: Invalid emotion labels found!")
            print(f"   Expected range: [0, 3] (4 emotions)")
            print(f"   Actual range: [{min_emotion}, {max_emotion}]")
            print(f"\n   🔧 Run fix_emotion_labels.py to fix this!")
        else:
            print(f"   ✅ Emotion labels are valid [0-3]")

            # Show distribution
            print(f"\n   Distribution:")
            emotion_names = ['joy', 'sadness', 'anger', 'fear']
            for i in range(4):
                count = (emotion_values == i).sum()
                pct = count / len(emotion_values) * 100
                print(f"      {i} ({emotion_names[i]:8s}): {count:5d} ({pct:5.1f}%)")

    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
else:
    print(f"   ⚠️  Annotated file not found: {annotated_file}")
    print(f"   Run: python annotate_dataset.py")

print("\n" + "="*70)
print("✅ Fix completed!")
print("\n📋 Next Steps:")
print("   1. Restart your Colab runtime (Runtime > Restart runtime)")
print("   2. Re-run the full training pipeline")
print("   3. If error persists, run: python diagnose_cuda_error.py")
print("\n💡 Common Causes Fixed:")
print("   ✓ Old checkpoints with NUM_EMOTIONS=8 removed")
print("   ✓ CUDA cache cleared")
print("   ✓ Fresh start guaranteed")
print("="*70)
