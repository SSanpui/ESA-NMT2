"""
Safe Full Training Cell for Colab - With Error Prevention
Replace Cell 17 in ESA_NMT_Colab.ipynb with this code
"""

# ============================================================================
# CELL: Full Training Mode (SAFE VERSION with Error Prevention)
# ============================================================================

import os
import torch
import shutil

print("🛡️ Full Training Mode - Safe Version with Error Prevention")
print("="*70)

# ----------------------------------------------------------------------------
# STEP 1: Clear Old Checkpoints (Prevent NUM_EMOTIONS mismatch)
# ----------------------------------------------------------------------------

print("\n1️⃣ Clearing old checkpoints...")
checkpoint_dir = "./checkpoints"

if os.path.exists(checkpoint_dir):
    # Backup
    backup_dir = "./checkpoints_backup"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(checkpoint_dir, backup_dir)
    print(f"   ✅ Backed up to: {backup_dir}")

    # Clear
    shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    print(f"   ✅ Cleared old checkpoints")
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"   ℹ️  No old checkpoints")

# ----------------------------------------------------------------------------
# STEP 2: Clear CUDA Cache
# ----------------------------------------------------------------------------

print("\n2️⃣ Clearing CUDA cache...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"   ✅ CUDA cache cleared")
    print(f"   Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
else:
    print(f"   ⚠️  CUDA not available")

# ----------------------------------------------------------------------------
# STEP 3: Enable CUDA Debugging (Get exact error location)
# ----------------------------------------------------------------------------

print("\n3️⃣ Enabling CUDA debugging...")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print(f"   ✅ CUDA_LAUNCH_BLOCKING=1 (synchronous execution)")
print(f"   This helps identify exact error location")

# ----------------------------------------------------------------------------
# STEP 4: Verify Dataset
# ----------------------------------------------------------------------------

print("\n4️⃣ Verifying dataset...")
import pandas as pd

csv_path = 'BHT25_All.csv'
annotated_path = 'BHT25_All_annotated.csv'

if os.path.exists(annotated_path):
    df = pd.read_csv(annotated_path)
    emotion_values = df['emotion_bn'].values

    print(f"   ✅ Dataset loaded: {len(df)} rows")
    print(f"   Emotion range: [{emotion_values.min()}, {emotion_values.max()}]")

    if emotion_values.max() > 3 or emotion_values.min() < 0:
        print(f"   ❌ ERROR: Invalid emotion labels!")
        print(f"   Run: !python fix_emotion_labels.py")
        raise ValueError("Dataset has invalid emotion labels")
    else:
        print(f"   ✅ All emotion labels valid [0-3]")
else:
    print(f"   ⚠️  Annotated file not found!")
    print(f"   Run: !python annotate_dataset.py")
    raise FileNotFoundError(f"Missing: {annotated_path}")

# ----------------------------------------------------------------------------
# STEP 5: Configuration
# ----------------------------------------------------------------------------

print("\n5️⃣ Configuration...")
TRANSLATION_PAIR = "bn-hi"  # Change to 'bn-te' for Bengali-Telugu
MODEL_TYPE = "nllb"         # 'nllb' or 'indictrans2'

print(f"   Translation pair: {TRANSLATION_PAIR}")
print(f"   Model type: {MODEL_TYPE}")

# ----------------------------------------------------------------------------
# STEP 6: Run Training Pipeline
# ----------------------------------------------------------------------------

print("\n6️⃣ Starting training pipeline...")
print("="*70)

from emotion_semantic_nmt_enhanced import full_training_pipeline

try:
    metrics = full_training_pipeline(
        csv_path=csv_path,           # Will auto-load annotated version
        translation_pair=TRANSLATION_PAIR,
        model_type=MODEL_TYPE
    )

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)

    print("\n📊 Final Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key:20s}: {value:.4f}")
        else:
            print(f"   {key:20s}: {value}")

    print("\n📁 Outputs:")
    print(f"   Model: ./checkpoints/final_model_{MODEL_TYPE}_{TRANSLATION_PAIR}.pt")
    print(f"   Results: ./outputs/full_training_results_{MODEL_TYPE}_{TRANSLATION_PAIR}.json")

except RuntimeError as e:
    print("\n" + "="*70)
    print("❌ CUDA ERROR DETECTED!")
    print("="*70)
    print(f"\nError: {e}")

    print("\n🔍 Diagnostic Steps:")
    print("   1. Run: !python diagnose_cuda_error.py")
    print("   2. Check error message above for exact line number")
    print("   3. Verify emotion labels in dataset")
    print("   4. Try reducing BATCH_SIZE from 2 to 1")

    print("\n💡 Common Causes:")
    print("   • Emotion labels > 3 (should be 0-3)")
    print("   • NaN or Inf values in data")
    print("   • GPU memory exhausted")
    print("   • Mixed precision issues")

    print("\n🔧 Quick Fixes:")
    print("   • Restart runtime (Runtime > Restart runtime)")
    print("   • Run this cell again (checkpoints already cleared)")
    print("   • Reduce batch size in emotion_semantic_nmt_enhanced.py:91")

    raise  # Re-raise to show full traceback

except Exception as e:
    print("\n" + "="*70)
    print("❌ UNEXPECTED ERROR!")
    print("="*70)
    print(f"\nError: {e}")
    raise

# ----------------------------------------------------------------------------
# STEP 7: Memory Cleanup
# ----------------------------------------------------------------------------

print("\n7️⃣ Cleaning up...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"   ✅ CUDA cache cleared")

print("\n" + "="*70)
print("🎉 All done!")
print("="*70)
