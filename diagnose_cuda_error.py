#!/usr/bin/env python3
"""
Comprehensive CUDA Error Diagnostic
Checks all possible sources of device-side assert errors
"""

import pandas as pd
import numpy as np
import torch

print("🔍 COMPREHENSIVE CUDA ERROR DIAGNOSTIC")
print("="*80)

# 1. Check CSV file
print("\n1️⃣ Checking CSV file...")
try:
    df = pd.read_csv('BHT25_All_annotated.csv')
    print(f"   ✅ Loaded {len(df)} rows")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# 2. Check emotion labels
print("\n2️⃣ Checking emotion labels (must be 0-3)...")
emotion_bn_values = df['emotion_bn'].values
print(f"   Min: {emotion_bn_values.min()}")
print(f"   Max: {emotion_bn_values.max()}")
print(f"   Unique values: {sorted(np.unique(emotion_bn_values))}")

if emotion_bn_values.max() > 3:
    print(f"   ❌ ERROR: Found emotion labels > 3!")
    invalid_count = (emotion_bn_values > 3).sum()
    print(f"   {invalid_count} samples have invalid labels")
else:
    print(f"   ✅ All emotion labels in valid range [0, 3]")

# 3. Check for NaN values
print("\n3️⃣ Checking for NaN/null values...")
for col in ['emotion_bn', 'emotion_hi', 'emotion_te', 'semantic_bn_hi', 'semantic_bn_te']:
    if col in df.columns:
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col].replace([np.inf, -np.inf], np.nan)).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"   ❌ {col}: {nan_count} NaN, {inf_count} Inf values")
        else:
            print(f"   ✅ {col}: No NaN/Inf values")

# 4. Check data types
print("\n4️⃣ Checking data types...")
for col in ['emotion_bn', 'emotion_hi', 'emotion_te']:
    if col in df.columns:
        dtype = df[col].dtype
        print(f"   {col}: {dtype}")
        if not np.issubdtype(dtype, np.integer):
            print(f"   ⚠️  WARNING: Should be integer, got {dtype}")
            # Try to convert
            try:
                df[col] = df[col].astype(int)
                print(f"   ✅ Converted to int")
            except:
                print(f"   ❌ Cannot convert to int - has non-integer values!")

# 5. Check semantic scores range
print("\n5️⃣ Checking semantic similarity scores (should be 0-1)...")
for col in ['semantic_bn_hi', 'semantic_bn_te']:
    if col in df.columns:
        values = df[col].values
        print(f"   {col}:")
        print(f"      Min: {values.min():.4f}, Max: {values.max():.4f}")
        print(f"      Mean: {values.mean():.4f}, Std: {values.std():.4f}")
        if values.min() < 0 or values.max() > 1:
            print(f"      ⚠️  WARNING: Values outside [0, 1] range")

# 6. Check for duplicate rows
print("\n6️⃣ Checking for duplicates...")
dup_count = df.duplicated().sum()
if dup_count > 0:
    print(f"   ⚠️  Found {dup_count} duplicate rows")
else:
    print(f"   ✅ No duplicate rows")

# 7. Check text fields
print("\n7️⃣ Checking text fields...")
for lang in ['bn', 'hi', 'te']:
    if lang in df.columns:
        empty_count = (df[lang].str.len() < 3).sum()
        null_count = df[lang].isna().sum()
        if empty_count > 0 or null_count > 0:
            print(f"   ⚠️  {lang}: {empty_count} empty, {null_count} null")
        else:
            print(f"   ✅ {lang}: All valid")

# 8. Test with PyTorch
print("\n8️⃣ Testing with PyTorch tensors...")
try:
    # Simulate what the model does
    emotion_labels = torch.tensor(df['emotion_bn'].values[:100].astype(int))

    print(f"   Emotion tensor shape: {emotion_labels.shape}")
    print(f"   Emotion tensor dtype: {emotion_labels.dtype}")
    print(f"   Emotion tensor min/max: {emotion_labels.min()}/{emotion_labels.max()}")

    # Test embedding lookup (this is where CUDA error happens)
    num_emotions = 4
    embedding = torch.nn.Embedding(num_emotions, 256)

    try:
        embedded = embedding(emotion_labels)
        print(f"   ✅ Embedding lookup successful!")
        print(f"   Embedded shape: {embedded.shape}")
    except Exception as e:
        print(f"   ❌ Embedding lookup failed: {e}")

    # Test CrossEntropyLoss
    logits = torch.randn(10, 4)  # batch_size=10, num_classes=4
    labels = emotion_labels[:10]

    try:
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        print(f"   ✅ CrossEntropyLoss successful!")
        print(f"   Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"   ❌ CrossEntropyLoss failed: {e}")

except Exception as e:
    print(f"   ❌ PyTorch test failed: {e}")

# 9. Check for negative values
print("\n9️⃣ Checking for negative emotion labels...")
negative_count = (df['emotion_bn'] < 0).sum()
if negative_count > 0:
    print(f"   ❌ Found {negative_count} negative emotion labels!")
else:
    print(f"   ✅ No negative values")

# 10. Sample some problematic rows (if any)
print("\n🔟 Checking for problematic samples...")
problem_rows = df[
    (df['emotion_bn'] < 0) |
    (df['emotion_bn'] > 3) |
    (df['emotion_bn'].isna())
]

if len(problem_rows) > 0:
    print(f"   ❌ Found {len(problem_rows)} problematic rows:")
    print(problem_rows[['bn', 'hi', 'te', 'emotion_bn', 'semantic_bn_hi']].head(10))
else:
    print(f"   ✅ No problematic rows found")

# 11. Final verdict
print("\n" + "="*80)
print("📊 DIAGNOSTIC SUMMARY:")

issues = []
if emotion_bn_values.max() > 3:
    issues.append("Emotion labels > 3 found")
if (emotion_bn_values < 0).sum() > 0:
    issues.append("Negative emotion labels found")
if df['emotion_bn'].isna().sum() > 0:
    issues.append("NaN emotion labels found")
if not np.issubdtype(df['emotion_bn'].dtype, np.integer):
    issues.append("Emotion labels not integer type")

if len(issues) == 0:
    print("✅ CSV file looks CORRECT - emotion labels are all 0-3")
    print("\n💡 CUDA error is likely caused by something else:")
    print("   1. Batch size too large (try reducing from 2 to 1)")
    print("   2. Out of GPU memory (check with !nvidia-smi)")
    print("   3. Model architecture issue (check NUM_EMOTIONS=4 everywhere)")
    print("   4. Gradient issues (try CUDA_LAUNCH_BLOCKING=1 for detailed error)")
    print("\nNext steps:")
    print("   Run with: !CUDA_LAUNCH_BLOCKING=1 python ...")
    print("   This will show exact line where error occurs")
else:
    print("❌ Issues found:")
    for issue in issues:
        print(f"   - {issue}")
    print("\nFix these issues before training!")

print("="*80)
