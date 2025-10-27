#!/usr/bin/env python3
"""
Diagnostic script to check annotated CSV for invalid emotion labels
Run this to identify the problem before training
"""

import pandas as pd
import numpy as np

print("🔍 Checking BHT25_All_annotated.csv for issues...")
print("="*60)

try:
    df = pd.read_csv('BHT25_All_annotated.csv')
    print(f"✅ File loaded: {len(df)} rows")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

print(f"\nColumns: {df.columns.tolist()}")

# Check emotion labels
print("\n📊 Emotion Label Distribution (emotion_bn):")
emotion_counts = df['emotion_bn'].value_counts().sort_index()
emotion_names = ['joy', 'sadness', 'anger', 'fear', 'trust', 'disgust', 'surprise', 'anticipation']

for emotion_id in sorted(emotion_counts.index):
    count = emotion_counts[emotion_id]
    pct = count / len(df) * 100
    name = emotion_names[emotion_id] if emotion_id < 8 else 'UNKNOWN'

    if emotion_id > 3:
        status = "❌ INVALID (model expects 0-3 only!)"
    else:
        status = "✅ Valid"

    print(f"  {emotion_id}: {name:12s} - {count:5d} ({pct:5.1f}%) {status}")

# Check for out-of-range values
invalid_emotions = df[df['emotion_bn'] > 3]
if len(invalid_emotions) > 0:
    print(f"\n❌ PROBLEM FOUND!")
    print(f"   {len(invalid_emotions)} samples have emotion_bn > 3")
    print(f"   Model only supports emotions 0-3 (4 emotions)")
    print(f"\n   This will cause CUDA error: device-side assert triggered")
    print(f"\n🔧 SOLUTION:")
    print(f"   1. Re-annotate dataset with MilaNLProc/xlm-emo-t (outputs 0-3)")
    print(f"   2. OR map emotions 4-7 to 0-3:")
    print(f"      - trust (4) → joy (0)")
    print(f"      - disgust (5) → anger (2)")
    print(f"      - surprise (6) → joy (0)")
    print(f"      - anticipation (7) → joy (0)")
else:
    print(f"\n✅ All emotion labels are valid (0-3)")
    print(f"   Model should work correctly")

# Check semantic scores
print(f"\n📊 Semantic Similarity Ranges:")
if 'semantic_bn_hi' in df.columns:
    print(f"   bn-hi: {df['semantic_bn_hi'].min():.4f} to {df['semantic_bn_hi'].max():.4f}")
    print(f"          mean={df['semantic_bn_hi'].mean():.4f}, std={df['semantic_bn_hi'].std():.4f}")

if 'semantic_bn_te' in df.columns:
    print(f"   bn-te: {df['semantic_bn_te'].min():.4f} to {df['semantic_bn_te'].max():.4f}")
    print(f"          mean={df['semantic_bn_te'].mean():.4f}, std={df['semantic_bn_te'].std():.4f}")

# Check for NaN values
print(f"\n📊 Missing Values:")
for col in ['emotion_bn', 'semantic_bn_hi', 'semantic_bn_te']:
    if col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"   ❌ {col}: {nan_count} NaN values")
        else:
            print(f"   ✅ {col}: No NaN values")

print("\n" + "="*60)
print("Diagnostic complete!")
