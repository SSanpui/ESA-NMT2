"""
QUICK START - Copy this entire file into a Colab cell and run
This sets up everything from scratch with automatic backups
"""

# ==============================================================================
# CELL 1: Complete Setup (Run this first)
# ==============================================================================

print("🚀 ESA-NMT Complete Setup - With Automatic Backups")
print("="*70)

# Step 1: Mount Google Drive
print("\n1️⃣ Mounting Google Drive...")
from google.colab import drive
import os

drive.mount('/content/drive')

# Create project directory in Drive
project_dir = '/content/drive/MyDrive/ESA_NMT_Project'
os.makedirs(project_dir, exist_ok=True)
print(f"✅ Drive mounted: {project_dir}")

# Step 2: Clone repository
print("\n2️⃣ Cloning repository...")
if os.path.exists('ESA-NMT'):
    import shutil
    shutil.rmtree('ESA-NMT')
    print("   Removed old directory")

import subprocess
subprocess.run(['git', 'clone', 'https://github.com/SSanpui/ESA-NMT.git'], check=True)
os.chdir('ESA-NMT')
print("✅ Repository cloned")

# Checkout correct branch
print("\n3️⃣ Checking out branch with CUDA fixes...")
subprocess.run(['git', 'checkout', 'claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj'], check=True)
result = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True)
print(f"✅ Branch: {result.stdout.strip()}")

# Step 3: Install packages
print("\n4️⃣ Installing dependencies...")
subprocess.run(['pip', 'install', '-q', 'torch', 'transformers', 'sentencepiece',
                'sacrebleu', 'rouge-score', 'bert-score', 'sentence-transformers',
                'accelerate', 'sacremoses', 'datasets', 'tokenizers', 'protobuf',
                'pandas', 'numpy', 'matplotlib', 'seaborn', 'tqdm'], check=True)
print("✅ Dependencies installed")

# Step 4: Check for dataset
print("\n5️⃣ Checking dataset...")
if os.path.exists('BHT25_All.csv'):
    import pandas as pd
    df = pd.read_csv('BHT25_All.csv')
    print(f"✅ Dataset found: {len(df)} rows, {os.path.getsize('BHT25_All.csv')/1024**2:.1f} MB")
else:
    print("⚠️  BHT25_All.csv not found - you need to upload it")

# Step 5: Check for annotated CSV backup in Drive
print("\n6️⃣ Checking for annotated CSV backup...")
drive_annotated = '/content/drive/MyDrive/ESA_NMT_Project/BHT25_All_annotated.csv'
local_annotated = 'BHT25_All_annotated.csv'

if os.path.exists(drive_annotated):
    print("🎉 Found backup in Google Drive!")
    import shutil
    shutil.copy(drive_annotated, local_annotated)

    df = pd.read_csv(local_annotated)
    print(f"✅ Restored: {len(df)} rows")

    if 'emotion_bn' in df.columns:
        emotion_values = df['emotion_bn'].values
        print(f"   Emotion range: [{emotion_values.min()}, {emotion_values.max()}]")

        if emotion_values.max() <= 3:
            print("✅ Valid dataset (4 emotions)")
            print("\n🎉 READY TO TRAIN! Skip annotation, go directly to training.")
            NEED_ANNOTATION = False
        else:
            print("⚠️  Has 8 emotions, need re-annotation")
            NEED_ANNOTATION = True
    else:
        NEED_ANNOTATION = True
else:
    print("❌ No backup found in Drive")
    NEED_ANNOTATION = True

print("\n" + "="*70)
if NEED_ANNOTATION:
    print("⏭️  NEXT: Run Cell 2 (Annotation - 3 hours)")
else:
    print("⏭️  NEXT: Skip to Cell 3 (Training)")
print("="*70)

# ==============================================================================
# CELL 2: Annotation (ONLY if needed - Run if Cell 1 says "NEXT: Run Cell 2")
# ==============================================================================

print("⏰ Starting annotation... This takes ~3 hours")
print("="*70)

import subprocess
import os

# Run annotation
subprocess.run(['python', 'annotate_dataset.py'], check=True)

# Verify and backup
if os.path.exists('BHT25_All_annotated.csv'):
    import pandas as pd
    import shutil

    df = pd.read_csv('BHT25_All_annotated.csv')
    print(f"\n✅ Annotation completed: {len(df)} rows")

    # IMMEDIATE backup to Drive
    drive_backup = '/content/drive/MyDrive/ESA_NMT_Project/BHT25_All_annotated.csv'
    shutil.copy('BHT25_All_annotated.csv', drive_backup)
    print(f"💾 BACKED UP to Drive: {drive_backup}")

    # Statistics
    emotion_values = df['emotion_bn'].values
    emotion_names = ['joy', 'sadness', 'anger', 'fear']
    print("\n📊 Emotion Distribution:")
    for i in range(4):
        count = (emotion_values == i).sum()
        pct = count / len(emotion_values) * 100
        print(f"   {i} ({emotion_names[i]:8s}): {count:5d} ({pct:5.1f}%)")

    print("\n✅ Your 3-hour work is safe in Google Drive!")
    print("⏭️  NEXT: Run Cell 3 (Training)")
else:
    print("❌ Annotation failed!")

# ==============================================================================
# CELL 3: Full Training with CUDA Fix (Run this to train)
# ==============================================================================

print("🚀 Full Training Pipeline with CUDA Fix")
print("="*70)

import os
import torch
import subprocess

# Apply CUDA fix first
print("\n1️⃣ Applying CUDA error fix...")
subprocess.run(['python', 'fix_cuda_error.py'], check=True)
print("✅ CUDA fix applied")

# Enable CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Configuration
TRANSLATION_PAIR = "bn-hi"  # Change to "bn-te" for Bengali-Telugu
MODEL_TYPE = "nllb"

print(f"\n2️⃣ Configuration:")
print(f"   Translation: {TRANSLATION_PAIR}")
print(f"   Model: {MODEL_TYPE}")

# Run training
print(f"\n3️⃣ Starting training...")
print("="*70)

from emotion_semantic_nmt_enhanced import full_training_pipeline, ComprehensiveEvaluator

try:
    metrics = full_training_pipeline(
        csv_path='BHT25_All.csv',
        translation_pair=TRANSLATION_PAIR,
        model_type=MODEL_TYPE
    )

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)

    # Save results to Drive
    import json
    import shutil

    # Save metrics
    results_path = f'/content/drive/MyDrive/ESA_NMT_Project/training_results_{MODEL_TYPE}_{TRANSLATION_PAIR}.json'
    with open(results_path, 'w') as f:
        json.dump(ComprehensiveEvaluator.convert_to_json_serializable(metrics), f, indent=2)
    print(f"\n💾 Results: {results_path}")

    # Save checkpoint
    checkpoint_local = f'./checkpoints/final_model_{MODEL_TYPE}_{TRANSLATION_PAIR}.pt'
    checkpoint_drive = f'/content/drive/MyDrive/ESA_NMT_Project/final_model_{MODEL_TYPE}_{TRANSLATION_PAIR}.pt'

    if os.path.exists(checkpoint_local):
        shutil.copy(checkpoint_local, checkpoint_drive)
        print(f"💾 Model: {checkpoint_drive}")

    # Backup all outputs
    import shutil
    if os.path.exists('./outputs'):
        shutil.copytree('./outputs',
                       '/content/drive/MyDrive/ESA_NMT_Project/outputs',
                       dirs_exist_ok=True)
        print(f"💾 Outputs: /content/drive/MyDrive/ESA_NMT_Project/outputs/")

    print("\n🎉 All results backed up to Google Drive!")
    print("   Your work is safe even if session disconnects.")

except RuntimeError as e:
    print("\n❌ CUDA Error detected!")
    print(f"Error: {e}")
    print("\nRun diagnostic:")
    subprocess.run(['python', 'diagnose_cuda_error.py'])

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ Training pipeline completed!")
print("="*70)
