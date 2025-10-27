"""
FASTEST START - With Pre-uploaded Annotated CSV
Copy this into Colab cells - No 3-hour annotation needed!
"""

# ==============================================================================
# CELL 1: Complete Setup with Your Backed-up CSV
# ==============================================================================

print("🚀 ESA-NMT Fast Setup - Using Your Backed-up Annotated CSV")
print("="*70)

import os
import subprocess
import shutil

# Step 1: Mount Google Drive (for saving outputs)
print("\n1️⃣ Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')
project_dir = '/content/drive/MyDrive/ESA_NMT_Project'
os.makedirs(project_dir, exist_ok=True)
print(f"✅ Drive mounted: {project_dir}")

# Step 2: Clone repository
print("\n2️⃣ Cloning repository...")
if os.path.exists('ESA-NMT'):
    shutil.rmtree('ESA-NMT')

subprocess.run(['git', 'clone', 'https://github.com/SSanpui/ESA-NMT.git'], check=True)
os.chdir('ESA-NMT')

# Checkout branch with all CUDA fixes
subprocess.run(['git', 'checkout', 'claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj'], check=True)
result = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True)
print(f"✅ Branch: {result.stdout.strip()}")

# Step 3: Install packages
print("\n3️⃣ Installing dependencies...")
subprocess.run(['pip', 'install', '-q', 'torch', 'transformers', 'sentencepiece',
                'sacrebleu', 'rouge-score', 'bert-score', 'sentence-transformers',
                'accelerate', 'sacremoses', 'datasets', 'tokenizers', 'protobuf',
                'pandas', 'numpy', 'matplotlib', 'seaborn', 'tqdm'], check=True)
print("✅ Dependencies installed")

# Step 4: Download your annotated CSV from GitHub
print("\n4️⃣ Downloading your annotated CSV from GitHub...")
import requests
import pandas as pd

# Try different possible locations where you might have uploaded it
possible_urls = [
    # Main branch
    "https://raw.githubusercontent.com/SSanpui/ESA-NMT/main/BHT25_All_annotated.csv",
    # Current branch
    "https://raw.githubusercontent.com/SSanpui/ESA-NMT/claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj/BHT25_All_annotated.csv",
    # Master branch (if exists)
    "https://raw.githubusercontent.com/SSanpui/ESA-NMT/master/BHT25_All_annotated.csv",
]

annotated_downloaded = False
for url in possible_urls:
    try:
        print(f"   Trying: {url.split('/')[-2]}...")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open('BHT25_All_annotated.csv', 'wb') as f:
                f.write(response.content)

            # Verify it's valid
            df = pd.read_csv('BHT25_All_annotated.csv')
            print(f"   ✅ Downloaded: {len(df)} rows")

            # Check columns
            required_cols = ['bn', 'hi', 'te', 'emotion_bn', 'semantic_bn_hi', 'semantic_bn_te']
            missing = [col for col in required_cols if col not in df.columns]

            if missing:
                print(f"   ⚠️  Missing columns: {missing}")
                continue

            # Check emotion range
            emotion_values = df['emotion_bn'].values
            print(f"   Emotion range: [{emotion_values.min()}, {emotion_values.max()}]")

            if emotion_values.max() <= 3 and emotion_values.min() >= 0:
                print("   ✅ Dataset is valid (4 emotions)!")
                annotated_downloaded = True

                # Backup to Drive immediately
                shutil.copy('BHT25_All_annotated.csv',
                           f'{project_dir}/BHT25_All_annotated.csv')
                print(f"   💾 Backed up to Drive")
                break
            else:
                print(f"   ⚠️  Invalid emotion range (expected 0-3, got {emotion_values.min()}-{emotion_values.max()})")

    except Exception as e:
        print(f"   ❌ {e}")
        continue

if not annotated_downloaded:
    print("\n⚠️  Could not download annotated CSV from GitHub.")
    print("   Please provide the exact GitHub URL or branch where you uploaded it.")
    print("\n   Alternative: Upload it manually:")
    print("   ```python")
    print("   from google.colab import files")
    print("   uploaded = files.upload()  # Select BHT25_All_annotated.csv")
    print("   ```")

# Step 5: Check original dataset
print("\n5️⃣ Checking original dataset...")
if os.path.exists('BHT25_All.csv'):
    df = pd.read_csv('BHT25_All.csv')
    print(f"✅ Original dataset: {len(df)} rows")
else:
    print("⚠️  BHT25_All.csv not found")

print("\n" + "="*70)
if annotated_downloaded:
    print("✅ SETUP COMPLETE - Ready to train!")
    print("⏭️  NEXT: Run Cell 2 (Training)")
else:
    print("⚠️  Setup incomplete - need annotated CSV")
    print("⏭️  NEXT: Upload annotated CSV manually")
print("="*70)


# ==============================================================================
# CELL 1B: Manual Upload (Only if Cell 1 couldn't download from GitHub)
# ==============================================================================

print("📤 Manual Upload - If auto-download failed")
print("="*70)

from google.colab import files
import pandas as pd
import shutil
import os

print("Please select BHT25_All_annotated.csv to upload:")
uploaded = files.upload()

# Verify uploaded file
if 'BHT25_All_annotated.csv' in uploaded:
    df = pd.read_csv('BHT25_All_annotated.csv')
    print(f"\n✅ Uploaded: {len(df)} rows")

    # Verify
    emotion_values = df['emotion_bn'].values
    print(f"Emotion range: [{emotion_values.min()}, {emotion_values.max()}]")

    if emotion_values.max() <= 3:
        print("✅ Valid dataset!")

        # Backup to Drive
        project_dir = '/content/drive/MyDrive/ESA_NMT_Project'
        shutil.copy('BHT25_All_annotated.csv', f'{project_dir}/BHT25_All_annotated.csv')
        print(f"💾 Backed up to: {project_dir}")

        print("\n⏭️  NEXT: Run Cell 2 (Training)")
    else:
        print("⚠️  Invalid emotion range! Should be 0-3")
else:
    print("❌ File not uploaded")


# ==============================================================================
# CELL 2: Training with CUDA Fix
# ==============================================================================

print("🚀 Full Training with CUDA Fix")
print("="*70)

import os
import subprocess

# Verify annotated CSV exists
if not os.path.exists('BHT25_All_annotated.csv'):
    print("❌ BHT25_All_annotated.csv not found!")
    print("Please run Cell 1 or Cell 1B first")
    raise FileNotFoundError("Annotated CSV required")

# Apply CUDA fix
print("\n1️⃣ Applying CUDA error fix...")
subprocess.run(['python', 'fix_cuda_error.py'], check=True)
print("✅ CUDA fix applied (old checkpoints cleared)")

# Enable debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Configuration
TRANSLATION_PAIR = "bn-hi"  # Change to "bn-te" for Bengali-Telugu
MODEL_TYPE = "nllb"

print(f"\n2️⃣ Configuration:")
print(f"   Translation: {TRANSLATION_PAIR}")
print(f"   Model: {MODEL_TYPE}")
print(f"   Epochs: 3")

# Run training
print(f"\n3️⃣ Starting training...")
print("="*70)

from emotion_semantic_nmt_enhanced import full_training_pipeline, ComprehensiveEvaluator

try:
    metrics = full_training_pipeline(
        csv_path='BHT25_All.csv',  # Will auto-load BHT25_All_annotated.csv
        translation_pair=TRANSLATION_PAIR,
        model_type=MODEL_TYPE
    )

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)

    # Save everything to Drive
    import json
    import shutil

    project_dir = '/content/drive/MyDrive/ESA_NMT_Project'

    # Save metrics
    results_path = f'{project_dir}/training_results_{MODEL_TYPE}_{TRANSLATION_PAIR}.json'
    with open(results_path, 'w') as f:
        json.dump(ComprehensiveEvaluator.convert_to_json_serializable(metrics), f, indent=2)
    print(f"\n💾 Results: {results_path}")

    # Save model checkpoint
    checkpoint_local = f'./checkpoints/final_model_{MODEL_TYPE}_{TRANSLATION_PAIR}.pt'
    checkpoint_drive = f'{project_dir}/final_model_{MODEL_TYPE}_{TRANSLATION_PAIR}.pt'

    if os.path.exists(checkpoint_local):
        shutil.copy(checkpoint_local, checkpoint_drive)
        file_size = os.path.getsize(checkpoint_drive) / 1024**2
        print(f"💾 Model: {checkpoint_drive} ({file_size:.1f} MB)")

    # Backup all outputs
    if os.path.exists('./outputs'):
        shutil.copytree('./outputs', f'{project_dir}/outputs', dirs_exist_ok=True)
        print(f"💾 Outputs: {project_dir}/outputs/")

    print("\n🎉 SUCCESS! All results saved to Google Drive:")
    print(f"   📂 {project_dir}/")
    print("   Your work is permanently saved!")

    # Print metrics summary
    print("\n📊 Final Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key:20s}: {value:.4f}")

except RuntimeError as e:
    if "CUDA" in str(e):
        print("\n❌ CUDA Error detected!")
        print(f"Error: {e}")
        print("\n🔍 Running diagnostic...")
        subprocess.run(['python', 'diagnose_cuda_error.py'])
        print("\nCheck diagnostic output above for the exact cause.")
    else:
        raise

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ Pipeline completed!")
print("="*70)
