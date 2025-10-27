# Complete Fresh Setup Guide for Colab (After Session Loss)

## What Happened

**"Restart session"** in Colab disconnects your session and you lose all files in `/content/`. This is different from "Restart runtime" which only clears memory.

**Lost files:**
- ❌ BHT25_All_annotated.csv (3 hours of work)
- ❌ Demo output files
- ❌ All code files

## Fresh Start - Complete Setup

### Cell 1: Clone Repository (With All CUDA Fixes)

```python
import os
from google.colab import drive

# Mount Google Drive to save important files
drive.mount('/content/drive')

# Create project directory in Drive (to prevent data loss)
project_dir = '/content/drive/MyDrive/ESA_NMT_Project'
os.makedirs(project_dir, exist_ok=True)

print(f"✅ Drive mounted. Project dir: {project_dir}")

# Clone the repository with all fixes
if os.path.exists('ESA-NMT'):
    print("⚠️  ESA-NMT already exists, removing...")
    !rm -rf ESA-NMT

print("📥 Cloning repository with CUDA fixes...")
!git clone https://github.com/SSanpui/ESA-NMT.git
%cd ESA-NMT

# Checkout the correct branch with all fixes
!git checkout claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj

print("\n✅ Repository cloned!")
print("Branch:", !git branch --show-current)

# List CUDA fix files
print("\n📁 CUDA Fix Files:")
!ls -lh fix_cuda_error.py diagnose_cuda_error.py colab_cell_full_training_safe.py 2>/dev/null || echo "Files should be present"
```

### Cell 2: Install Dependencies

```python
print("📦 Installing dependencies...")

!pip install -q torch transformers sentencepiece sacrebleu rouge-score bert-score sentence-transformers
!pip install -q accelerate sacremoses datasets tokenizers protobuf
!pip install -q pandas numpy matplotlib seaborn tqdm

print("✅ All dependencies installed!")
```

### Cell 3: Verify Dataset

```python
import pandas as pd
import os

# Check if original dataset exists
if os.path.exists('BHT25_All.csv'):
    df = pd.read_csv('BHT25_All.csv')
    print(f"✅ Dataset found: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Size: {os.path.getsize('BHT25_All.csv') / 1024**2:.1f} MB")
else:
    print("❌ BHT25_All.csv not found!")
    print("You need to upload it first.")
```

### Cell 4: Smart Annotation Check

```python
import os

# Check if annotated file already exists in Drive (backup location)
drive_annotated = '/content/drive/MyDrive/ESA_NMT_Project/BHT25_All_annotated.csv'
local_annotated = 'BHT25_All_annotated.csv'

if os.path.exists(drive_annotated):
    print("🎉 Found annotated CSV in Google Drive backup!")
    print("Copying from Drive to local...")
    !cp "$drive_annotated" "$local_annotated"

    # Verify
    import pandas as pd
    df = pd.read_csv(local_annotated)
    print(f"✅ Restored: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Check emotion range
    if 'emotion_bn' in df.columns:
        emotion_values = df['emotion_bn'].values
        print(f"Emotion range: [{emotion_values.min()}, {emotion_values.max()}]")

        if emotion_values.max() <= 3:
            print("✅ Dataset is valid (4 emotions)!")
            print("\n👉 Skip to Cell 6 (Apply CUDA Fix)")
        else:
            print("⚠️  Dataset has 8 emotions, need to re-annotate")

elif os.path.exists(local_annotated):
    print("✅ Annotated CSV found locally!")
    print("Backing up to Drive...")
    !cp "$local_annotated" "$drive_annotated"
    print("✅ Backed up to Drive!")
    print("\n👉 Skip to Cell 6 (Apply CUDA Fix)")

else:
    print("❌ No annotated CSV found.")
    print("👉 Continue to Cell 5 (Run Annotation)")
```

### Cell 5: Run Annotation (ONLY if needed - takes ~3 hours)

```python
import os
from google.colab import drive

print("⏰ Starting annotation... This will take ~3 hours!")
print("📊 Progress will be shown below.")

# Run annotation
!python annotate_dataset.py

# Verify output
if os.path.exists('BHT25_All_annotated.csv'):
    import pandas as pd
    df = pd.read_csv('BHT25_All_annotated.csv')
    print(f"\n✅ Annotation completed: {len(df)} rows")

    # IMMEDIATELY backup to Google Drive
    drive_backup = '/content/drive/MyDrive/ESA_NMT_Project/BHT25_All_annotated.csv'
    !cp BHT25_All_annotated.csv "$drive_backup"
    print(f"✅ BACKED UP to Drive: {drive_backup}")
    print("   Your 3-hour work is now safe in Google Drive!")

    # Also show statistics
    emotion_values = df['emotion_bn'].values
    emotion_names = ['joy', 'sadness', 'anger', 'fear']
    print("\n📊 Emotion Distribution:")
    for i in range(4):
        count = (emotion_values == i).sum()
        pct = count / len(emotion_values) * 100
        print(f"   {i} ({emotion_names[i]:8s}): {count:5d} ({pct:5.1f}%)")
else:
    print("❌ Annotation failed!")
```

### Cell 6: Apply CUDA Error Fix

```python
print("🔧 Applying CUDA error fix...")

# Run the fix script
!python fix_cuda_error.py

print("\n✅ CUDA fix applied!")
print("\n⚠️  Important: DO NOT restart session!")
print("    Continue to next cell directly.")
```

### Cell 7: Configuration

```python
# Set your configuration
TRANSLATION_PAIR = "bn-hi"  # or "bn-te"
MODEL_TYPE = "nllb"
NUM_EPOCHS = 3
BATCH_SIZE = 2

print(f"📋 Configuration:")
print(f"   Translation: {TRANSLATION_PAIR}")
print(f"   Model: {MODEL_TYPE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch Size: {BATCH_SIZE}")
```

### Cell 8: Run Training (Safe Version)

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error messages

from emotion_semantic_nmt_enhanced import full_training_pipeline

print("🚀 Starting Full Training...")
print("="*70)

try:
    metrics = full_training_pipeline(
        csv_path='BHT25_All.csv',  # Will auto-load annotated version
        translation_pair=TRANSLATION_PAIR,
        model_type=MODEL_TYPE
    )

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)

    # Save results to Drive
    import json
    from emotion_semantic_nmt_enhanced import ComprehensiveEvaluator

    results_path = f'/content/drive/MyDrive/ESA_NMT_Project/training_results_{MODEL_TYPE}_{TRANSLATION_PAIR}.json'
    with open(results_path, 'w') as f:
        json.dump(ComprehensiveEvaluator.convert_to_json_serializable(metrics), f, indent=2)

    print(f"\n💾 Results saved to Drive: {results_path}")

    # Copy checkpoint to Drive
    checkpoint_local = f'./checkpoints/final_model_{MODEL_TYPE}_{TRANSLATION_PAIR}.pt'
    checkpoint_drive = f'/content/drive/MyDrive/ESA_NMT_Project/final_model_{MODEL_TYPE}_{TRANSLATION_PAIR}.pt'

    if os.path.exists(checkpoint_local):
        !cp "$checkpoint_local" "$checkpoint_drive"
        print(f"💾 Model saved to Drive: {checkpoint_drive}")

    print("\n🎉 All outputs backed up to Google Drive!")

except Exception as e:
    print("\n❌ Error during training:")
    print(e)
    print("\nRun: !python diagnose_cuda_error.py")

```

### Cell 9: (Optional) Save Work to Drive Manually

```python
# Run this periodically to backup everything

import shutil
from datetime import datetime

backup_dir = f'/content/drive/MyDrive/ESA_NMT_Project/backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(backup_dir, exist_ok=True)

# Backup important files
files_to_backup = [
    'BHT25_All_annotated.csv',
    './outputs/',
    './checkpoints/'
]

for item in files_to_backup:
    if os.path.exists(item):
        if os.path.isdir(item):
            shutil.copytree(item, f"{backup_dir}/{os.path.basename(item)}", dirs_exist_ok=True)
        else:
            shutil.copy(item, backup_dir)
        print(f"✅ Backed up: {item}")

print(f"\n💾 Backup completed: {backup_dir}")
```

## Key Changes to Prevent Data Loss

1. **Mount Google Drive** - Store important files there
2. **Auto-backup annotated CSV** - Immediately after annotation
3. **Check Drive for existing files** - Don't re-annotate if backup exists
4. **Backup checkpoints** - Save trained models to Drive
5. **Save results to Drive** - Keep all outputs safe

## Summary

**Run cells in order:**
1. ✅ Clone repo (Cell 1)
2. ✅ Install packages (Cell 2)
3. ✅ Check dataset (Cell 3)
4. ✅ Check for backup in Drive (Cell 4)
5. ⏰ Annotate ONLY if no backup found (Cell 5) - 3 hours
6. 🔧 Apply CUDA fix (Cell 6)
7. ⚙️ Configure (Cell 7)
8. 🚀 Train (Cell 8)

**Important:** Cell 4 will check if you have a backup in Google Drive, so you might not need to wait 3 hours again!

## If You Have the Annotated CSV Elsewhere

If you downloaded `BHT25_All_annotated.csv` to your computer before:

```python
from google.colab import files
uploaded = files.upload()  # Upload BHT25_All_annotated.csv

# Then backup to Drive
!cp BHT25_All_annotated.csv /content/drive/MyDrive/ESA_NMT_Project/
print("✅ Uploaded and backed up!")
```

Then skip to Cell 6!
