# Updated Colab Notebook Instructions

## 🚨 IMPORTANT: The notebook needs to be updated!

**Current notebook uses OLD code with random labels → 99% fake accuracy!**

---

## ✅ **How to Use (CORRECT WAY)**

### **Option 1: Run Annotation + Training in Colab** (Recommended)

**Add these cells to your notebook:**

```python
# ============================================================================
# CELL 1: Clone & Setup (Replace existing clone cell)
# ============================================================================

!git clone https://github.com/SSanpui/ESA-NMT.git
%cd ESA-NMT
!git checkout claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj
!git pull origin claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj

print("✅ Latest code loaded!")
```

```python
# ============================================================================
# CELL 2: Install Dependencies
# ============================================================================

!pip install -q transformers>=4.30.0 sentence-transformers>=2.2.0 \
    sacrebleu>=2.3.0 rouge-score>=0.1.2 accelerate>=0.20.0 datasets>=2.12.0

import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("✅ Dependencies installed!")
```

```python
# ============================================================================
# CELL 3: 🔥 ANNOTATE DATASET (ONE-TIME, 30-60 mins)
# ============================================================================

import os

# Check if already annotated
if os.path.exists('BHT25_All_annotated.csv'):
    print("✅ Annotated dataset already exists!")
else:
    print("🔄 Annotating dataset... (this will take 30-60 minutes)")
    print("⏰ Grab a coffee! This creates REAL emotion/semantic labels.")

    !python annotate_dataset.py

    print("✅ Annotation complete!")
```

```python
# ============================================================================
# CELL 4: Train with PROPER annotations
# ============================================================================

from dataset_with_annotations import BHT25AnnotatedDataset  # ← FIXED dataset
from emotion_semantic_nmt_enhanced import (
    EmotionSemanticNMT, Config, Trainer, ComprehensiveEvaluator
)
from torch.utils.data import DataLoader
import torch

config = Config()
device = torch.device('cuda')

# Configuration
TRANSLATION_PAIR = "bn-hi"
MODEL_TYPE = "nllb"
config.EPOCHS['phase1'] = 3

print(f"\n{'='*60}")
print(f"Training: {TRANSLATION_PAIR} with PROPER annotations")
print(f"{'='*60}\n")

# Create model
model = EmotionSemanticNMT(config, model_type=MODEL_TYPE).to(device)

# ✅ Use ANNOTATED dataset (not random labels!)
train_dataset = BHT25AnnotatedDataset(
    'BHT25_All.csv',  # Will auto-load BHT25_All_annotated.csv
    model.tokenizer,
    TRANSLATION_PAIR,
    config.MAX_LENGTH,
    'train',
    MODEL_TYPE
)

test_dataset = BHT25AnnotatedDataset(
    'BHT25_All.csv',
    model.tokenizer,
    TRANSLATION_PAIR,
    config.MAX_LENGTH,
    'test',
    MODEL_TYPE
)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                         shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                        shuffle=False, num_workers=0)

# Train
trainer = Trainer(model, config, TRANSLATION_PAIR)

for epoch in range(config.EPOCHS['phase1']):
    loss = trainer.train_epoch(train_loader, epoch)
    print(f"✅ Epoch {epoch+1}/{config.EPOCHS['phase1']} - Loss: {loss:.4f}")

# Evaluate
evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, TRANSLATION_PAIR)
metrics, preds, refs, sources = evaluator.evaluate(test_loader)

print("\n" + "="*60)
print("📊 FINAL RESULTS (with REAL annotations):")
print("="*60)

for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key:25s}: {value:.4f}")
    else:
        print(f"{key:25s}: {value}")

print("\n⚠️ Expected realistic values:")
print("  - Emotion Accuracy: 73-78% (NOT 99%!)")
print("  - Semantic Score: 0.79-0.87 (NOT 0.99!)")
```

---

### **Option 2: Just Run Python Scripts** (Simpler)

If you don't want to use notebook:

```bash
# 1. Clone repo
git clone https://github.com/SSanpui/ESA-NMT.git
cd ESA-NMT

# 2. Install deps
pip install -r requirements.txt

# 3. Annotate (one-time, 30-60 mins)
python annotate_dataset.py

# 4. Train with annotations
python -c "
from dataset_with_annotations import BHT25AnnotatedDataset
from emotion_semantic_nmt_enhanced import *
# ... (rest of training code)
"

# Or use the comparison script
python generate_table4_colab.py
```

---

## ⚠️ **DON'T Use Old Notebook As-Is!**

The current notebook will give you **99% fake accuracy** because it uses:
- `BHT25Dataset` ← OLD (random labels)

You need:
- `BHT25AnnotatedDataset` ← NEW (real labels)

---

## 📝 **Summary**

**What you need to do:**

1. ✅ Run `python annotate_dataset.py` (creates BHT25_All_annotated.csv)
2. ✅ Use `BHT25AnnotatedDataset` instead of `BHT25Dataset`
3. ✅ Get realistic 73-78% accuracy (not 99%)

**Annotated dataset:**
- ❌ NOT in GitHub (too large + generated locally)
- ✅ Generated by YOU using annotation script
- ✅ Saved as `BHT25_All_annotated.csv` (UTF-8 encoding)
- ✅ One-time process (30-60 mins)

**Files to use:**
- `annotate_dataset.py` - Creates annotated dataset
- `dataset_with_annotations.py` - Fixed dataset class
- `generate_table4_colab.py` - Complete training script

---

**I'll create an updated notebook file now!**
