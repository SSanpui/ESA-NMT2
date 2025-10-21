"""
ESA-NMT Google Colab Notebook
Emotion-Semantic-Aware Neural Machine Translation

Run this notebook in Google Colab to train and evaluate the ESA-NMT model.

Requirements:
- Google Colab Pro (recommended) or Free tier with limitations
- Runtime: GPU (T4, V100, or A100)
- Estimated time: 4-8 hours for complete experiments
"""

# ============================================================================
# SECTION 1: ENVIRONMENT SETUP
# ============================================================================

print("="*60)
print("ESA-NMT: Emotion-Semantic-Aware Neural Machine Translation")
print("="*60)

# Check GPU availability
import torch
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️ WARNING: No GPU detected. This will be very slow!")
    print("   Go to Runtime > Change runtime type > Hardware accelerator > GPU")

# ============================================================================
# SECTION 2: CLONE REPOSITORY
# ============================================================================

print("\n📥 Cloning repository...")
!git clone https://github.com/SSanpui/ESA-NMT.git
%cd ESA-NMT

# Checkout the enhanced branch
!git checkout claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj

print("✅ Repository cloned!")

# ============================================================================
# SECTION 3: INSTALL DEPENDENCIES
# ============================================================================

print("\n📦 Installing dependencies...")
!pip install -q transformers>=4.30.0 \
    sentence-transformers>=2.2.0 \
    sacrebleu>=2.3.0 \
    rouge-score>=0.1.2 \
    accelerate>=0.20.0 \
    datasets>=2.12.0

# Install NLTK data for METEOR
import nltk
print("📚 Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("✅ All dependencies installed!")

# ============================================================================
# SECTION 4: VERIFY DATASET
# ============================================================================

print("\n📊 Checking dataset...")
import os
import pandas as pd

if os.path.exists('BHT25_All.csv'):
    df = pd.read_csv('BHT25_All.csv')
    print(f"✅ Dataset loaded: {len(df)} parallel sentences")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"\n   Sample data:")
    print(df.head(3))
else:
    print("❌ Dataset not found!")

# ============================================================================
# SECTION 5: CONFIGURATION
# ============================================================================

print("\n⚙️ Configuration...")

# Choose what to run
RUN_MODE = "quick_demo"  # Options: "quick_demo", "full_training", "ablation", "tuning"
TRANSLATION_PAIR = "bn-hi"  # Options: "bn-hi", "bn-te"
MODEL_TYPE = "nllb"  # Options: "nllb", "indictrans2"

print(f"""
Configuration:
- Mode: {RUN_MODE}
- Translation Pair: {TRANSLATION_PAIR}
- Model Type: {MODEL_TYPE}

Estimated Runtime:
- quick_demo: ~30 minutes (1 epoch, small validation)
- full_training: ~2-3 hours (3 epochs, full evaluation)
- ablation: ~4-6 hours (tests 6 configurations)
- tuning: ~3-5 hours (grid search hyperparameters)
""")

# ============================================================================
# SECTION 6: QUICK DEMO MODE
# ============================================================================

if RUN_MODE == "quick_demo":
    print("\n" + "="*60)
    print("RUNNING QUICK DEMO")
    print("="*60)

    from emotion_semantic_nmt_enhanced import (
        EmotionSemanticNMT, Config, BHT25Dataset, Trainer,
        ComprehensiveEvaluator
    )
    from torch.utils.data import DataLoader
    import torch

    # Reduce config for quick demo
    config = Config()
    config.BATCH_SIZE = 2
    config.EPOCHS['phase1'] = 1  # Just 1 epoch for demo
    config.MAX_LENGTH = 96

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n1️⃣ Creating model...")
    model = EmotionSemanticNMT(config, model_type=MODEL_TYPE).to(device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n2️⃣ Loading dataset...")
    train_dataset = BHT25Dataset('BHT25_All.csv', model.tokenizer, TRANSLATION_PAIR,
                                config.MAX_LENGTH, 'train', MODEL_TYPE)
    val_dataset = BHT25Dataset('BHT25_All.csv', model.tokenizer, TRANSLATION_PAIR,
                              config.MAX_LENGTH, 'val', MODEL_TYPE)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                           shuffle=False, num_workers=0)

    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")

    print("\n3️⃣ Training (1 epoch)...")
    trainer = Trainer(model, config, TRANSLATION_PAIR)
    train_loss = trainer.train_epoch(train_loader, 0)
    print(f"   Training Loss: {train_loss:.4f}")

    print("\n4️⃣ Evaluating...")
    evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, TRANSLATION_PAIR)
    metrics, preds, refs, sources = evaluator.evaluate(val_loader)

    print("\n📊 RESULTS:")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    print("\n📝 Sample Translations:")
    print("="*60)
    for i in range(min(5, len(preds))):
        print(f"\nExample {i+1}:")
        print(f"  Source:     {sources[i][:100]}...")
        print(f"  Reference:  {refs[i][:100]}...")
        print(f"  Prediction: {preds[i][:100]}...")

    # Save results
    import json
    results = {
        'mode': 'quick_demo',
        'translation_pair': TRANSLATION_PAIR,
        'model_type': MODEL_TYPE,
        'metrics': metrics,
        'train_loss': train_loss
    }

    with open('quick_demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Quick demo completed!")
    print("   Results saved to: quick_demo_results.json")

# ============================================================================
# SECTION 7: FULL TRAINING MODE
# ============================================================================

elif RUN_MODE == "full_training":
    print("\n" + "="*60)
    print("RUNNING FULL TRAINING")
    print("="*60)

    !python emotion_semantic_nmt_enhanced.py <<EOF
4
{TRANSLATION_PAIR}
{MODEL_TYPE}
EOF

# ============================================================================
# SECTION 8: ABLATION STUDY MODE
# ============================================================================

elif RUN_MODE == "ablation":
    print("\n" + "="*60)
    print("RUNNING ABLATION STUDY")
    print("="*60)

    !python emotion_semantic_nmt_enhanced.py <<EOF
2
{TRANSLATION_PAIR}
{MODEL_TYPE}
EOF

# ============================================================================
# SECTION 9: HYPERPARAMETER TUNING MODE
# ============================================================================

elif RUN_MODE == "tuning":
    print("\n" + "="*60)
    print("RUNNING HYPERPARAMETER TUNING")
    print("="*60)

    !python emotion_semantic_nmt_enhanced.py <<EOF
3
{TRANSLATION_PAIR}
{MODEL_TYPE}
EOF

# ============================================================================
# SECTION 10: VISUALIZE RESULTS
# ============================================================================

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Check if results exist
import os
if os.path.exists('./outputs'):
    print("📊 Generating semantic score visualizations...")
    !python visualize_semantic_scores.py

    # Display saved images
    from IPython.display import Image, display
    import glob

    print("\n🎨 Visualizations:")
    for img_file in glob.glob('./outputs/*.png'):
        print(f"\n{img_file}:")
        display(Image(filename=img_file))
else:
    print("⚠️ No results to visualize yet. Run training first.")

# ============================================================================
# SECTION 11: DOWNLOAD RESULTS
# ============================================================================

print("\n" + "="*60)
print("DOWNLOADING RESULTS")
print("="*60)

# Zip all results
!zip -r esa_nmt_results.zip ./outputs ./checkpoints ./models -x "*.git*"

print("""
✅ Results packaged!

To download:
1. Click on the folder icon (📁) in the left sidebar
2. Right-click on 'esa_nmt_results.zip'
3. Select 'Download'

Or run this in a cell:
    from google.colab import files
    files.download('esa_nmt_results.zip')
""")

# ============================================================================
# SECTION 12: SUMMARY
# ============================================================================

print("\n" + "="*60)
print("EXECUTION SUMMARY")
print("="*60)

import json
import glob

print("\n📁 Files generated:")
for pattern in ['./outputs/*.json', './outputs/*.png', './checkpoints/*.pt']:
    files = glob.glob(pattern)
    if files:
        print(f"\n{pattern}:")
        for f in files:
            size = os.path.getsize(f) / (1024*1024)  # MB
            print(f"  - {f} ({size:.2f} MB)")

print("\n" + "="*60)
print("✅ EXECUTION COMPLETE!")
print("="*60)

print("""
Next steps:
1. Review results in ./outputs/
2. Download results using the zip file
3. Check visualizations above
4. Deploy model to Hugging Face (optional):
   !python deploy_to_huggingface.py --model_type nllb --translation_pair bn-hi --hf_username YOUR_USERNAME
""")
