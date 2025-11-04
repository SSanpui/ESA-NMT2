#!/usr/bin/env python3
"""
Retrain ESA-NMT with properly integrated emotion/semantic modules
This fixes the bug where modules were only used for loss, not generation
"""

import torch
import gc
import os

# Import from main file
from emotion_semantic_nmt_enhanced import (
    Config, EmotionSemanticNMT, Trainer, ComprehensiveEvaluator,
    device, config
)
from dataset_with_annotations import BHT25AnnotatedDataset
from torch.utils.data import DataLoader

print("""
╔═══════════════════════════════════════════════════════════════╗
║  RETRAIN ESA-NMT with Fixed Code (GAMMA=0.5)                 ║
║  This will properly integrate emotion/semantic modules        ║
╚═══════════════════════════════════════════════════════════════╝
""")

# Configuration
TRANSLATION_PAIR = 'bn-hi'  # Change to 'bn-te' for Bengali-Telugu
CSV_PATH = 'BHT25_All_annotated.csv'  # Your annotated dataset
print(f"\n📋 Configuration:")
print(f"   Translation pair: {TRANSLATION_PAIR}")
print(f"   Loss weights: α={config.ALPHA}, β={config.BETA}, γ={config.GAMMA}, δ={config.DELTA}")
print(f"   Epochs: {config.EPOCHS['phase1']}")
print(f"   Batch size: {config.BATCH_SIZE}")
print(f"   Device: {device}")

# Create model with ALL modules enabled
print(f"\n1️⃣ Creating ESA-NMT model with all modules...")
model = EmotionSemanticNMT(
    config,
    model_type='nllb',
    use_emotion=True,   # ← Enabled
    use_semantic=True,  # ← Enabled
    use_style=True      # ← Enabled
).to(device)

print(f"   ✅ Model created with emotion, semantic, and style modules")

# Create datasets - MUST use annotated dataset
print(f"\n2️⃣ Loading annotated dataset...")
train_dataset = BHT25AnnotatedDataset(
    CSV_PATH,
    model.tokenizer,
    TRANSLATION_PAIR,
    config.MAX_LENGTH,
    'train',
    'nllb'
)
val_dataset = BHT25AnnotatedDataset(
    CSV_PATH,
    model.tokenizer,
    TRANSLATION_PAIR,
    config.MAX_LENGTH,
    'val',
    'nllb'
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"   ✅ Train samples: {len(train_dataset)}")
print(f"   ✅ Val samples: {len(val_dataset)}")

# Create trainer
print(f"\n3️⃣ Starting training...")
trainer = Trainer(model, config, TRANSLATION_PAIR)

# Train for 3 epochs
for epoch in range(config.EPOCHS['phase1']):
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch+1}/{config.EPOCHS['phase1']}")
    print(f"{'='*70}")

    train_loss = trainer.train_epoch(train_loader, epoch)

    print(f"\n✅ Epoch {epoch+1} completed - Train Loss: {train_loss:.4f}")

    # Validation every epoch
    print(f"\n📊 Running validation...")
    evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, TRANSLATION_PAIR)
    metrics, _, _, _ = evaluator.evaluate(val_loader)

    print(f"\nValidation Results:")
    print(f"   BLEU:    {metrics.get('bleu', 0):.2f}")
    print(f"   METEOR:  {metrics.get('meteor', 0):.2f}")
    print(f"   chrF:    {metrics.get('chrf', 0):.2f}")
    print(f"   Emotion: {metrics.get('emotion_accuracy', 0):.2f}%")
    print(f"   Semantic: {metrics.get('semantic_score', 0):.4f}")

    # Save checkpoint after each epoch
    checkpoint_path = f"{config.CHECKPOINT_DIR}/esa_nmt_{TRANSLATION_PAIR}_epoch{epoch+1}.pt"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics
    }, checkpoint_path)
    print(f"   💾 Checkpoint saved: {checkpoint_path}")

    # Memory cleanup
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n{'='*70}")
print(f"✅ TRAINING COMPLETED!")
print(f"{'='*70}")

# Final evaluation on validation set
print(f"\n4️⃣ Final evaluation on validation set...")
evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, TRANSLATION_PAIR)
final_metrics, _, _, _ = evaluator.evaluate(val_loader)

print(f"\n📊 FINAL RESULTS:")
print(f"{'='*70}")
print(f"BLEU:     {final_metrics.get('bleu', 0):.2f}")
print(f"METEOR:   {final_metrics.get('meteor', 0):.2f}")
print(f"chrF:     {final_metrics.get('chrf', 0):.2f}")
print(f"ROUGE-L:  {final_metrics.get('rouge_l', 0):.2f}")
print(f"Emotion:  {final_metrics.get('emotion_accuracy', 0):.2f}%")
print(f"Semantic: {final_metrics.get('semantic_score', 0):.4f}")
print(f"{'='*70}")

# Save final model
final_path = f"{config.CHECKPOINT_DIR}/final_esa_nmt_{TRANSLATION_PAIR}.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'metrics': final_metrics,
    'translation_pair': TRANSLATION_PAIR
}, final_path)

print(f"\n💾 Final model saved: {final_path}")

# Save metrics to JSON
import json
results_path = f"{config.OUTPUT_DIR}/training_results_{TRANSLATION_PAIR}.json"
with open(results_path, 'w') as f:
    json.dump(ComprehensiveEvaluator.convert_to_json_serializable(final_metrics), f, indent=2)

print(f"📄 Results saved: {results_path}")

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  ✅ TRAINING COMPLETE!                                        ║
║                                                               ║
║  Next step: Run ablation study with new checkpoint:          ║
║  - Edit ablation_study_only.py line 37                       ║
║  - Set checkpoint path to: {final_path}
║  - Run: python ablation_study_only.py                        ║
╚═══════════════════════════════════════════════════════════════╝
""")
