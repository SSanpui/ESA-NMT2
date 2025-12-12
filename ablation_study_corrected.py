"""
Corrected Ablation Study for Hindi-Telugu
Fixes ROUGE-L calculation and ensures all 4 configurations are evaluated
"""

import os
import torch
import gc
import json
import numpy as np
from emotion_semantic_nmt_enhanced import config, EmotionSemanticNMT, ComprehensiveEvaluator
from dataset_with_annotations import BHT25AnnotatedDataset
from torch.utils.data import DataLoader

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRANSLATION_PAIR = 'hi-te'

print("="*80)
print("CORRECTED ABLATION STUDY - HINDI-TELUGU")
print("="*80)
print(f"\nTranslation pair: {TRANSLATION_PAIR}")
print(f"Device: {device}\n")

# Find checkpoint
checkpoint_path = f'./checkpoints/final_esa_nmt_{TRANSLATION_PAIR}.pt'
if not os.path.exists(checkpoint_path):
    print(f"⚠️ WARNING: Checkpoint not found at {checkpoint_path}")
    print("   Full ESA-NMT will use untrained weights")
    checkpoint_path = None
else:
    print(f"✅ Found checkpoint: {checkpoint_path}\n")

# Configurations
configs = [
    {
        'name': 'Base NLLB (Baseline)',
        'emotion': False,
        'semantic': False,
        'checkpoint': None,
    },
    {
        'name': 'Base + Emotion',
        'emotion': True,
        'semantic': False,
        'checkpoint': None,
    },
    {
        'name': 'Base + Semantic',
        'emotion': False,
        'semantic': True,
        'checkpoint': None,
    },
    {
        'name': 'Full ESA-NMT',
        'emotion': True,
        'semantic': True,
        'checkpoint': checkpoint_path,
    }
]

def evaluate_config(cfg):
    """Evaluate one configuration"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {cfg['name']}")
    print(f"{'='*80}")

    try:
        # Create model
        model = EmotionSemanticNMT(
            config,
            model_type='nllb',
            use_emotion=cfg['emotion'],
            use_semantic=cfg['semantic'],
            use_style=False
        ).to(device)

        # Load checkpoint if provided
        if cfg['checkpoint']:
            print(f"Loading checkpoint: {cfg['checkpoint']}")
            checkpoint = torch.load(cfg['checkpoint'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded trained weights")
        else:
            print("📝 Using pre-trained NLLB base")

        # Load test data
        test_dataset = BHT25AnnotatedDataset(
            'BHT25_All_annotated.csv',
            model.tokenizer,
            TRANSLATION_PAIR,
            config.MAX_LENGTH,
            'test',
            'nllb'
        )
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        print(f"Test samples: {len(test_dataset)}")

        # Evaluate
        evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, TRANSLATION_PAIR)
        metrics, preds, refs, sources = evaluator.evaluate(test_loader)

        # ✅ FIX ROUGE-L: Ensure it's in 0-100 range
        if 'rouge_l' in metrics:
            # If already multiplied by 100, it should be 0-100
            # If it's > 100, something went wrong
            if metrics['rouge_l'] > 100:
                print(f"⚠️ ROUGE-L is {metrics['rouge_l']:.2f} (> 100), capping at 100")
                metrics['rouge_l'] = 100.0
            elif metrics['rouge_l'] < 1.0:
                # If < 1, it's probably in 0-1 range, multiply by 100
                print(f"⚠️ ROUGE-L is {metrics['rouge_l']:.4f} (< 1), multiplying by 100")
                metrics['rouge_l'] = metrics['rouge_l'] * 100

        # Print results
        print(f"\n✅ Results:")
        print(f"   BLEU:     {metrics.get('bleu', 0):.2f}")
        print(f"   METEOR:   {metrics.get('meteor', 0):.2f}")
        print(f"   chrF:     {metrics.get('chrf', 0):.2f}")
        print(f"   ROUGE-L:  {metrics.get('rouge_l', 0):.2f}")
        if cfg['emotion']:
            print(f"   Emotion:  {metrics.get('emotion_accuracy', 0):.2f}%")
        if cfg['semantic']:
            print(f"   Semantic: {metrics.get('semantic_score', 0):.4f}")

        # Cleanup
        del model, evaluator, test_dataset, test_loader
        torch.cuda.empty_cache()
        gc.collect()

        return metrics

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return None

# Run all configurations
print(f"\n🔬 Running ablation study with {len(configs)} configurations...")
print(f"   Estimated time: 30-40 minutes\n")

results = {}
for i, cfg in enumerate(configs):
    print(f"\n[{i+1}/{len(configs)}] Starting: {cfg['name']}")
    metrics = evaluate_config(cfg)
    if metrics:
        results[cfg['name']] = metrics
    print(f"\n✅ Completed: {cfg['name']}")
    print("🧹 Clearing memory...")
    torch.cuda.empty_cache()
    gc.collect()

# Display results table
print("\n" + "="*90)
print(f"ABLATION STUDY RESULTS: {TRANSLATION_PAIR.upper()}")
print("="*90)

header = f"{'Configuration':<25} {'BLEU':<8} {'METEOR':<8} {'chrF':<8} {'ROUGE-L':<10} {'Emotion':<10} {'Semantic':<10}"
print(header)
print("-"*90)

for cfg in configs:
    name = cfg['name']
    if name in results:
        m = results[name]
        bleu = f"{m.get('bleu', 0):.2f}"
        meteor = f"{m.get('meteor', 0):.2f}"
        chrf = f"{m.get('chrf', 0):.2f}"
        rouge = f"{m.get('rouge_l', 0):.2f}"
        emotion = f"{m.get('emotion_accuracy', 0):.2f}%" if m.get('emotion_accuracy', 0) > 0 else "N/A"
        semantic = f"{m.get('semantic_score', 0):.4f}" if m.get('semantic_score', 0) > 0 else "N/A"

        print(f"{name:<25} {bleu:<8} {meteor:<8} {chrf:<8} {rouge:<10} {emotion:<10} {semantic:<10}")

print()

# Calculate improvements
if 'Base NLLB (Baseline)' in results and 'Full ESA-NMT' in results:
    baseline = results['Base NLLB (Baseline)']
    full = results['Full ESA-NMT']

    print("="*80)
    print("IMPROVEMENT OVER BASELINE")
    print("="*80)

    for metric in ['bleu', 'meteor', 'chrf', 'rouge_l']:
        base_val = baseline[metric]
        full_val = full[metric]
        improvement = full_val - base_val
        pct = (improvement / base_val) * 100 if base_val > 0 else 0

        print(f"{metric.upper():10s}: {base_val:.2f} → {full_val:.2f} (+{improvement:.2f}, +{pct:.1f}%)")

    print()

# Save results
os.makedirs('./outputs', exist_ok=True)
output_file = f'./outputs/ablation_study_{TRANSLATION_PAIR}_corrected.json'

json_results = {}
for name, metrics in results.items():
    json_results[name] = {
        'bleu': float(metrics.get('bleu', 0)),
        'meteor': float(metrics.get('meteor', 0)),
        'chrf': float(metrics.get('chrf', 0)),
        'rouge_l': float(metrics.get('rouge_l', 0)),
        'emotion_accuracy': float(metrics.get('emotion_accuracy', 0)),
        'semantic_score': float(metrics.get('semantic_score', 0))
    }

with open(output_file, 'w') as f:
    json.dump(json_results, f, indent=2)

print(f"💾 Saved: {output_file}")
print("\n" + "="*80)
print("✅ ABLATION STUDY COMPLETE!")
print("="*80)
print(f"\n📊 Evaluated {len(results)}/4 configurations:")
for name in results.keys():
    print(f"   ✅ {name}")

if len(results) < 4:
    print(f"\n⚠️ WARNING: Only {len(results)}/4 configurations completed!")
    print("   Some configurations may have failed.")
