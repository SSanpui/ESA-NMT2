"""
Lightweight Ablation for Memory-Constrained GPUs
Run this in Kaggle - trains only required configs, one at a time
"""

import os
os.chdir('/kaggle/working/ESA-NMT')

import torch
import gc
from emotion_semantic_nmt_enhanced import config, EmotionSemanticNMT, Trainer, ComprehensiveEvaluator
from dataset_with_annotations import BHT25AnnotatedDataset
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np

# Reduce batch size for memory
config.BATCH_SIZE = 1  # ← Changed from 2 to 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("🔬 Lightweight Ablation Study (Required Configs Only)")
print("="*70)

# ONLY 4 required configurations (not 7!)
configurations = [
    {'name': 'Baseline', 'emotion': False, 'semantic': False, 'style': False},
    {'name': 'Emotion Only', 'emotion': True, 'semantic': False, 'style': False},
    {'name': 'Semantic Only', 'emotion': False, 'semantic': True, 'style': False},
    {'name': 'Full Model', 'emotion': True, 'semantic': True, 'style': True},
]

def run_lightweight_ablation(translation_pair):
    """Run ablation for one pair with memory management"""

    results = {}

    for i, conf in enumerate(configurations):
        print(f"\n{'='*70}")
        print(f"[{i+1}/4] Testing: {conf['name']} ({translation_pair})")
        print(f"{'='*70}")

        # Clear memory before each config
        torch.cuda.empty_cache()
        gc.collect()
        print(f"💾 Memory before: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        try:
            # Create model
            model = EmotionSemanticNMT(
                config,
                model_type='nllb',
                use_emotion=conf['emotion'],
                use_semantic=conf['semantic'],
                use_style=conf['style']
            ).to(device)

            # Load datasets
            train_dataset = BHT25AnnotatedDataset(
                'BHT25_All.csv', model.tokenizer, translation_pair,
                config.MAX_LENGTH, 'train', 'nllb'
            )
            val_dataset = BHT25AnnotatedDataset(
                'BHT25_All.csv', model.tokenizer, translation_pair,
                config.MAX_LENGTH, 'val', 'nllb'
            )

            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

            # Train for 1 epoch only (instead of 2) to save time/memory
            trainer = Trainer(model, config, translation_pair)
            print(f"Training 1 epoch...")
            train_loss = trainer.train_epoch(train_loader, 0)
            print(f"Train Loss: {train_loss:.4f}")

            # Evaluate
            evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
            metrics, _, _, _ = evaluator.evaluate(val_loader)

            results[conf['name']] = metrics

            print(f"\n✅ Results for {conf['name']}:")
            print(f"  BLEU: {metrics['bleu']:.2f}")
            print(f"  chrF: {metrics['chrf']:.2f}")
            print(f"  ROUGE-L: {metrics['rouge_l']:.2f}")

            # IMPORTANT: Delete everything to free memory
            del model
            del trainer
            del train_dataset
            del val_dataset
            del train_loader
            del val_loader
            del evaluator
            del metrics

            torch.cuda.empty_cache()
            gc.collect()

            print(f"🧹 Memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        except Exception as e:
            print(f"❌ Error in {conf['name']}: {e}")
            # Still try to clean up
            torch.cuda.empty_cache()
            gc.collect()
            continue

    # Save results
    os.makedirs('./outputs', exist_ok=True)
    output_file = f'./outputs/ablation_lightweight_{translation_pair}.json'

    # Convert to JSON-serializable
    json_results = {}
    for key, val in results.items():
        json_results[key] = {
            'bleu': float(val['bleu']),
            'chrf': float(val['chrf']),
            'rouge_l': float(val['rouge_l'])
        }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Saved: {output_file}")

    # Create visualization
    create_visualization(results, translation_pair)

    return results

def create_visualization(results, translation_pair):
    """Create simple bar chart"""

    configs = list(results.keys())
    bleu_scores = [results[c]['bleu'] for c in configs]
    chrf_scores = [results[c]['chrf'] for c in configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # BLEU
    ax1.barh(configs, bleu_scores, color='skyblue')
    ax1.set_xlabel('BLEU Score')
    ax1.set_title(f'BLEU Scores - {translation_pair}')
    ax1.grid(axis='x', alpha=0.3)

    # chrF
    ax2.barh(configs, chrf_scores, color='lightgreen')
    ax2.set_xlabel('chrF Score')
    ax2.set_title(f'chrF Scores - {translation_pair}')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output_png = f'./outputs/ablation_lightweight_{translation_pair}.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_png}")
    plt.close()

# Run for bn-hi ONLY first (to test)
print("\n🚀 Starting ablation for bn-hi...")
results_hi = run_lightweight_ablation('bn-hi')

print("\n" + "="*70)
print("✅ bn-hi ablation complete!")
print("="*70)

# Copy to /kaggle/working
import shutil
shutil.copy('./outputs/ablation_lightweight_bn-hi.json', '/kaggle/working/')
shutil.copy('./outputs/ablation_lightweight_bn-hi.png', '/kaggle/working/')

print("\n📥 Files ready in /kaggle/working - click refresh to see them")
print("\n⏭️ If successful, run this cell again changing 'bn-hi' to 'bn-te'")
