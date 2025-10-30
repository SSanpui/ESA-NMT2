"""
Simple 3-Model Comparison: NLLB Baseline vs ESA-NMT vs IndicTrans2
Evaluates on test set only - NO TRAINING!
"""

import os
os.chdir('/kaggle/working/ESA-NMT')

import torch
import json
from emotion_semantic_nmt_enhanced import config, EmotionSemanticNMT, ComprehensiveEvaluator
from dataset_with_annotations import BHT25AnnotatedDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model_name, translation_pair, use_emotion, use_semantic, checkpoint_path=None):
    """
    Evaluate one model on test set
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name} ({translation_pair})")
    print(f"{'='*70}")

    try:
        # Create model
        model = EmotionSemanticNMT(
            config,
            model_type='nllb',
            use_emotion=use_emotion,
            use_semantic=use_semantic,
            use_style=False
        ).to(device)

        # Load checkpoint if provided (for ESA-NMT)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📥 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Checkpoint loaded")
        else:
            print("📝 Using untrained model (baseline)")

        # Load test dataset
        test_dataset = BHT25AnnotatedDataset(
            'BHT25_All.csv',
            model.tokenizer,
            translation_pair,
            config.MAX_LENGTH,
            'test',
            'nllb'
        )

        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        print(f"Test samples: {len(test_dataset)}")

        # Evaluate
        evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
        metrics, preds, refs, sources = evaluator.evaluate(test_loader)

        print(f"\n✅ Results:")
        print(f"  BLEU:             {metrics['bleu']:.2f}")
        print(f"  chrF:             {metrics['chrf']:.2f}")
        print(f"  ROUGE-L:          {metrics['rouge_l']:.2f}")
        if 'emotion_accuracy' in metrics:
            print(f"  Emotion Accuracy: {metrics.get('emotion_accuracy', 0):.2f}%")
        if 'semantic_score' in metrics:
            print(f"  Semantic Score:   {metrics.get('semantic_score', 0):.4f}")

        # Cleanup
        del model
        torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"❌ Error: {e}")
        torch.cuda.empty_cache()
        return None

def create_comparison_table(results, translation_pair):
    """Create comparison table"""

    print(f"\n{'='*70}")
    print(f"COMPARISON TABLE: {translation_pair.upper()}")
    print(f"{'='*70}\n")

    # Header
    print(f"{'Model':<25} {'BLEU':<8} {'chrF':<8} {'ROUGE-L':<10} {'Emotion':<10} {'Semantic':<10}")
    print(f"{'-'*75}")

    for model_name, metrics in results.items():
        if metrics:
            bleu = f"{metrics['bleu']:.2f}"
            chrf = f"{metrics['chrf']:.2f}"
            rouge = f"{metrics['rouge_l']:.2f}"
            emotion = f"{metrics.get('emotion_accuracy', 0):.2f}%" if 'emotion_accuracy' in metrics else "N/A"
            semantic = f"{metrics.get('semantic_score', 0):.4f}" if 'semantic_score' in metrics else "N/A"

            print(f"{model_name:<25} {bleu:<8} {chrf:<8} {rouge:<10} {emotion:<10} {semantic:<10}")

    print()

    # Save to JSON
    os.makedirs('./outputs', exist_ok=True)
    output_file = f'./outputs/model_comparison_{translation_pair}.json'

    json_results = {}
    for model_name, metrics in results.items():
        if metrics:
            json_results[model_name] = {
                'bleu': float(metrics['bleu']),
                'chrf': float(metrics['chrf']),
                'rouge_l': float(metrics['rouge_l']),
                'emotion_accuracy': float(metrics.get('emotion_accuracy', 0)),
                'semantic_score': float(metrics.get('semantic_score', 0))
            }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"💾 Saved: {output_file}")

    return json_results

# =============================================================================
# RUN COMPARISON FOR ONE PAIR
# =============================================================================

TRANSLATION_PAIR = 'bn-hi'  # Change to 'bn-te' for second run

print(f"\n🔍 Starting 3-Model Comparison for {TRANSLATION_PAIR}")
print(f"{'='*70}\n")

results = {}

# Model 1: NLLB Baseline (no modules, no training)
print("\n1️⃣ NLLB Baseline (no emotion/semantic modules)")
baseline_metrics = evaluate_model(
    model_name="NLLB Baseline",
    translation_pair=TRANSLATION_PAIR,
    use_emotion=False,
    use_semantic=False,
    checkpoint_path=None  # No checkpoint = untrained baseline
)
if baseline_metrics:
    results["NLLB Baseline"] = baseline_metrics

# Model 2: ESA-NMT (your full model with trained checkpoint)
print("\n2️⃣ ESA-NMT (Full Model)")
checkpoint = f'/kaggle/working/model_{TRANSLATION_PAIR}.pt'

if not os.path.exists(checkpoint):
    # Try alternative path
    checkpoint = f'./checkpoints/final_model_nllb_{TRANSLATION_PAIR}.pt'

if os.path.exists(checkpoint):
    esa_metrics = evaluate_model(
        model_name="ESA-NMT (Proposed)",
        translation_pair=TRANSLATION_PAIR,
        use_emotion=True,
        use_semantic=True,
        checkpoint_path=checkpoint
    )
    if esa_metrics:
        results["ESA-NMT (Proposed)"] = esa_metrics
else:
    print(f"⚠️ ESA-NMT checkpoint not found at: {checkpoint}")
    print(f"   Looking for: /kaggle/working/model_{TRANSLATION_PAIR}.pt")
    print(f"   Make sure you downloaded the trained model!")

# Model 3: IndicTrans2 (optional - skip if auth issues)
print("\n3️⃣ IndicTrans2 (Optional)")
print("   Skipping - requires authentication")
print("   You can add manually later if needed")

# Create comparison table
print("\n" + "="*70)
print("GENERATING COMPARISON TABLE")
print("="*70)

comparison = create_comparison_table(results, TRANSLATION_PAIR)

# Copy to /kaggle/working
import shutil
output_file = f'./outputs/model_comparison_{TRANSLATION_PAIR}.json'
if os.path.exists(output_file):
    shutil.copy(output_file, '/kaggle/working/')
    print(f"\n✅ Comparison saved to /kaggle/working/")
    print(f"   Click refresh to download!")

print("\n" + "="*70)
print("✅ COMPARISON COMPLETE!")
print("="*70)
print(f"\nResults for {TRANSLATION_PAIR}:")
for model_name in results.keys():
    print(f"  ✅ {model_name}")

print(f"\n⏭️ Next: Change TRANSLATION_PAIR to 'bn-te' and run again")
