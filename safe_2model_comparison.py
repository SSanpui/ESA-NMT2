"""
Safe 2-Model Comparison: NLLB Baseline vs ESA-NMT
With METEOR score, no IndicTrans2 complications
"""

import os
os.chdir('/kaggle/working/ESA-NMT')

import torch
import json
import gc
from emotion_semantic_nmt_enhanced import config, EmotionSemanticNMT, ComprehensiveEvaluator
from dataset_with_annotations import BHT25AnnotatedDataset
from torch.utils.data import DataLoader

# Setup METEOR
print("📦 Setting up METEOR score...")
try:
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("✅ METEOR ready")
except:
    print("⚠️ METEOR setup failed (will skip)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model_name, translation_pair, use_emotion, use_semantic, checkpoint_path=None):
    """Evaluate one model on test set"""

    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name} ({translation_pair})")
    print(f"{'='*70}")

    try:
        # Create model - ALWAYS use NLLB
        print(f"Creating model with emotion={use_emotion}, semantic={use_semantic}")

        model = EmotionSemanticNMT(
            config,
            model_type='nllb',  # ← ALWAYS NLLB (not indictrans2)
            use_emotion=use_emotion,
            use_semantic=use_semantic,
            use_style=False
        ).to(device)

        print(f"✅ Model created on {device}")

        # Load checkpoint if provided
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                print(f"📥 Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Checkpoint loaded")
            else:
                print(f"⚠️ Checkpoint not found: {checkpoint_path}")
                print("   Using untrained model as baseline")
        else:
            print("📝 Using pre-trained NLLB (baseline)")

        # Load test dataset
        print("Loading test dataset...")
        test_dataset = BHT25AnnotatedDataset(
            'BHT25_All.csv',
            model.tokenizer,
            translation_pair,
            config.MAX_LENGTH,
            'test',
            'nllb'  # ← ALWAYS NLLB tokenizer
        )

        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        print(f"✅ Test samples: {len(test_dataset)}")

        # Evaluate
        print("Evaluating...")
        evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
        metrics, preds, refs, sources = evaluator.evaluate(test_loader)

        print(f"\n✅ {model_name} Results:")
        print(f"  BLEU:             {metrics.get('bleu', 0):.2f}")
        print(f"  METEOR:           {metrics.get('meteor', 0):.2f}")
        print(f"  chrF:             {metrics.get('chrf', 0):.2f}")
        print(f"  ROUGE-L:          {metrics.get('rouge_l', 0):.2f}")

        if use_emotion and 'emotion_accuracy' in metrics:
            print(f"  Emotion Accuracy: {metrics.get('emotion_accuracy', 0):.2f}%")
        if use_semantic and 'semantic_score' in metrics:
            print(f"  Semantic Score:   {metrics.get('semantic_score', 0):.4f}")

        # Cleanup
        del model
        del evaluator
        torch.cuda.empty_cache()
        gc.collect()

        print(f"🧹 Memory cleaned")

        return metrics

    except Exception as e:
        print(f"❌ Error in {model_name}: {e}")
        import traceback
        traceback.print_exc()

        # Cleanup on error
        try:
            del model
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()

        return None

def create_comparison_table(results, translation_pair):
    """Create comparison table"""

    print(f"\n{'='*80}")
    print(f"COMPARISON TABLE: {translation_pair.upper()}")
    print(f"{'='*80}\n")

    # Header
    print(f"{'Model':<25} {'BLEU':<8} {'METEOR':<8} {'chrF':<8} {'ROUGE-L':<10} {'Emotion':<10} {'Semantic':<10}")
    print(f"{'-'*85}")

    for model_name, metrics in results.items():
        if metrics:
            bleu = f"{metrics.get('bleu', 0):.2f}"
            meteor = f"{metrics.get('meteor', 0):.2f}"
            chrf = f"{metrics.get('chrf', 0):.2f}"
            rouge = f"{metrics.get('rouge_l', 0):.2f}"
            emotion = f"{metrics.get('emotion_accuracy', 0):.2f}%" if metrics.get('emotion_accuracy', 0) > 0 else "N/A"
            semantic = f"{metrics.get('semantic_score', 0):.4f}" if metrics.get('semantic_score', 0) > 0 else "N/A"

            print(f"{model_name:<25} {bleu:<8} {meteor:<8} {chrf:<8} {rouge:<10} {emotion:<10} {semantic:<10}")

    print()

    # Save to JSON
    os.makedirs('./outputs', exist_ok=True)
    output_file = f'./outputs/comparison_nllb_vs_esa_{translation_pair}.json'

    json_results = {}
    for model_name, metrics in results.items():
        if metrics:
            json_results[model_name] = {
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
    return json_results

# =============================================================================
# RUN COMPARISON
# =============================================================================

TRANSLATION_PAIR = 'bn-hi'  # Change to 'bn-te' for second run

print(f"\n🔍 2-Model Comparison: NLLB Baseline vs ESA-NMT")
print(f"Translation pair: {TRANSLATION_PAIR}")
print(f"{'='*70}\n")

# Clear memory first
torch.cuda.empty_cache()
gc.collect()

results = {}

# Model 1: NLLB Baseline (no modules)
print("\n" + "="*70)
print("MODEL 1: NLLB Baseline")
print("="*70)

baseline_metrics = evaluate_model(
    model_name="NLLB Baseline",
    translation_pair=TRANSLATION_PAIR,
    use_emotion=False,
    use_semantic=False,
    checkpoint_path=None  # No checkpoint = pre-trained NLLB
)

if baseline_metrics:
    results["NLLB Baseline"] = baseline_metrics
    print("✅ Baseline evaluation complete")
else:
    print("❌ Baseline evaluation failed")

# Clear memory between models
print("\n🧹 Clearing memory between models...")
torch.cuda.empty_cache()
gc.collect()

# Model 2: ESA-NMT (your trained model)
print("\n" + "="*70)
print("MODEL 2: ESA-NMT (Full Model with Emotion + Semantic)")
print("="*70)

# Find checkpoint
checkpoint_paths = [
    f'/kaggle/working/model_{TRANSLATION_PAIR}.pt',
    f'./checkpoints/final_model_nllb_{TRANSLATION_PAIR}.pt',
    f'/kaggle/working/ESA-NMT/checkpoints/final_model_nllb_{TRANSLATION_PAIR}.pt'
]

checkpoint = None
for path in checkpoint_paths:
    if os.path.exists(path):
        checkpoint = path
        print(f"✅ Found checkpoint: {path}")
        break

if checkpoint:
    esa_metrics = evaluate_model(
        model_name="ESA-NMT (Proposed)",
        translation_pair=TRANSLATION_PAIR,
        use_emotion=True,
        use_semantic=True,
        checkpoint_path=checkpoint
    )

    if esa_metrics:
        results["ESA-NMT (Proposed)"] = esa_metrics
        print("✅ ESA-NMT evaluation complete")
    else:
        print("❌ ESA-NMT evaluation failed")
else:
    print("❌ ESA-NMT checkpoint not found!")
    print("   Searched:")
    for path in checkpoint_paths:
        print(f"   - {path}")
    print("\n   Please make sure you have the trained model checkpoint!")

# Create comparison table
if len(results) > 0:
    print("\n" + "="*70)
    print("GENERATING COMPARISON TABLE")
    print("="*70)

    comparison = create_comparison_table(results, TRANSLATION_PAIR)

    # Copy to /kaggle/working
    import shutil
    output_file = f'./outputs/comparison_nllb_vs_esa_{TRANSLATION_PAIR}.json'
    if os.path.exists(output_file):
        shutil.copy(output_file, '/kaggle/working/')
        print(f"\n✅ Results copied to /kaggle/working/")
        print(f"   Click refresh button to download!")

    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nEvaluated models:")
    for model_name in results.keys():
        print(f"  ✅ {model_name}")

    print(f"\n⏭️ Next Steps:")
    print(f"   1. Download the JSON file")
    print(f"   2. Change TRANSLATION_PAIR to 'bn-te' and run again")
    print(f"   3. (Optional) Add IndicTrans2 later if needed")
else:
    print("\n❌ No models evaluated successfully!")
    print("   Check the errors above")
