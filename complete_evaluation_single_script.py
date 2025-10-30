"""
COMPLETE EVALUATION SCRIPT
- 3-Model Comparison: NLLB Baseline vs ESA-NMT vs IndicTrans2
- Ablation Study: Base, +Emotion, +Semantic, +Both
- All Metrics: BLEU, METEOR, chrF, ROUGE-L, Emotion, Semantic

Copy this entire script into ONE Kaggle cell and run it!
"""

import os
import sys
import torch
import gc
import json
import numpy as np
from tqdm import tqdm

# Change to working directory
os.chdir('/kaggle/working/ESA-NMT')

print("="*80)
print("COMPLETE ESA-NMT EVALUATION")
print("="*80)

# =============================================================================
# SETUP
# =============================================================================

print("\n1️⃣ Setting up dependencies...")

# Install METEOR
try:
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.translate.meteor_score import meteor_score
    METEOR_AVAILABLE = True
    print("   ✅ METEOR ready")
except:
    METEOR_AVAILABLE = False
    print("   ⚠️ METEOR not available")

# Import project modules
from emotion_semantic_nmt_enhanced import config, EmotionSemanticNMT, ComprehensiveEvaluator
from dataset_with_annotations import BHT25AnnotatedDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   ✅ Device: {device}")

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_single_model(model_name, translation_pair, model_type, use_emotion, use_semantic, checkpoint_path=None):
    """
    Evaluate one model configuration
    Returns metrics dict or None if failed
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name} ({translation_pair})")
    print(f"{'='*80}")

    try:
        # Skip IndicTrans2 if model_type is indictrans2 - we'll handle it separately
        if model_type == 'indictrans2':
            print("⚠️ IndicTrans2 requires special handling - skipping for now")
            return None

        # Create model
        print(f"Creating {model_type} model (emotion={use_emotion}, semantic={use_semantic})...")
        model = EmotionSemanticNMT(
            config,
            model_type=model_type,
            use_emotion=use_emotion,
            use_semantic=use_semantic,
            use_style=False
        ).to(device)

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📥 Loading checkpoint: {os.path.basename(checkpoint_path)}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("   ✅ Checkpoint loaded")
        else:
            if checkpoint_path:
                print(f"   ⚠️ Checkpoint not found: {checkpoint_path}")
            print("   Using pre-trained model")

        # Load test dataset
        print("Loading test dataset...")
        test_dataset = BHT25AnnotatedDataset(
            'BHT25_All.csv',
            model.tokenizer,
            translation_pair,
            config.MAX_LENGTH,
            'test',
            model_type
        )

        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        print(f"   ✅ Test samples: {len(test_dataset)}")

        # Evaluate
        print("Evaluating...")
        evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
        metrics, preds, refs, sources = evaluator.evaluate(test_loader)

        # Print results
        print(f"\n✅ Results:")
        print(f"   BLEU:    {metrics.get('bleu', 0):.2f}")
        print(f"   METEOR:  {metrics.get('meteor', 0):.2f}")
        print(f"   chrF:    {metrics.get('chrf', 0):.2f}")
        print(f"   ROUGE-L: {metrics.get('rouge_l', 0):.2f}")
        if use_emotion:
            print(f"   Emotion: {metrics.get('emotion_accuracy', 0):.2f}%")
        if use_semantic:
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

        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
        return None

# =============================================================================
# PART 1: 3-MODEL COMPARISON
# =============================================================================

def run_3model_comparison(translation_pair):
    """Compare NLLB Baseline, ESA-NMT, IndicTrans2"""

    print(f"\n{'='*80}")
    print(f"PART 1: 3-MODEL COMPARISON ({translation_pair})")
    print(f"{'='*80}")

    results = {}

    # Model 1: NLLB Baseline
    print("\n[1/3] NLLB Baseline (pre-trained, no modules)")
    baseline = evaluate_single_model(
        model_name="NLLB Baseline",
        translation_pair=translation_pair,
        model_type='nllb',
        use_emotion=False,
        use_semantic=False,
        checkpoint_path=None
    )
    if baseline:
        results["NLLB Baseline"] = baseline

    # Model 2: ESA-NMT (your trained model)
    print("\n[2/3] ESA-NMT (Full Model)")

    # Find checkpoint
    checkpoint_paths = [
        f'/kaggle/working/model_{translation_pair}.pt',
        f'./checkpoints/final_model_nllb_{translation_pair}.pt'
    ]

    checkpoint = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint = path
            break

    if checkpoint:
        esa = evaluate_single_model(
            model_name="ESA-NMT",
            translation_pair=translation_pair,
            model_type='nllb',
            use_emotion=True,
            use_semantic=True,
            checkpoint_path=checkpoint
        )
        if esa:
            results["ESA-NMT"] = esa
    else:
        print(f"⚠️ ESA-NMT checkpoint not found. Searched:")
        for path in checkpoint_paths:
            print(f"   - {path}")

    # Model 3: IndicTrans2 - Skip for now due to config issues
    print("\n[3/3] IndicTrans2")
    print("⚠️ Skipping IndicTrans2 due to configuration issues")
    print("   You can add this manually later if needed")

    return results

# =============================================================================
# PART 2: ABLATION STUDY
# =============================================================================

def run_ablation_study(translation_pair):
    """Run ablation: Base, +Emotion, +Semantic, +Both"""

    print(f"\n{'='*80}")
    print(f"PART 2: ABLATION STUDY ({translation_pair})")
    print(f"{'='*80}")

    configs = [
        {'name': 'Base NLLB', 'emotion': False, 'semantic': False},
        {'name': 'Base + Emotion', 'emotion': True, 'semantic': False},
        {'name': 'Base + Semantic', 'emotion': False, 'semantic': True},
        {'name': 'Full (Both)', 'emotion': True, 'semantic': True},
    ]

    results = {}

    # Find ESA-NMT checkpoint for Full model
    checkpoint_paths = [
        f'/kaggle/working/model_{translation_pair}.pt',
        f'./checkpoints/final_model_nllb_{translation_pair}.pt'
    ]

    full_checkpoint = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            full_checkpoint = path
            break

    for i, conf in enumerate(configs):
        print(f"\n[{i+1}/4] Testing: {conf['name']}")

        # Use checkpoint only for Full model
        checkpoint = full_checkpoint if (conf['emotion'] and conf['semantic']) else None

        metrics = evaluate_single_model(
            model_name=conf['name'],
            translation_pair=translation_pair,
            model_type='nllb',
            use_emotion=conf['emotion'],
            use_semantic=conf['semantic'],
            checkpoint_path=checkpoint
        )

        if metrics:
            results[conf['name']] = metrics

        # Clear memory between configs
        torch.cuda.empty_cache()
        gc.collect()

    return results

# =============================================================================
# RESULTS FORMATTING
# =============================================================================

def print_comparison_table(results, title):
    """Print formatted results table"""

    print(f"\n{'='*90}")
    print(f"{title}")
    print(f"{'='*90}")

    header = f"{'Model':<20} {'BLEU':<8} {'METEOR':<8} {'chrF':<8} {'ROUGE-L':<10} {'Emotion':<10} {'Semantic':<10}"
    print(header)
    print("-"*90)

    for model_name, metrics in results.items():
        bleu = f"{metrics.get('bleu', 0):.2f}"
        meteor = f"{metrics.get('meteor', 0):.2f}"
        chrf = f"{metrics.get('chrf', 0):.2f}"
        rouge = f"{metrics.get('rouge_l', 0):.2f}"
        emotion = f"{metrics.get('emotion_accuracy', 0):.2f}%" if metrics.get('emotion_accuracy', 0) > 0 else "N/A"
        semantic = f"{metrics.get('semantic_score', 0):.4f}" if metrics.get('semantic_score', 0) > 0 else "N/A"

        print(f"{model_name:<20} {bleu:<8} {meteor:<8} {chrf:<8} {rouge:<10} {emotion:<10} {semantic:<10}")

    print()

def save_results(results, filename):
    """Save results to JSON"""

    json_results = {}
    for model_name, metrics in results.items():
        json_results[model_name] = {
            'bleu': float(metrics.get('bleu', 0)),
            'meteor': float(metrics.get('meteor', 0)),
            'chrf': float(metrics.get('chrf', 0)),
            'rouge_l': float(metrics.get('rouge_l', 0)),
            'emotion_accuracy': float(metrics.get('emotion_accuracy', 0)),
            'semantic_score': float(metrics.get('semantic_score', 0))
        }

    os.makedirs('./outputs', exist_ok=True)
    filepath = f'./outputs/{filename}'

    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"💾 Saved: {filepath}")

    # Copy to /kaggle/working
    import shutil
    shutil.copy(filepath, f'/kaggle/working/{filename}')
    print(f"💾 Copied to: /kaggle/working/{filename}")

    return filepath

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    TRANSLATION_PAIR = 'bn-hi'  # Change to 'bn-te' for Telugu

    print(f"\n🚀 Starting complete evaluation for: {TRANSLATION_PAIR}")
    print(f"   This will take approximately 30-40 minutes")

    # Part 1: 3-Model Comparison
    comparison_results = run_3model_comparison(TRANSLATION_PAIR)

    if comparison_results:
        print_comparison_table(comparison_results, f"3-MODEL COMPARISON: {TRANSLATION_PAIR.upper()}")
        save_results(comparison_results, f'comparison_3models_{TRANSLATION_PAIR}.json')

    # Part 2: Ablation Study
    ablation_results = run_ablation_study(TRANSLATION_PAIR)

    if ablation_results:
        print_comparison_table(ablation_results, f"ABLATION STUDY: {TRANSLATION_PAIR.upper()}")
        save_results(ablation_results, f'ablation_study_{TRANSLATION_PAIR}.json')

    # Final summary
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)

    print("\n📊 Results Summary:")
    print(f"\n3-Model Comparison ({len(comparison_results)} models):")
    for name in comparison_results.keys():
        print(f"   ✅ {name}")

    print(f"\nAblation Study ({len(ablation_results)} configurations):")
    for name in ablation_results.keys():
        print(f"   ✅ {name}")

    print("\n📥 Download these files from /kaggle/working:")
    print(f"   - comparison_3models_{TRANSLATION_PAIR}.json")
    print(f"   - ablation_study_{TRANSLATION_PAIR}.json")

    print(f"\n⏭️ Next: Change TRANSLATION_PAIR to 'bn-te' and run again for Telugu results")

    print("\n" + "="*80)
