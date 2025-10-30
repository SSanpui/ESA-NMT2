"""
Complete 3-Model Comparison with METEOR Score and IndicTrans2
Includes: NLLB Baseline, ESA-NMT, IndicTrans2
"""

import os
os.chdir('/kaggle/working/ESA-NMT')

import torch
import json
from emotion_semantic_nmt_enhanced import config, EmotionSemanticNMT, ComprehensiveEvaluator
from dataset_with_annotations import BHT25AnnotatedDataset
from torch.utils.data import DataLoader

# =============================================================================
# STEP 1: Setup METEOR Score
# =============================================================================

print("📦 Setting up METEOR score...")
try:
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.translate.meteor_score import meteor_score
    print("✅ METEOR ready")
    METEOR_AVAILABLE = True
except:
    print("⚠️ METEOR not available (will skip)")
    METEOR_AVAILABLE = False

# =============================================================================
# STEP 2: Setup Hugging Face Authentication for IndicTrans2
# =============================================================================

print("\n🔐 Setting up IndicTrans2 authentication...")
print("To use IndicTrans2, you need a Hugging Face token:")
print("1. Go to https://huggingface.co/settings/tokens")
print("2. Create a token (read access)")
print("3. Request access to: https://huggingface.co/ai4bharat/indictrans2-en-indic-1B")
print("4. Paste token below (or press Enter to skip IndicTrans2)")

HF_TOKEN = input("\nEnter your Hugging Face token (or Enter to skip): ").strip()

if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
    print("✅ Token set")
else:
    print("⚠️ No token - will skip IndicTrans2")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# STEP 3: Evaluation Function
# =============================================================================

def evaluate_model(model_name, translation_pair, model_type, use_emotion, use_semantic, checkpoint_path=None):
    """
    Evaluate one model on test set with ALL metrics including METEOR
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name} ({translation_pair})")
    print(f"{'='*70}")

    try:
        # Create model
        model = EmotionSemanticNMT(
            config,
            model_type=model_type,
            use_emotion=use_emotion,
            use_semantic=use_semantic,
            use_style=False
        ).to(device)

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📥 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Checkpoint loaded")
        else:
            print("📝 Using pre-trained model (baseline)")

        # Load test dataset
        test_dataset = BHT25AnnotatedDataset(
            'BHT25_All.csv',
            model.tokenizer,
            translation_pair,
            config.MAX_LENGTH,
            'test',
            model_type
        )

        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        print(f"Test samples: {len(test_dataset)}")

        # Evaluate
        evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
        metrics, preds, refs, sources = evaluator.evaluate(test_loader)

        print(f"\n✅ Results:")
        print(f"  BLEU:             {metrics.get('bleu', 0):.2f}")
        print(f"  METEOR:           {metrics.get('meteor', 0):.2f}")
        print(f"  chrF:             {metrics.get('chrf', 0):.2f}")
        print(f"  ROUGE-L:          {metrics.get('rouge_l', 0):.2f}")
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
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        return None

# =============================================================================
# STEP 4: Create Comparison Table
# =============================================================================

def create_comparison_table(results, translation_pair):
    """Create detailed comparison table with METEOR"""

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
            emotion = f"{metrics.get('emotion_accuracy', 0):.2f}%" if 'emotion_accuracy' in metrics else "N/A"
            semantic = f"{metrics.get('semantic_score', 0):.4f}" if 'semantic_score' in metrics else "N/A"

            print(f"{model_name:<25} {bleu:<8} {meteor:<8} {chrf:<8} {rouge:<10} {emotion:<10} {semantic:<10}")

    print()

    # Save to JSON
    os.makedirs('./outputs', exist_ok=True)
    output_file = f'./outputs/complete_comparison_{translation_pair}.json'

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
# STEP 5: RUN COMPARISON
# =============================================================================

TRANSLATION_PAIR = 'bn-hi'  # Change to 'bn-te' for second run

print(f"\n🔍 Starting Complete 3-Model Comparison for {TRANSLATION_PAIR}")
print(f"{'='*70}\n")

results = {}

# Model 1: NLLB Baseline
print("\n1️⃣ NLLB Baseline (no emotion/semantic modules)")
baseline_metrics = evaluate_model(
    model_name="NLLB Baseline",
    translation_pair=TRANSLATION_PAIR,
    model_type='nllb',
    use_emotion=False,
    use_semantic=False,
    checkpoint_path=None
)
if baseline_metrics:
    results["NLLB Baseline"] = baseline_metrics

# Model 2: ESA-NMT (your full model)
print("\n2️⃣ ESA-NMT (Full Model)")
checkpoint = f'/kaggle/working/model_{TRANSLATION_PAIR}.pt'

if not os.path.exists(checkpoint):
    checkpoint = f'./checkpoints/final_model_nllb_{TRANSLATION_PAIR}.pt'

if os.path.exists(checkpoint):
    esa_metrics = evaluate_model(
        model_name="ESA-NMT (Proposed)",
        translation_pair=TRANSLATION_PAIR,
        model_type='nllb',
        use_emotion=True,
        use_semantic=True,
        checkpoint_path=checkpoint
    )
    if esa_metrics:
        results["ESA-NMT (Proposed)"] = esa_metrics
else:
    print(f"⚠️ ESA-NMT checkpoint not found at: {checkpoint}")
    print(f"   Make sure you have the trained model!")

# Model 3: IndicTrans2
print("\n3️⃣ IndicTrans2")
if HF_TOKEN:
    print("Attempting to load IndicTrans2...")
    try:
        # Login to Hugging Face
        from huggingface_hub import login
        login(token=HF_TOKEN)

        indic_metrics = evaluate_model(
            model_name="IndicTrans2",
            translation_pair=TRANSLATION_PAIR,
            model_type='indictrans2',
            use_emotion=False,
            use_semantic=False,
            checkpoint_path=None
        )
        if indic_metrics:
            results["IndicTrans2"] = indic_metrics
    except Exception as e:
        print(f"⚠️ IndicTrans2 failed: {e}")
        print("   Make sure you requested access at:")
        print("   https://huggingface.co/ai4bharat/indictrans2-en-indic-1B")
else:
    print("   Skipped (no token provided)")

# Create comparison table
print("\n" + "="*70)
print("GENERATING COMPARISON TABLE")
print("="*70)

comparison = create_comparison_table(results, TRANSLATION_PAIR)

# Copy to /kaggle/working
import shutil
output_file = f'./outputs/complete_comparison_{TRANSLATION_PAIR}.json'
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
