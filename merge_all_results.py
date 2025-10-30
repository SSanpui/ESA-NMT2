"""
Merge All Results into Final Comparison Tables

Combines:
1. NLLB Baseline results
2. ESA-NMT results
3. IndicTrans2 results (if available)
4. Ablation study results

Creates final comparison tables for your paper
"""

import json
import os

os.chdir('/kaggle/working/ESA-NMT')

TRANSLATION_PAIR = input("Enter translation pair (bn-hi or bn-te): ").strip() or 'bn-hi'

print("="*90)
print(f"Merging All Results for {TRANSLATION_PAIR.upper()}")
print("="*90)

# =============================================================================
# Load All Results
# =============================================================================

results = {}

# 1. Load main comparison (NLLB Baseline + ESA-NMT)
main_file = f'./outputs/comparison_3models_{TRANSLATION_PAIR}.json'
if os.path.exists(main_file):
    with open(main_file, 'r') as f:
        main_results = json.load(f)
        results.update(main_results)
    print(f"✅ Loaded main comparison: {list(main_results.keys())}")
else:
    print(f"⚠️ Main comparison not found: {main_file}")

# 2. Load IndicTrans2 results
indic_file = f'./outputs/indictrans2_results_{TRANSLATION_PAIR}.json'
indic_file_working = f'/kaggle/working/indictrans2_results_{TRANSLATION_PAIR}.json'

indic_results = None
if os.path.exists(indic_file):
    with open(indic_file, 'r') as f:
        indic_data = json.load(f)
        indic_results = {
            'bleu': indic_data['bleu'],
            'meteor': indic_data['meteor'],
            'chrf': indic_data['chrf'],
            'rouge_l': indic_data['rouge_l'],
            'emotion_accuracy': 0,
            'semantic_score': 0
        }
        results['IndicTrans2'] = indic_results
    print(f"✅ Loaded IndicTrans2 results")
elif os.path.exists(indic_file_working):
    with open(indic_file_working, 'r') as f:
        indic_data = json.load(f)
        indic_results = {
            'bleu': indic_data['bleu'],
            'meteor': indic_data['meteor'],
            'chrf': indic_data['chrf'],
            'rouge_l': indic_data['rouge_l'],
            'emotion_accuracy': 0,
            'semantic_score': 0
        }
        results['IndicTrans2'] = indic_results
    print(f"✅ Loaded IndicTrans2 results from /kaggle/working")
else:
    print(f"⚠️ IndicTrans2 results not found (optional)")

# 3. Load ablation study
ablation_file = f'./outputs/ablation_study_{TRANSLATION_PAIR}.json'
ablation_results = {}
if os.path.exists(ablation_file):
    with open(ablation_file, 'r') as f:
        ablation_results = json.load(f)
    print(f"✅ Loaded ablation study: {list(ablation_results.keys())}")
else:
    print(f"⚠️ Ablation study not found: {ablation_file}")

# =============================================================================
# Create Final Comparison Table
# =============================================================================

print("\n" + "="*90)
print(f"FINAL 3-MODEL COMPARISON: {TRANSLATION_PAIR.upper()}")
print("="*90)

header = f"{'Model':<25} {'BLEU':<8} {'METEOR':<8} {'chrF':<8} {'ROUGE-L':<10} {'Emotion':<10} {'Semantic':<10}"
print(header)
print("-"*90)

# Define order for better presentation
model_order = ['NLLB Baseline', 'IndicTrans2', 'ESA-NMT']

for model_name in model_order:
    if model_name in results:
        metrics = results[model_name]
        bleu = f"{metrics.get('bleu', 0):.2f}"
        meteor = f"{metrics.get('meteor', 0):.2f}"
        chrf = f"{metrics.get('chrf', 0):.2f}"
        rouge = f"{metrics.get('rouge_l', 0):.2f}"
        emotion = f"{metrics.get('emotion_accuracy', 0):.2f}%" if metrics.get('emotion_accuracy', 0) > 0 else "N/A"
        semantic = f"{metrics.get('semantic_score', 0):.4f}" if metrics.get('semantic_score', 0) > 0 else "N/A"

        print(f"{model_name:<25} {bleu:<8} {meteor:<8} {chrf:<8} {rouge:<10} {emotion:<10} {semantic:<10}")

print()

# =============================================================================
# Create Ablation Table
# =============================================================================

if ablation_results:
    print("\n" + "="*90)
    print(f"ABLATION STUDY: {TRANSLATION_PAIR.upper()}")
    print("="*90)

    print(header)
    print("-"*90)

    ablation_order = ['Base NLLB', 'Base + Emotion', 'Base + Semantic', 'Full (Both)']

    for config_name in ablation_order:
        if config_name in ablation_results:
            metrics = ablation_results[config_name]
            bleu = f"{metrics.get('bleu', 0):.2f}"
            meteor = f"{metrics.get('meteor', 0):.2f}"
            chrf = f"{metrics.get('chrf', 0):.2f}"
            rouge = f"{metrics.get('rouge_l', 0):.2f}"
            emotion = f"{metrics.get('emotion_accuracy', 0):.2f}%" if metrics.get('emotion_accuracy', 0) > 0 else "N/A"
            semantic = f"{metrics.get('semantic_score', 0):.4f}" if metrics.get('semantic_score', 0) > 0 else "N/A"

            print(f"{config_name:<25} {bleu:<8} {meteor:<8} {chrf:<8} {rouge:<10} {emotion:<10} {semantic:<10}")

    print()

# =============================================================================
# Save Final Merged Results
# =============================================================================

final_results = {
    'translation_pair': TRANSLATION_PAIR,
    'model_comparison': results,
    'ablation_study': ablation_results
}

output_file = f'./outputs/final_complete_results_{TRANSLATION_PAIR}.json'
with open(output_file, 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"💾 Saved merged results: {output_file}")

# Copy to /kaggle/working
import shutil
shutil.copy(output_file, f'/kaggle/working/final_complete_results_{TRANSLATION_PAIR}.json')
print(f"💾 Copied to: /kaggle/working/final_complete_results_{TRANSLATION_PAIR}.json")

print("\n" + "="*90)
print("✅ FINAL RESULTS READY!")
print("="*90)

print("\n📥 Download this file:")
print(f"   final_complete_results_{TRANSLATION_PAIR}.json")

print("\n📊 Results include:")
print(f"   ✅ {len(results)} models in comparison")
if ablation_results:
    print(f"   ✅ {len(ablation_results)} ablation configurations")

print("\n🎉 All evaluation complete! Use these tables in your paper.")
