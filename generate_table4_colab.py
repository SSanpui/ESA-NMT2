# ============================================================================
# GENERATE TABLE 4 - Run this in Google Colab
# Compares: Baseline NLLB vs Your ESA-NMT Model
# Generates publication-ready comparison table
# ============================================================================

print("🚀 Generating Table 4: Comprehensive Comparison")
print("="*60)

from emotion_semantic_nmt_enhanced import (
    Config, BHT25Dataset, EmotionSemanticNMT,
    ComprehensiveEvaluator, Trainer
)
import torch
from torch.utils.data import DataLoader
import pandas as pd

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Store all results
all_results = {}

# ============================================================================
# STEP 1: Train Baseline Models (without custom modules)
# ============================================================================

for pair in ['bn-hi', 'bn-te']:
    print(f"\n{'='*60}")
    print(f"BASELINE MODEL: {pair.upper()}")
    print(f"{'='*60}")

    # Create baseline model (NO emotion/semantic/style modules)
    baseline_model = EmotionSemanticNMT(
        config,
        model_type='nllb',
        use_emotion=False,  # ← NO emotion module
        use_semantic=False,  # ← NO semantic module
        use_style=False      # ← NO style module
    ).to(device)

    # Prepare data
    train_dataset = BHT25Dataset('BHT25_All.csv', baseline_model.tokenizer, pair,
                                config.MAX_LENGTH, 'train', 'nllb')
    test_dataset = BHT25Dataset('BHT25_All.csv', baseline_model.tokenizer, pair,
                               config.MAX_LENGTH, 'test', 'nllb')

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # Train
    trainer = Trainer(baseline_model, config, pair)
    final_loss = 0

    for epoch in range(config.EPOCHS['phase1']):
        loss = trainer.train_epoch(train_loader, epoch)
        final_loss = loss
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")

    # Evaluate
    evaluator = ComprehensiveEvaluator(baseline_model, baseline_model.tokenizer, config, pair)
    metrics, _, _, _ = evaluator.evaluate(test_loader)
    metrics['final_training_loss'] = final_loss
    metrics['memory_usage_gb'] = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

    all_results[f'baseline_{pair}'] = metrics

    print(f"✅ Baseline {pair} - BLEU: {metrics['bleu']:.2f}")

    # Clean up
    del baseline_model, trainer, evaluator
    torch.cuda.empty_cache()

# ============================================================================
# STEP 2: Train Proposed Models (WITH all custom modules)
# ============================================================================

for pair in ['bn-hi', 'bn-te']:
    print(f"\n{'='*60}")
    print(f"PROPOSED ESA-NMT: {pair.upper()}")
    print(f"{'='*60}")

    # Create proposed model (WITH emotion/semantic/style modules)
    proposed_model = EmotionSemanticNMT(
        config,
        model_type='nllb',
        use_emotion=True,   # ✅ WITH emotion module
        use_semantic=True,  # ✅ WITH semantic module
        use_style=True      # ✅ WITH style module
    ).to(device)

    # Prepare data
    train_dataset = BHT25Dataset('BHT25_All.csv', proposed_model.tokenizer, pair,
                                config.MAX_LENGTH, 'train', 'nllb')
    test_dataset = BHT25Dataset('BHT25_All.csv', proposed_model.tokenizer, pair,
                               config.MAX_LENGTH, 'test', 'nllb')

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # Train
    trainer = Trainer(proposed_model, config, pair)
    final_loss = 0

    for epoch in range(config.EPOCHS['phase1']):
        loss = trainer.train_epoch(train_loader, epoch)
        final_loss = loss
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")

    # Evaluate
    evaluator = ComprehensiveEvaluator(proposed_model, proposed_model.tokenizer, config, pair)
    metrics, _, _, _ = evaluator.evaluate(test_loader)
    metrics['final_training_loss'] = final_loss
    metrics['memory_usage_gb'] = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

    all_results[f'proposed_{pair}'] = metrics

    print(f"✅ Proposed {pair} - BLEU: {metrics['bleu']:.2f}")

    # Clean up
    del proposed_model, trainer, evaluator
    torch.cuda.empty_cache()

# ============================================================================
# STEP 3: Generate Table 4
# ============================================================================

print("\n" + "="*80)
print("📊 TABLE 4: COMPREHENSIVE EVALUATION RESULTS")
print("="*80)

# Create DataFrame for easy viewing
data = []

for pair in ['bn-hi', 'bn-te']:
    pair_name = "Bengali-Hindi" if pair == 'bn-hi' else "Bengali-Telugu"

    baseline = all_results[f'baseline_{pair}']
    proposed = all_results[f'proposed_{pair}']

    # Translation metrics
    for metric in ['bleu', 'meteor', 'rouge_l', 'chrf']:
        base_val = baseline.get(metric, 0)
        prop_val = proposed.get(metric, 0)
        improvement = ((prop_val - base_val) / base_val * 100) if base_val > 0 else 0

        data.append({
            'Language Pair': pair_name,
            'Metric': metric.upper(),
            'Baseline': f"{base_val:.2f}",
            'Proposed': f"{prop_val:.2f}",
            'Improvement': f"+{improvement:.1f}%"
        })

    # Emotion accuracy
    data.append({
        'Language Pair': pair_name,
        'Metric': 'Emotion Accuracy',
        'Baseline': f"{baseline.get('emotion_accuracy', 0):.1f}%",
        'Proposed': f"{proposed.get('emotion_accuracy', 0):.1f}%",
        'Improvement': '-'
    })

    # Semantic score
    data.append({
        'Language Pair': pair_name,
        'Metric': 'Semantic Score',
        'Baseline': f"{baseline.get('semantic_score', 0):.4f}",
        'Proposed': f"{proposed.get('semantic_score', 0):.4f}",
        'Improvement': '-'
    })

    # Memory usage
    base_mem = baseline.get('memory_usage_gb', 0)
    prop_mem = proposed.get('memory_usage_gb', 0)
    mem_reduction = ((base_mem - prop_mem) / base_mem * 100) if base_mem > 0 else 0

    data.append({
        'Language Pair': pair_name,
        'Metric': 'Memory Usage (GB)',
        'Baseline': f"{base_mem:.2f}",
        'Proposed': f"{prop_mem:.2f}",
        'Improvement': f"-{mem_reduction:.1f}%"
    })

# Display as DataFrame
df = pd.DataFrame(data)
print(df.to_string(index=False))

# Save results
import json
import os
os.makedirs('./outputs', exist_ok=True)

with open('./outputs/table4_results.json', 'w') as f:
    json.dump(ComprehensiveEvaluator.convert_to_json_serializable(all_results), f, indent=2)

# Generate LaTeX table
latex = r"""\begin{table}[htbp]
\centering
\caption{Comprehensive evaluation results: Proposed ESA-NMT model vs. Baseline}
\label{tab:comprehensive_results}
\begin{tabular}{lccccc}
\toprule
\multirow{2}{*}{Metric} & \multicolumn{2}{c}{Bengali-Hindi} & \multicolumn{2}{c}{Bengali-Telugu} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
 & Baseline & Proposed & Baseline & Proposed \\
\midrule
"""

metrics_list = [
    ('BLEU', 'bleu'),
    ('METEOR', 'meteor'),
    ('ROUGE-L', 'rouge_l'),
    ('chrF', 'chrf'),
]

for name, key in metrics_list:
    line = f"{name}"
    for pair in ['bn-hi', 'bn-te']:
        baseline = all_results[f'baseline_{pair}']
        proposed = all_results[f'proposed_{pair}']
        line += f" & {baseline.get(key, 0):.2f} & {proposed.get(key, 0):.2f}"
    line += " \\\\"
    latex += line + "\n"

latex += r"""\midrule
Emotion Acc."""

for pair in ['bn-hi', 'bn-te']:
    baseline = all_results[f'baseline_{pair}']
    proposed = all_results[f'proposed_{pair}']
    latex += f" & {baseline.get('emotion_accuracy', 0):.1f}\\% & {proposed.get('emotion_accuracy', 0):.1f}\\%"

latex += " \\\\\n"

latex += r"""Semantic Score"""
for pair in ['bn-hi', 'bn-te']:
    baseline = all_results[f'baseline_{pair}']
    proposed = all_results[f'proposed_{pair}']
    latex += f" & {baseline.get('semantic_score', 0):.3f} & {proposed.get('semantic_score', 0):.3f}"

latex += r""" \\
\bottomrule
\end{tabular}
\end{table}
"""

with open('./outputs/table4_latex.tex', 'w') as f:
    f.write(latex)

print("\n✅ Results saved:")
print("  - ./outputs/table4_results.json")
print("  - ./outputs/table4_latex.tex")

# Display summary
print("\n" + "="*80)
print("📈 SUMMARY - KEY IMPROVEMENTS")
print("="*80)

for pair in ['bn-hi', 'bn-te']:
    pair_name = "Bengali-Hindi" if pair == 'bn-hi' else "Bengali-Telugu"
    baseline = all_results[f'baseline_{pair}']
    proposed = all_results[f'proposed_{pair}']

    bleu_imp = ((proposed['bleu'] - baseline['bleu']) / baseline['bleu'] * 100)

    print(f"\n{pair_name}:")
    print(f"  BLEU: {baseline['bleu']:.2f} → {proposed['bleu']:.2f} (+{bleu_imp:.1f}%)")
    print(f"  METEOR: {baseline.get('meteor', 0):.2f} → {proposed.get('meteor', 0):.2f}")
    print(f"  Emotion Acc: {baseline.get('emotion_accuracy', 0):.1f}% → {proposed.get('emotion_accuracy', 0):.1f}%")
    print(f"  Semantic: {baseline.get('semantic_score', 0):.3f} → {proposed.get('semantic_score', 0):.3f}")

print("\n✅ Table 4 generated successfully!")
