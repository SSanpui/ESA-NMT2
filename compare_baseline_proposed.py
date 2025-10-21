#!/usr/bin/env python3
"""
Comprehensive Model Comparison Script
Compares: Baseline NLLB → IndicTrans2 → ESA-NMT (Proposed)

Generates Table 4 from paper with all metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
from emotion_semantic_nmt_enhanced import (
    Config, BHT25Dataset, EmotionSemanticNMT,
    ComprehensiveEvaluator, Trainer
)

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate_model(model_name: str, translation_pair: str,
                              use_emotion: bool, use_semantic: bool, use_style: bool):
    """Train and evaluate a model configuration"""

    print(f"\n{'='*60}")
    print(f"Training: {model_name} ({translation_pair})")
    print(f"{'='*60}")

    # Create model with specific configuration
    model = EmotionSemanticNMT(
        config,
        model_type='nllb',
        use_emotion=use_emotion,
        use_semantic=use_semantic,
        use_style=use_style
    ).to(device)

    # Create datasets
    train_dataset = BHT25Dataset('BHT25_All.csv', model.tokenizer, translation_pair,
                                config.MAX_LENGTH, 'train', 'nllb')
    test_dataset = BHT25Dataset('BHT25_All.csv', model.tokenizer, translation_pair,
                               config.MAX_LENGTH, 'test', 'nllb')

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # Train
    trainer = Trainer(model, config, translation_pair)
    final_loss = 0
    max_memory = 0

    for epoch in range(config.EPOCHS['phase1']):
        loss = trainer.train_epoch(train_loader, epoch)
        final_loss = loss

        if torch.cuda.is_available():
            max_memory = max(max_memory, torch.cuda.max_memory_allocated() / 1024**3)

        print(f"Epoch {epoch+1}/{config.EPOCHS['phase1']} - Loss: {loss:.4f}")

    # Evaluate
    evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
    metrics, preds, refs, sources = evaluator.evaluate(test_loader)

    # Add training info
    metrics['final_training_loss'] = final_loss
    metrics['memory_usage_gb'] = max_memory

    return metrics

def generate_comparison_table(results: dict):
    """Generate comparison table like Table 4"""

    print("\n" + "="*80)
    print("Table 4. Comprehensive Evaluation Results: ESA-NMT vs Baselines")
    print("="*80)

    for pair in ['bn-hi', 'bn-te']:
        pair_name = "Bengali-Hindi" if pair == 'bn-hi' else "Bengali-Telugu"
        print(f"\n{pair_name}:")
        print("-" * 80)

        baseline = results[f'baseline_{pair}']
        proposed = results[f'proposed_{pair}']

        # Translation Quality Metrics
        print("\n📊 Translation Quality Metrics:")
        print(f"{'Metric':<20} {'Baseline':<12} {'Proposed':<12} {'Improvement':<15}")
        print("-" * 80)

        metrics_to_show = [
            ('BLEU', 'bleu'),
            ('METEOR', 'meteor'),
            ('ROUGE-L', 'rouge_l'),
            ('chrF', 'chrf')
        ]

        for name, key in metrics_to_show:
            base_val = baseline.get(key, 0)
            prop_val = proposed.get(key, 0)
            improvement = ((prop_val - base_val) / base_val * 100) if base_val > 0 else 0

            print(f"{name:<20} {base_val:<12.2f} {prop_val:<12.2f} {improvement:>+12.1f}%")

        # Emotion Classification
        print("\n🎭 Emotion Classification:")
        print(f"{'Metric':<20} {'Baseline':<12} {'Proposed':<12}")
        print("-" * 80)

        base_emotion = baseline.get('emotion_accuracy', 0)
        prop_emotion = proposed.get('emotion_accuracy', 0)

        print(f"{'Overall Accuracy':<20} {base_emotion:<12.2f} {prop_emotion:<12.2f}")

        # Semantic Similarity
        print("\n🔗 Semantic Similarity:")
        base_sem = baseline.get('semantic_score', 0)
        prop_sem = proposed.get('semantic_score', 0)
        print(f"{'Semantic Score':<20} {base_sem:<12.4f} {prop_sem:<12.4f}")

        # Training Efficiency
        print("\n⚡ Training Efficiency:")
        print(f"{'Final Training Loss':<20} {baseline.get('final_training_loss', 0):<12.4f} "
              f"{proposed.get('final_training_loss', 0):<12.4f}")
        print(f"{'Memory Usage (GB)':<20} {baseline.get('memory_usage_gb', 0):<12.2f} "
              f"{proposed.get('memory_usage_gb', 0):<12.2f}")

        memory_reduction = ((baseline.get('memory_usage_gb', 0) - proposed.get('memory_usage_gb', 0))
                           / baseline.get('memory_usage_gb', 1) * 100)
        print(f"{'Memory Reduction':<20} {'':<12} {memory_reduction:>+12.1f}%")

def create_latex_table(results: dict):
    """Generate LaTeX table for paper"""

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Comprehensive evaluation results: Proposed ESA-NMT model vs. Baselines}
\label{tab:comprehensive_results}
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{Metric} & \multicolumn{3}{c}{Bengali-Hindi} & \multicolumn{3}{c}{Bengali-Telugu} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
 & Baseline & Proposed & Rel. Imp. & Baseline & Proposed & Rel. Imp. \\
\midrule
"""

    metrics = [
        ('BLEU', 'bleu'),
        ('METEOR', 'meteor'),
        ('ROUGE-L', 'rouge_l'),
        ('chrF', 'chrf'),
    ]

    for name, key in metrics:
        row = f"{name}"

        for pair in ['bn-hi', 'bn-te']:
            baseline = results[f'baseline_{pair}']
            proposed = results[f'proposed_{pair}']

            base_val = baseline.get(key, 0)
            prop_val = proposed.get(key, 0)
            improvement = ((prop_val - base_val) / base_val * 100) if base_val > 0 else 0

            row += f" & {base_val:.2f} & {prop_val:.2f} & +{improvement:.1f}\\%"

        row += " \\\\"
        latex += row + "\n"

    latex += r"""
\midrule
\multicolumn{7}{l}{\textbf{Emotion Classification Results}} \\
"""

    for pair in ['bn-hi', 'bn-te']:
        baseline = results[f'baseline_{pair}']
        proposed = results[f'proposed_{pair}']

        latex += f"Overall Emotion Acc. & {baseline.get('emotion_accuracy', 0):.1f}\\% & "
        latex += f"{proposed.get('emotion_accuracy', 0):.1f}\\% & -- & "

    latex = latex.rstrip(" & ") + " \\\\\n"

    latex += r"""
\midrule
\multicolumn{7}{l}{\textbf{Semantic Similarity}} \\
"""

    for pair in ['bn-hi', 'bn-te']:
        baseline = results[f'baseline_{pair}']
        proposed = results[f'proposed_{pair}']

        latex += f"Semantic Score & {baseline.get('semantic_score', 0):.2f} & "
        latex += f"{proposed.get('semantic_score', 0):.2f} & -- & "

    latex = latex.rstrip(" & ") + " \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    return latex

def main():
    """Run complete comparison"""

    print("🚀 Starting Comprehensive Model Comparison")
    print("This will train and evaluate:")
    print("  1. Baseline (NLLB without custom modules)")
    print("  2. Proposed (ESA-NMT with emotion+semantic+style)")
    print("  For both bn-hi and bn-te pairs")
    print("\n⏰ Estimated time: 6-8 hours total\n")

    results = {}

    # For each language pair
    for pair in ['bn-hi', 'bn-te']:

        # Train baseline (no custom modules)
        print(f"\n{'#'*60}")
        print(f"# {pair.upper()}: BASELINE MODEL")
        print(f"{'#'*60}")

        baseline_metrics = train_and_evaluate_model(
            f"Baseline_{pair}",
            pair,
            use_emotion=False,
            use_semantic=False,
            use_style=False
        )
        results[f'baseline_{pair}'] = baseline_metrics

        # Train proposed (with all custom modules)
        print(f"\n{'#'*60}")
        print(f"# {pair.upper()}: PROPOSED ESA-NMT MODEL")
        print(f"{'#'*60}")

        proposed_metrics = train_and_evaluate_model(
            f"Proposed_{pair}",
            pair,
            use_emotion=True,
            use_semantic=True,
            use_style=True
        )
        results[f'proposed_{pair}'] = proposed_metrics

    # Save results
    with open('./outputs/comparison_results_table4.json', 'w') as f:
        json.dump(ComprehensiveEvaluator.convert_to_json_serializable(results), f, indent=2)

    # Generate comparison table
    generate_comparison_table(results)

    # Generate LaTeX table
    latex_table = create_latex_table(results)
    with open('./outputs/table4_latex.tex', 'w') as f:
        f.write(latex_table)

    print("\n" + "="*80)
    print("✅ Comparison complete!")
    print(f"📁 Results saved to:")
    print(f"   - ./outputs/comparison_results_table4.json")
    print(f"   - ./outputs/table4_latex.tex")
    print("="*80)

    return results

if __name__ == "__main__":
    results = main()
