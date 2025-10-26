#!/usr/bin/env python3
"""
Generate Additional Ablation Study Graphics
Creates publication-ready visualizations:
1. Component importance chart
2. Heatmap of all metrics
3. Radar chart comparison
4. Emotion/Semantic accuracy by configuration
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load ablation results
def load_results(json_file):
    """Load ablation study results from JSON"""
    with open(json_file, 'r') as f:
        return json.load(f)

# ============================================================================
# GRAPHIC 1: Component Importance (BLEU Improvement)
# ============================================================================

def plot_component_importance(results, output_dir, translation_pair, model_type):
    """Show contribution of each module to BLEU improvement"""

    baseline_bleu = results['Baseline']['bleu']

    # Calculate improvements
    improvements = {
        'Emotion\nModule': results['Emotion Only']['bleu'] - baseline_bleu,
        'Semantic\nModule': results['Full Model']['bleu'] - results['No Semantic']['bleu'],
        'Style\nModule': results['Full Model']['bleu'] - results['No Style']['bleu'],
        'Full\nModel': results['Full Model']['bleu'] - baseline_bleu
    }

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax.bar(improvements.keys(), improvements.values(), color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'+{height:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('BLEU Score Improvement', fontsize=14, fontweight='bold')
    ax.set_xlabel('Component', fontsize=14, fontweight='bold')
    ax.set_title(f'Component Contribution Analysis ({translation_pair.upper()})',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(improvements.values()) * 1.2)

    # Add baseline reference line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/component_importance_{model_type}_{translation_pair}.png",
                dpi=300, bbox_inches='tight')
    print(f"✅ Component importance chart saved!")
    plt.close()

# ============================================================================
# GRAPHIC 2: Heatmap of All Metrics
# ============================================================================

def plot_metrics_heatmap(results, output_dir, translation_pair, model_type):
    """Heatmap showing all metrics for all configurations"""

    # Define configurations and metrics
    configs = ['Baseline', 'Emotion Only', 'No Style', 'No Semantic', 'No Emotion', 'Full Model']
    metric_names = ['BLEU', 'METEOR', 'ROUGE-L', 'chrF', 'Emotion\nAccuracy', 'Semantic\nScore']
    metrics = ['bleu', 'meteor', 'rouge_l', 'chrf', 'emotion_accuracy', 'semantic_score']

    # Extract data
    data = []
    for config in configs:
        if config in results:
            row = []
            for m in metrics:
                value = results[config].get(m, 0)
                # Normalize semantic_score to 0-100 scale
                if m == 'semantic_score':
                    value *= 100
                row.append(value)
            data.append(row)

    data = np.array(data)

    # Normalize to 0-1 for color mapping (but keep original values for annotation)
    data_normalized = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        min_val, max_val = col.min(), col.max()
        if max_val > min_val:
            data_normalized[:, j] = (col - min_val) / (max_val - min_val)
        else:
            data_normalized[:, j] = 0.5

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use normalized data for colors, original data for annotations
    im = ax.imshow(data_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(configs)))
    ax.set_xticklabels(metric_names, fontsize=12, fontweight='bold')
    ax.set_yticklabels(configs, fontsize=12, fontweight='bold')

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add annotations with original values
    for i in range(len(configs)):
        for j in range(len(metrics)):
            value = data[i, j]
            # Choose text color based on background
            text_color = 'white' if data_normalized[i, j] < 0.5 else 'black'
            text = ax.text(j, i, f'{value:.1f}',
                          ha="center", va="center", color=text_color,
                          fontsize=11, fontweight='bold')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Normalized Performance (0=worst, 1=best)',
                       rotation=-90, va="bottom", fontsize=11, fontweight='bold')

    ax.set_title(f'Comprehensive Metrics Heatmap ({translation_pair.upper()})',
                fontsize=16, fontweight='bold', pad=20)

    # Add grid
    ax.set_xticks(np.arange(len(metric_names))-.5, minor=True)
    ax.set_yticks(np.arange(len(configs))-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=2)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_heatmap_{model_type}_{translation_pair}.png",
                dpi=300, bbox_inches='tight')
    print(f"✅ Metrics heatmap saved!")
    plt.close()

# ============================================================================
# GRAPHIC 3: Radar Chart (Baseline vs Full Model)
# ============================================================================

def plot_radar_chart(results, output_dir, translation_pair, model_type):
    """Radar chart comparing Baseline vs Full Model across all metrics"""

    # Metrics to compare
    categories = ['BLEU', 'METEOR', 'ROUGE-L', 'chrF', 'Emotion\nAccuracy', 'Semantic\nScore']
    metrics = ['bleu', 'meteor', 'rouge_l', 'chrf', 'emotion_accuracy', 'semantic_score']

    N = len(categories)

    # Get values
    baseline_values = []
    full_values = []

    for m in metrics:
        base_val = results['Baseline'].get(m, 0)
        full_val = results['Full Model'].get(m, 0)

        # Normalize semantic_score to 0-100 scale
        if m == 'semantic_score':
            base_val *= 100
            full_val *= 100

        baseline_values.append(base_val)
        full_values.append(full_val)

    # Setup radar chart
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Close the plot
    baseline_values += baseline_values[:1]
    full_values += full_values[:1]
    angles += angles[:1]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Plot baseline
    ax.plot(angles, baseline_values, 'o-', linewidth=3, label='Baseline NLLB',
            color='#FF6B6B', markersize=8)
    ax.fill(angles, baseline_values, alpha=0.15, color='#FF6B6B')

    # Plot full model
    ax.plot(angles, full_values, 'o-', linewidth=3, label='ESA-NMT (Proposed)',
            color='#4ECDC4', markersize=8)
    ax.fill(angles, full_values, alpha=0.15, color='#4ECDC4')

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13, fontweight='bold')

    # Set y-axis limits
    max_val = max(max(baseline_values), max(full_values))
    ax.set_ylim(0, max_val * 1.1)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=13, frameon=True, shadow=True)

    # Title
    plt.title(f'Comprehensive Performance Comparison\n{translation_pair.upper()}',
              size=18, fontweight='bold', y=1.12)

    # Grid
    ax.grid(True, linewidth=1.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/radar_chart_{model_type}_{translation_pair}.png",
                dpi=300, bbox_inches='tight')
    print(f"✅ Radar chart saved!")
    plt.close()

# ============================================================================
# GRAPHIC 4: Emotion & Semantic Accuracy by Configuration
# ============================================================================

def plot_specialized_metrics(results, output_dir, translation_pair, model_type):
    """Bar charts for emotion accuracy and semantic scores"""

    configs = ['Baseline', 'No Emotion', 'No Semantic', 'Emotion Only', 'Full Model']

    # Extract values
    emotion_acc = [results[c].get('emotion_accuracy', 0) for c in configs]
    semantic_scores = [results[c].get('semantic_score', 0) * 100 for c in configs]  # Scale to 0-100

    # Create side-by-side bar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Emotion Accuracy
    colors1 = ['#FF6B6B' if 'No Emotion' in c or c == 'Baseline' else '#4ECDC4' for c in configs]
    bars1 = ax1.bar(range(len(configs)), emotion_acc, color=colors1, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=15, ha='right', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Emotion Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Emotion Classification Accuracy', fontsize=15, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(emotion_acc) * 1.15)

    # Semantic Similarity Score
    colors2 = ['#FF6B6B' if 'No Semantic' in c or c == 'Baseline' else '#45B7D1' for c in configs]
    bars2 = ax2.bar(range(len(configs)), semantic_scores, color=colors2, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=15, ha='right', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Semantic Similarity Score (scaled to 100)', fontsize=13, fontweight='bold')
    ax2.set_title('Semantic Preservation', fontsize=15, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max(semantic_scores) * 1.15)

    plt.suptitle(f'Specialized Module Performance ({translation_pair.upper()})',
                 fontsize=17, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/specialized_metrics_{model_type}_{translation_pair}.png",
                dpi=300, bbox_inches='tight')
    print(f"✅ Specialized metrics chart saved!")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate all additional graphics"""

    print("🎨 Generating Additional Ablation Study Graphics")
    print("="*60)

    # Configuration
    output_dir = "./outputs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process both translation pairs
    for translation_pair in ['bn-hi', 'bn-te']:
        for model_type in ['nllb']:  # Can add 'indictrans2' if needed

            json_file = f"{output_dir}/ablation_study_{model_type}_{translation_pair}.json"

            if not Path(json_file).exists():
                print(f"⚠️  {json_file} not found. Run ablation study first!")
                continue

            print(f"\n📊 Processing {translation_pair.upper()} ({model_type})...")

            # Load results
            results = load_results(json_file)

            # Generate all graphics
            print("\n1️⃣ Generating component importance chart...")
            plot_component_importance(results, output_dir, translation_pair, model_type)

            print("2️⃣ Generating metrics heatmap...")
            plot_metrics_heatmap(results, output_dir, translation_pair, model_type)

            print("3️⃣ Generating radar chart...")
            plot_radar_chart(results, output_dir, translation_pair, model_type)

            print("4️⃣ Generating specialized metrics charts...")
            plot_specialized_metrics(results, output_dir, translation_pair, model_type)

    print("\n" + "="*60)
    print("✅ All graphics generated successfully!")
    print("\n📁 Output files:")
    print(f"   - {output_dir}/component_importance_*.png")
    print(f"   - {output_dir}/metrics_heatmap_*.png")
    print(f"   - {output_dir}/radar_chart_*.png")
    print(f"   - {output_dir}/specialized_metrics_*.png")
    print("\n🎯 These graphics are publication-ready!")

if __name__ == "__main__":
    main()
