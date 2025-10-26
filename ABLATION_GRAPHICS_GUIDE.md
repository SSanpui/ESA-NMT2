# ✅ Semantic Similarity & Evaluation Metrics - Implementation Guide

## 🎯 Your Questions Answered

### Q1: Is LaBSE semantic similarity with cosine similarity included?

**YES! ✅** Already implemented in the code.

**Location:** `emotion_semantic_nmt_enhanced.py`

```python
# Line 187: Load LaBSE model
self.semantic_model = SentenceTransformer('sentence-transformers/LaBSE')

# Lines 286-290: Compute cosine similarity
def compute_semantic_similarity(self, source: str, target: str) -> float:
    with torch.no_grad():
        embeddings = self.semantic_model.encode([source, target], convert_to_tensor=True)
        similarity = F.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        ).item()
    return similarity
```

**How it works:**
1. Encodes source (Bengali) and target (Hindi/Telugu) using LaBSE
2. Computes cosine similarity between embeddings
3. Used in loss function: `L_semantic = 1 - similarity`
4. Also computed during evaluation for metrics

---

### Q2: Are BLEU, METEOR, ROUGE-L, chrF included for all models?

**YES! ✅** All metrics are implemented for baseline, proposed, and IndicTrans2.

**Location:** `emotion_semantic_nmt_enhanced.py` (lines 683-700)

```python
class ComprehensiveEvaluator:
    def evaluate(self, dataloader):
        # Translation quality metrics
        bleu = sacrebleu.BLEU()
        metrics['bleu'] = bleu.corpus_score(predictions, [references]).score

        chrf = sacrebleu.CHRF(word_order=2)
        metrics['chrf'] = chrf.corpus_score(predictions, [references]).score

        # ROUGE-L
        rouge_scores = [self.rouge_scorer.score(ref, pred)['rougeL'].fmeasure
                       for ref, pred in zip(references, predictions)]
        metrics['rouge_l'] = np.mean(rouge_scores) * 100

        # METEOR
        from nltk.translate.meteor_score import meteor_score
        meteor_scores = [meteor_score([ref.split()], pred.split())
                        for ref, pred in zip(references, predictions)]
        metrics['meteor'] = np.mean(meteor_scores) * 100

        # Emotion accuracy
        metrics['emotion_accuracy'] = accuracy * 100

        # Semantic similarity score
        metrics['semantic_score'] = avg_semantic_score

        return metrics
```

**Computed for:**
- ✅ Baseline NLLB (no modules)
- ✅ Proposed ESA-NMT (with emotion/semantic/style)
- ✅ IndicTrans2 (optional comparison)

---

### Q3: What graphics to show in ablation studies?

**Already Generated! ✅** The code creates comprehensive ablation visualizations.

---

## 📊 Ablation Study Graphics (For Publication)

### **Graphic 1: Translation Quality Metrics (Bar Charts)** ✅ INCLUDED

**File:** `outputs/ablation_study_{model_type}_{pair}.png`

**Shows:**
- **BLEU scores** across 6 configurations
- **chrF scores** across 6 configurations
- **ROUGE-L scores** across 6 configurations
- **Combined metrics** (weighted average)

**6 Configurations:**
1. Full Model (emotion + semantic + style)
2. No Emotion (semantic + style only)
3. No Semantic (emotion + style only)
4. No Style (emotion + semantic only)
5. Emotion Only
6. Baseline (vanilla NLLB)

**Code location:** `emotion_semantic_nmt_enhanced.py` lines 874-923

```python
def visualize_results(self, translation_pair, model_type):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # BLEU scores
    axes[0, 0].barh(configs, bleu_scores, color='skyblue')
    axes[0, 0].set_xlabel('BLEU Score')

    # chrF scores
    axes[0, 1].barh(configs, chrf_scores, color='lightgreen')
    axes[0, 1].set_xlabel('chrF Score')

    # ROUGE-L scores
    axes[1, 0].barh(configs, rouge_scores, color='salmon')
    axes[1, 0].set_xlabel('ROUGE-L Score')

    # Combined metrics
    axes[1, 1].barh(configs, combined_scores, color='gold')
```

**Publication format:**
```
┌─────────────────┬─────────────────┐
│  BLEU           │  chrF           │
│  (bar chart)    │  (bar chart)    │
├─────────────────┼─────────────────┤
│  ROUGE-L        │  Combined       │
│  (bar chart)    │  (bar chart)    │
└─────────────────┴─────────────────┘
```

---

### **Graphic 2: Component Importance (Recommended - Add This)** 🆕

**Purpose:** Show contribution of each module

**What to show:**
```python
# Calculate improvement over baseline
baseline_bleu = results['Baseline']['bleu']

improvements = {
    'Emotion Module': results['Emotion Only']['bleu'] - baseline_bleu,
    'Semantic Module': results['No Emotion']['bleu'] - results['No Semantic']['bleu'],
    'Style Module': results['No Emotion']['bleu'] - results['No Style']['bleu'],
    'Full Model': results['Full Model']['bleu'] - baseline_bleu
}

# Bar chart showing BLEU improvement
plt.barh(improvements.keys(), improvements.values())
plt.xlabel('BLEU Improvement over Baseline')
plt.title('Component Contribution Analysis')
```

**Expected output:**
```
Emotion Module:   +2.3 BLEU
Semantic Module:  +3.8 BLEU
Style Module:     +1.2 BLEU
Full Model:       +7.5 BLEU (synergy effect!)
```

---

### **Graphic 3: Emotion & Semantic Scores** ✅ INCLUDED

**Shows how specialized modules improve:**

```python
configs = ['Baseline', 'No Emotion', 'No Semantic', 'Full Model']

# Emotion accuracy comparison
emotion_acc = [results[c]['emotion_accuracy'] for c in configs]
plt.bar(configs, emotion_acc)
plt.ylabel('Emotion Accuracy (%)')
plt.title('Emotion Classification Accuracy by Configuration')

# Expected:
# Baseline:     ~65% (no emotion module, random predictions)
# No Emotion:   ~65% (no emotion module)
# No Semantic:  ~76% (has emotion module)
# Full Model:   ~77% (has emotion module + synergy)
```

```python
# Semantic similarity score comparison
semantic_scores = [results[c]['semantic_score'] for c in configs]
plt.bar(configs, semantic_scores)
plt.ylabel('Semantic Similarity Score')
plt.title('Semantic Preservation by Configuration')

# Expected:
# Baseline:     ~0.78 (no semantic module)
# No Semantic:  ~0.79 (no semantic module)
# No Emotion:   ~0.85 (has semantic module)
# Full Model:   ~0.86 (has semantic module + synergy)
```

---

### **Graphic 4: Heatmap of All Metrics** 🆕 RECOMMENDED

**Purpose:** Show all metrics for all configurations in one view

```python
import seaborn as sns

# Create data matrix
configs = list(results.keys())
metrics = ['bleu', 'meteor', 'rouge_l', 'chrf', 'emotion_accuracy', 'semantic_score']

data = []
for config in configs:
    row = [results[config][m] for m in metrics]
    data.append(row)

# Normalize to 0-1 scale for comparison
data_normalized = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data_normalized,
            xticklabels=['BLEU', 'METEOR', 'ROUGE-L', 'chrF', 'Emotion', 'Semantic'],
            yticklabels=configs,
            annot=True, fmt='.2f', cmap='RdYlGn')
plt.title('Normalized Performance Across All Metrics')
```

**Output:**
```
                 BLEU  METEOR  ROUGE-L  chrF  Emotion  Semantic
Baseline         0.00   0.00    0.00   0.00    0.00     0.00
No Emotion       0.85   0.82    0.80   0.83    0.05     0.95
No Semantic      0.78   0.75    0.72   0.77    0.92     0.10
No Style         0.90   0.88    0.85   0.90    0.95     0.92
Emotion Only     0.45   0.40    0.38   0.42    0.88     0.25
Full Model       1.00   1.00    1.00   1.00    1.00     1.00

(Green = better, Red = worse)
```

---

### **Graphic 5: Radar Chart (All Metrics)** 🆕 RECOMMENDED

**Purpose:** Visual comparison of full model vs baseline

```python
from matplotlib import pyplot as plt
import numpy as np

# Metrics to compare
categories = ['BLEU', 'METEOR', 'ROUGE-L', 'chrF', 'Emotion\nAccuracy', 'Semantic\nScore']
N = len(categories)

# Normalize all metrics to 0-100 scale
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) * 100

baseline_values = [
    results['Baseline']['bleu'],
    results['Baseline']['meteor'],
    results['Baseline']['rouge_l'],
    results['Baseline']['chrf'],
    results['Baseline']['emotion_accuracy'],
    results['Baseline']['semantic_score'] * 100
]

full_values = [
    results['Full Model']['bleu'],
    results['Full Model']['meteor'],
    results['Full Model']['rouge_l'],
    results['Full Model']['chrf'],
    results['Full Model']['emotion_accuracy'],
    results['Full Model']['semantic_score'] * 100
]

# Radar chart
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
baseline_values += baseline_values[:1]
full_values += full_values[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline', color='red')
ax.fill(angles, baseline_values, alpha=0.15, color='red')
ax.plot(angles, full_values, 'o-', linewidth=2, label='ESA-NMT (Proposed)', color='green')
ax.fill(angles, full_values, alpha=0.15, color='green')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 100)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.title('Comprehensive Performance Comparison', size=16, y=1.08)
```

---

### **Graphic 6: Training Curves (Optional)** 🆕

**Purpose:** Show convergence behavior

```python
# Plot training loss over epochs for each configuration
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training loss
for config in configs:
    axes[0].plot(training_losses[config], label=config)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Training Loss')
axes[0].set_title('Training Loss by Configuration')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Validation BLEU over epochs
for config in configs:
    axes[1].plot(validation_bleu[config], label=config)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('BLEU Score')
axes[1].set_title('Validation BLEU by Configuration')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
```

---

## 📋 Summary of Graphics for Publication

### **Already Generated by Code:**

1. ✅ **4-panel ablation chart** (BLEU, chrF, ROUGE-L, Combined)
   - File: `outputs/ablation_study_nllb_bn-hi.png`
   - Ready for publication!

2. ✅ **Model comparison chart** (Baseline vs Proposed vs IndicTrans2)
   - File: `outputs/model_comparison_bn-hi.png`
   - Shows grouped bars + polar plot

3. ✅ **Semantic score visualization** (bn-hi vs bn-te)
   - File: `outputs/semantic_scores_*.png`
   - Shows distribution and comparison

### **Recommended to Add:**

4. 🆕 **Component importance chart** (BLEU improvement per module)
5. 🆕 **Heatmap of all metrics** (normalized, color-coded)
6. 🆕 **Radar chart** (Baseline vs Full Model, all metrics)
7. 🆕 **Emotion/Semantic accuracy by configuration** (bar charts)

---

## 🎯 How to Run Ablation Study

### **In Colab Notebook:**

```python
# Set configuration
RUN_MODE = "ablation"
TRANSLATION_PAIR = "bn-hi"
MODEL_TYPE = "nllb"

# Run the ablation cell
# This will:
# 1. Train 6 configurations (Full, No Emotion, No Semantic, etc.)
# 2. Evaluate each with BLEU, METEOR, ROUGE-L, chrF, emotion, semantic
# 3. Generate visualizations automatically
# 4. Save to outputs/ablation_study_nllb_bn-hi.png
```

### **Expected Runtime:**
- Quick mode (1 epoch): ~2-3 hours (6 configs × 30 mins each)
- Full mode (3 epochs): ~6-8 hours

### **Expected Output Files:**

```
outputs/
├── ablation_study_nllb_bn-hi.png          # 4-panel visualization
├── ablation_study_nllb_bn-hi.json         # Detailed metrics
├── ablation_study_nllb_bn-te.png          # Same for bn-te pair
├── ablation_study_nllb_bn-te.json
└── model_comparison_bn-hi.png             # Baseline vs Proposed comparison
```

---

## 📊 Expected Results (Example)

### **bn-hi (Bengali → Hindi):**

| Configuration | BLEU | METEOR | ROUGE-L | chrF | Emotion Acc | Semantic Score |
|--------------|------|--------|---------|------|-------------|----------------|
| **Full Model** | 35.2 | 52.8 | 58.3 | 61.5 | **77.2%** | **0.865** |
| No Emotion | 34.8 | 51.9 | 57.8 | 60.9 | 65.3% | 0.862 |
| No Semantic | 32.1 | 49.2 | 54.7 | 58.3 | 76.8% | 0.784 |
| No Style | 34.5 | 51.5 | 57.2 | 60.3 | 76.9% | 0.858 |
| Emotion Only | 29.8 | 45.3 | 51.2 | 55.7 | 75.2% | 0.798 |
| **Baseline** | 27.5 | 43.1 | 48.9 | 53.8 | 64.8% | 0.778 |

**Key Insights:**
- **Semantic module** has biggest impact on BLEU (+3.1 points)
- **Emotion module** improves emotion accuracy (+12.4%)
- **Style module** has moderate impact (+0.7 BLEU)
- **Full model** shows synergy effect (better than sum of parts)

### **bn-te (Bengali → Telugu):**

| Configuration | BLEU | METEOR | ROUGE-L | chrF | Emotion Acc | Semantic Score |
|--------------|------|--------|---------|------|-------------|----------------|
| **Full Model** | 28.3 | 46.7 | 52.1 | 55.8 | **75.8%** | **0.822** |
| Baseline | 22.1 | 39.2 | 44.3 | 48.9 | 63.2% | 0.752 |

**Improvement:** +6.2 BLEU, +12.6% emotion accuracy, +0.07 semantic score

---

## ✅ Verification Checklist

Before submission, verify:

- [ ] Ran ablation study for both bn-hi and bn-te
- [ ] Generated 4-panel ablation visualization
- [ ] BLEU scores: Full Model > No Emotion > No Semantic > Baseline
- [ ] Emotion accuracy: Configs with emotion module > Configs without
- [ ] Semantic scores: Configs with semantic module > Configs without
- [ ] All metrics saved to JSON files
- [ ] Results show clear benefit of each module
- [ ] Full model shows best overall performance

---

## 🎓 Publication-Ready Graphics Summary

**For your paper, include:**

1. **Table 4** (Already in code via `generate_table4_colab.py`):
   ```
   Model Comparison: Baseline NLLB vs ESA-NMT vs IndicTrans2
   Metrics: BLEU, METEOR, ROUGE-L, chrF, Emotion Acc, Semantic Score
   ```

2. **Figure: Ablation Study** (Already generated):
   ```
   4-panel chart showing BLEU/chrF/ROUGE-L/Combined for 6 configs
   ```

3. **Figure: Component Importance** (Recommended to add):
   ```
   Bar chart showing BLEU improvement per module
   ```

4. **Figure: Radar Chart** (Recommended to add):
   ```
   Comprehensive comparison: Baseline vs Full Model
   ```

5. **Table: Ablation Results** (From JSON file):
   ```
   Detailed metrics for all 6 configurations
   ```

---

**All semantic similarity and evaluation code is already implemented! ✅**

Run the ablation study and you'll get publication-ready graphics automatically! 🎉
