# Complete Evaluation Guide (Including IndicTrans2)

## Overview

This guide will help you get **ALL** the results you need:

✅ **3-Model Comparison**: NLLB Baseline, ESA-NMT, IndicTrans2
✅ **Ablation Study**: Base, +Emotion, +Semantic, +Both
✅ **All Metrics**: BLEU, METEOR, chrF, ROUGE-L, Emotion Accuracy, Semantic Score

---

## Step-by-Step Process (3 Scripts)

### 📊 Script 1: Main Evaluation (30-40 min)
Run first - evaluates NLLB Baseline, ESA-NMT, and Ablation Study

### 🔤 Script 2: IndicTrans2 (15 min)
Run separately - evaluates IndicTrans2 with your HF token

### 🔗 Script 3: Merge Results (1 min)
Combines everything into final tables

---

## Part 1: Setup (Run Once)

### Cell 1: Fresh Setup

```python
import os, subprocess, shutil

if os.path.exists('ESA-NMT'):
    shutil.rmtree('ESA-NMT')

subprocess.run(['git', 'clone', 'https://github.com/SSanpui/ESA-NMT.git'], check=True)
os.chdir('/kaggle/working/ESA-NMT')
subprocess.run(['git', 'checkout', 'claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj'], check=True)

subprocess.run(['pip', 'install', '-q', 'transformers', 'sentencepiece',
                'sacrebleu', 'rouge-score', 'bert-score', 'sentence-transformers',
                'accelerate', 'nltk', 'huggingface-hub'], check=True)

print("✅ Setup complete!")
```

**Wait for "✅ Setup complete!"**

---

## Part 2: Main Evaluation (NLLB Baseline + ESA-NMT + Ablation)

### Cell 2: Run Main Evaluation (30-40 minutes)

```python
exec(open('complete_evaluation_single_script.py').read())
```

**This evaluates:**
- ✅ NLLB Baseline (~10 min)
- ✅ ESA-NMT Full Model (~10 min)
- ✅ Ablation Study - 4 configs (~20 min)

**Wait for completion**, then click **refresh button** and download:
- `comparison_3models_bn-hi.json`
- `ablation_study_bn-hi.json`

---

## Part 3: IndicTrans2 Evaluation (Separate)

### Cell 3: Evaluate IndicTrans2 (15 minutes)

```python
exec(open('evaluate_indictrans2_separate.py').read())
```

**The script will ask for:**

1. **HuggingFace Token:**
   ```
   Token: hf_xxxxxxxxxxxxxxxxxxxxxxxxx
   ```
   Paste your token and press Enter

2. **Translation Pair:**
   ```
   Enter translation pair (bn-hi or bn-te): bn-hi
   ```

**This will:**
- ✅ Login to HuggingFace
- ✅ Load IndicTrans2 model
- ✅ Translate test set
- ✅ Calculate all metrics
- ✅ Save results

**Wait for completion**, then download:
- `indictrans2_results_bn-hi.json`

---

## Part 4: Merge All Results

### Cell 4: Create Final Tables (1 minute)

```python
exec(open('merge_all_results.py').read())
```

**The script will ask:**
```
Enter translation pair (bn-hi or bn-te): bn-hi
```

**This creates:**
- ✅ Final 3-model comparison table
- ✅ Ablation study table
- ✅ Merged JSON with all results

**Download:**
- `final_complete_results_bn-hi.json`

---

## Expected Output

### Final 3-Model Comparison Table:

```
Model                     BLEU    METEOR  chrF    ROUGE-L   Emotion    Semantic
---------------------------------------------------------------------------------
NLLB Baseline            XX.XX   XX.XX   XX.XX   XX.XX     N/A        N/A
IndicTrans2              XX.XX   XX.XX   XX.XX   XX.XX     N/A        N/A
ESA-NMT                  XX.XX   XX.XX   XX.XX   XX.XX     XX.XX%     X.XXXX
```

### Ablation Study Table:

```
Model                     BLEU    METEOR  chrF    ROUGE-L   Emotion    Semantic
---------------------------------------------------------------------------------
Base NLLB                XX.XX   XX.XX   XX.XX   XX.XX     N/A        N/A
Base + Emotion           XX.XX   XX.XX   XX.XX   XX.XX     XX.XX%     N/A
Base + Semantic          XX.XX   XX.XX   XX.XX   XX.XX     N/A        X.XXXX
Full (Both)              XX.XX   XX.XX   XX.XX   XX.XX     XX.XX%     X.XXXX
```

---

## For bn-te (Telugu)

After completing bn-hi, repeat for bn-te:

1. **Cell 2** - Run main evaluation (change line 252: `TRANSLATION_PAIR = 'bn-te'`)
2. **Cell 3** - Run IndicTrans2 (enter `bn-te` when asked)
3. **Cell 4** - Merge results (enter `bn-te` when asked)

---

## Timeline Summary

| Task | Time |
|------|------|
| Setup | 2 min |
| Main Evaluation (bn-hi) | 30-40 min |
| IndicTrans2 (bn-hi) | 15 min |
| Merge Results (bn-hi) | 1 min |
| **Total for bn-hi** | **~50 min** |
| Repeat for bn-te | ~50 min |
| **Grand Total** | **~1h 40min** |

---

## Important Notes

### About IndicTrans2:

1. **HF Token Required**: Get it from https://huggingface.co/settings/tokens
2. **Request Access**: https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B
3. **Wait for Approval**: Usually instant, max 24 hours
4. **Paste Token**: When Cell 3 asks for it

### About ESA-NMT Checkpoint:

Make sure your trained model is available:
- `/kaggle/working/model_bn-hi.pt` (preferred location)
- OR `./checkpoints/final_model_nllb_bn-hi.pt`

If not found, copy it:
```python
shutil.copy('path/to/your/model.pt', '/kaggle/working/model_bn-hi.pt')
```

---

## Troubleshooting

### "IndicTrans2 failed to load"

1. Check you requested access at HuggingFace
2. Wait for approval email
3. Make sure token is correct (starts with `hf_`)
4. Try again

### "Checkpoint not found"

```python
# Find your checkpoint
!find /kaggle -name "*.pt" -type f 2>/dev/null

# Copy to correct location
shutil.copy('found/path.pt', '/kaggle/working/model_bn-hi.pt')
```

### "CUDA out of memory"

```python
# Clear memory and run one at a time
import torch, gc
torch.cuda.empty_cache()
gc.collect()
```

---

## Files You'll Download (for bn-hi)

After completing all 4 cells:

```
📄 comparison_3models_bn-hi.json      - Main comparison (after Cell 2)
📄 ablation_study_bn-hi.json          - Ablation study (after Cell 2)
📄 indictrans2_results_bn-hi.json     - IndicTrans2 results (after Cell 3)
📄 final_complete_results_bn-hi.json  - Everything merged (after Cell 4)
```

Repeat for bn-te to get same files for Telugu.

---

## Summary

**4 Cells = Complete Results!**

1. ✅ Setup
2. ✅ Main Evaluation (NLLB + ESA-NMT + Ablation)
3. ✅ IndicTrans2 Evaluation (with your HF token)
4. ✅ Merge Everything

**Total time**: ~50 minutes per translation pair

**You get**: All comparisons, all metrics, ready for your paper! 🎉

---

## Quick Reference

```
Cell 1: Setup                                    (2 min)
Cell 2: exec(open('complete_evaluation_single_script.py').read())   (30-40 min)
Cell 3: exec(open('evaluate_indictrans2_separate.py').read())       (15 min)
Cell 4: exec(open('merge_all_results.py').read())                   (1 min)
```

**Just copy and run!** 🚀
