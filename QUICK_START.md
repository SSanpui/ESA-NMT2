# 🚀 Quick Start Guide - ESA-NMT

**Get results in 3 easy steps!**

---

## ✅ YES, Use Google Colab Pro (Recommended!)

**Why Colab Pro?**
- ✅ Better GPUs (V100 or A100 vs T4)
- ✅ Longer runtime (24 hours vs 12 hours)
- ✅ Faster training (2-3 hours vs 6-8 hours)
- ✅ Worth $10/month for this project!

---

## 🎯 3 Steps to Get Results

### Step 1: Open in Colab (2 minutes)

**Option A: Direct Link (Easiest)**
1. Click this badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSanpui/ESA-NMT/blob/claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj/ESA_NMT_Colab.ipynb)

**Option B: Manual Upload**
1. Go to https://colab.research.google.com/
2. File → Upload notebook
3. Upload `ESA_NMT_Colab.ipynb` from this repository

### Step 2: Configure & Run (1 minute)

1. **Enable GPU**:
   - Runtime → Change runtime type
   - Hardware accelerator: **GPU**
   - GPU type: **V100** (Pro) or **T4** (Free)
   - Click **Save**

2. **Choose Mode** (in the notebook):
   ```python
   RUN_MODE = "quick_demo"  # Start with this! (30-45 mins)
   # Or: "full_training"    # For paper results (3-4 hours)
   # Or: "complete"         # Everything (6-8 hours)

   TRANSLATION_PAIR = "bn-hi"  # or "bn-te"
   ```

3. **Run All**:
   - Runtime → Run all (`Ctrl+F9`)
   - ☕ Grab coffee!

### Step 3: Download Results (1 minute)

After training completes:

```python
# Automatically runs at the end of notebook:
from google.colab import files
files.download('esa_nmt_results.zip')
```

**Done! 🎉**

---

## 📊 What You'll Get

### Results in `esa_nmt_results.zip`:

```
outputs/
├── final_evaluation_nllb_bn-hi.json  ← YOUR METRICS HERE
│   {
│     "bleu": 32.5,
│     "meteor": 45.2,
│     "rouge_l": 48.7,
│     "chrf": 52.3,
│     "emotion_accuracy": 78.4,
│     "semantic_score": 0.867
│   }
│
├── semantic_scores_comparison.png     ← VISUALIZATIONS
├── language_family_analysis.png
└── ablation_study_nllb_bn-hi.png

checkpoints/
└── best_model_nllb_bn-hi.pt          ← TRAINED MODEL
```

---

## ⏱️ How Long?

| Mode | Free (T4) | Pro (V100) | What You Get |
|------|-----------|------------|--------------|
| **quick_demo** | 45 min | 20 min | Basic metrics, verify it works |
| **full_training** | 4 hours | 2 hours | ⭐ **Paper results** - all metrics |
| **complete** | 8 hours | 4 hours | Everything + ablation + tuning |

**Recommendation**:
1. First run: `quick_demo` (verify everything works)
2. Second run: `full_training` (get paper results)

---

## 💡 Pro Tips

### 1. Keep Colab Alive
```javascript
// Run this in browser console to prevent disconnection:
function KeepClicking(){
    console.log("Clicking");
    document.querySelector("colab-connect-button").click();
}
setInterval(KeepClicking, 60000);
```

### 2. Monitor Progress
Watch the output cells - you'll see:
- ✅ Progress bars
- ✅ Loss values (should decrease)
- ✅ BLEU scores (should increase)

### 3. If Disconnected
- Colab saves checkpoints every epoch
- You can resume from where it stopped

### 4. Run Both Language Pairs
```python
# First run:
TRANSLATION_PAIR = "bn-hi"  # Bengali-Hindi
# Download results

# Second run:
TRANSLATION_PAIR = "bn-te"  # Bengali-Telugu
# Download results

# Then compare!
```

---

## 🎓 Expected Results

**Good Results** (should achieve on first try):
- BLEU: 25-30
- Emotion Accuracy: 70-75%
- Semantic Score: 0.80-0.85

**Excellent Results** (with tuning):
- BLEU: 30-35+
- Emotion Accuracy: 75-85%
- Semantic Score: 0.85-0.90

**Cross-Family (bn-te)** may be ~5-10% lower than bn-hi (expected!)

---

## 🆘 Troubleshooting

### "No GPU available"
**Fix**: Runtime → Change runtime type → GPU → Save

### "CUDA Out of Memory"
**Fix**: Restart runtime, it will use smaller batch size

### "Session disconnected"
**Fix**: Check if checkpoint exists, can resume from there

### "Too slow on free tier"
**Fix**: Upgrade to Colab Pro ($10/month, worth it!)

---

## 🎯 Quick Reference

### Full Training Command (if using terminal)
```bash
# Clone
git clone https://github.com/SSanpui/ESA-NMT.git
cd ESA-NMT

# Install
pip install -r requirements.txt

# Run
python run_all_experiments.py --translation_pair bn-hi
```

### Just Want to Try a Demo?
```python
# In Colab, just run:
RUN_MODE = "quick_demo"
# Then Run All
```

### Need All Results for Paper?
```python
# In Colab:
RUN_MODE = "full_training"
# Then Run All
# Wait ~3-4 hours (Pro) or ~6-8 hours (Free)
```

---

## 📈 Timeline for Full Paper Results

**Using Colab Pro (V100):**
- ⏰ 0:00 - Start notebook
- ⏰ 0:05 - Setup complete, training starts
- ⏰ 2:00 - Training complete
- ⏰ 2:30 - Evaluation complete
- ⏰ 2:35 - Download results
- ✅ **Total: ~2.5-3 hours**

**Using Colab Free (T4):**
- Similar timeline but ~2x slower
- ✅ **Total: ~5-6 hours**

---

## ✨ Summary

1. **Open Colab Notebook** (2 min)
2. **Enable GPU** (1 min)
3. **Run All Cells** (press one button!)
4. **Wait** (2-8 hours depending on GPU)
5. **Download Results** (1 min)
6. **Analyze Results** (in your local machine)

**Total Active Time: ~5 minutes**
**Total Waiting Time: 2-8 hours**

---

## 🎁 Bonus: What Else Can You Do?

### Compare Models
```python
RUN_MODE = "complete"  # Compares NLLB vs IndicTrans2
```

### Test Different Hyperparameters
```python
RUN_MODE = "tuning"  # Finds optimal α, β, γ
```

### See Component Importance
```python
RUN_MODE = "ablation"  # Tests each module's contribution
```

---

## 🚀 Ready to Start?

**Click here**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSanpui/ESA-NMT/blob/claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj/ESA_NMT_Colab.ipynb)

Or see detailed instructions: [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md)

---

**Questions?** Check [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md) for detailed troubleshooting!

**Good luck with your experiments! 🎉**
