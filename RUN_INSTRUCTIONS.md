# How to Run ESA-NMT Experiments

This guide explains how to run the Emotion-Semantic-Aware NMT experiments and get results.

---

## 🎯 Option 1: Google Colab (RECOMMENDED)

### Why Colab?
- ✅ **Free GPU access** (T4 GPU in free tier)
- ✅ **Colab Pro** gives better GPUs (V100, A100)
- ✅ **No setup required** - everything runs in browser
- ✅ **Easy to download results**

### System Requirements

| Tier | GPU | RAM | Runtime | Cost |
|------|-----|-----|---------|------|
| **Colab Free** | T4 (16GB) | 12GB | ~6-8 hours | Free |
| **Colab Pro** | V100 (16GB) | 32GB | ~3-4 hours | $10/month |
| **Colab Pro+** | A100 (40GB) | 52GB | ~2-3 hours | $50/month |

### Quick Start with Colab

**Method 1: Direct Upload (Easiest)**

1. **Open Google Colab**: https://colab.research.google.com/

2. **Upload the notebook**:
   - File → Upload notebook
   - Upload `ESA_NMT_Colab_Notebook.py` (or create new notebook and copy code)

3. **Change Runtime**:
   - Runtime → Change runtime type
   - Hardware accelerator: **GPU**
   - GPU type: **T4** (free) or **V100/A100** (Pro)
   - Click **Save**

4. **Run all cells**:
   - Runtime → Run all
   - Or press `Ctrl+F9`

5. **Wait for completion** (3-8 hours depending on GPU)

6. **Download results**:
   ```python
   # Run in a cell at the end:
   from google.colab import files
   files.download('esa_nmt_results.zip')
   ```

**Method 2: Clone from GitHub**

```python
# Cell 1: Clone repository
!git clone https://github.com/SSanpui/ESA-NMT.git
%cd ESA-NMT
!git checkout claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj

# Cell 2: Install dependencies
!pip install -q transformers sentence-transformers sacrebleu rouge-score accelerate
!python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Cell 3: Run experiments
!python run_all_experiments.py --translation_pair bn-hi
```

### Running Different Modes

**Quick Demo (30 mins - 1 hour)**
```python
# Edit in the notebook:
RUN_MODE = "quick_demo"
TRANSLATION_PAIR = "bn-hi"  # or "bn-te"
MODEL_TYPE = "nllb"
```
- Trains for 1 epoch
- Quick evaluation
- Good for testing

**Full Training (3-4 hours)**
```python
RUN_MODE = "full_training"
```
- Complete 3-epoch training
- Full evaluation with all metrics
- Best for paper results

**Ablation Study (4-6 hours)**
```python
RUN_MODE = "ablation"
```
- Tests 6 configurations
- Shows component importance
- Generates comparison charts

**Hyperparameter Tuning (3-5 hours)**
```python
RUN_MODE = "tuning"
```
- Grid search for α, β, γ
- Finds optimal weights
- Saves best parameters

---

## 🖥️ Option 2: Local Machine with GPU

### Requirements
- NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4090, A100, etc.)
- CUDA 11.8+
- Python 3.8+
- 20GB+ free disk space

### Setup

```bash
# 1. Clone repository
git clone https://github.com/SSanpui/ESA-NMT.git
cd ESA-NMT
git checkout claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# 5. Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Running Experiments

**Option A: Run All Experiments (Complete Pipeline)**
```bash
python run_all_experiments.py --translation_pair bn-hi
```

**Option B: Interactive Menu**
```bash
python emotion_semantic_nmt_enhanced.py

# Then choose:
# 1 - Model comparison
# 2 - Ablation study
# 3 - Hyperparameter tuning
# 4 - Train specific model
# 5 - Evaluate model
# 6 - Prepare for deployment
```

**Option C: Individual Commands**

```bash
# Just training
python emotion_semantic_nmt_enhanced.py
# Select: 4
# Enter: bn-hi
# Enter: nllb

# Just evaluation
python emotion_semantic_nmt_enhanced.py
# Select: 5
# Enter: bn-hi
# Enter: nllb

# Just visualization
python visualize_semantic_scores.py
```

---

## 🐧 Option 3: Cloud Platforms

### Amazon SageMaker
```bash
# Use g4dn.xlarge or higher
# Estimated cost: ~$0.50-2.00/hour
```

### Azure Machine Learning
```bash
# Use Standard_NC6s_v3 or higher
# Estimated cost: ~$0.90-3.00/hour
```

### Google Cloud Platform
```bash
# Use n1-standard-8 with T4 GPU
# Estimated cost: ~$0.35-1.50/hour
```

---

## 📊 What Results Will You Get?

### Generated Files

After running experiments, you'll get:

```
outputs/
├── model_comparison_bn-hi.json          # NLLB vs IndicTrans2 comparison
├── ablation_study_nllb_bn-hi.json      # Component importance
├── hyperparameter_tuning_bn-hi.json     # Optimal α, β, γ
├── final_evaluation_nllb_bn-hi.json    # Complete metrics
├── semantic_scores_comparison.png       # Visualization
├── language_family_analysis.png         # Family comparison
├── ablation_study_nllb_bn-hi.png       # Ablation charts
└── experiment_report_bn-hi_*.md         # Full report

checkpoints/
├── best_model_nllb_bn-hi.pt            # Best model checkpoint
└── best_model_nllb_bn-te.pt            # For Telugu pair

models/
└── emotion-semantic-nmt-nllb-bn-hi/    # Deployment-ready
    ├── pytorch_model.bin
    ├── config.json
    ├── tokenizer_config.json
    └── README.md
```

### Metrics You'll Get

**Translation Quality:**
- BLEU score (higher is better, typically 25-35)
- METEOR score (higher is better, typically 40-50)
- ROUGE-L score (higher is better, typically 45-55)
- chrF score (higher is better, typically 50-60)

**Specialized Metrics:**
- Emotion Classification Accuracy (%) - typically 70-85%
- Semantic Similarity Score (0-1) - typically 0.80-0.90

**For Each Language Pair:**
- bn-hi (Bengali → Hindi): Same language family
- bn-te (Bengali → Telugu): Cross language family

### Example Output

```json
{
  "bleu": 32.5,
  "meteor": 45.2,
  "rouge_l": 48.7,
  "chrf": 52.3,
  "emotion_accuracy": 78.4,
  "semantic_score": 0.867,
  "num_samples": 500
}
```

---

## ⏱️ Estimated Runtimes

### Google Colab Free (T4 GPU)
- Quick Demo: 30-45 minutes
- Full Training (bn-hi): 3-4 hours
- Full Training (bn-te): 3-4 hours
- Ablation Study: 5-7 hours
- Hyperparameter Tuning: 4-6 hours
- **Complete Pipeline**: 6-8 hours

### Google Colab Pro (V100 GPU)
- Quick Demo: 15-20 minutes
- Full Training (bn-hi): 1.5-2 hours
- Full Training (bn-te): 1.5-2 hours
- Ablation Study: 3-4 hours
- Hyperparameter Tuning: 2-3 hours
- **Complete Pipeline**: 3-4 hours

### Local RTX 3090 / RTX 4090
- Similar to Colab Pro
- **Complete Pipeline**: 3-5 hours

### Local A100 (40GB)
- Quick Demo: 10 minutes
- Full Training: 45-60 minutes
- **Complete Pipeline**: 2-3 hours

---

## 🚀 Recommended Workflow

### For Quick Testing (30 minutes)
```python
# In Colab or local
RUN_MODE = "quick_demo"
TRANSLATION_PAIR = "bn-hi"
```
**Result**: Basic metrics, sample translations

### For Paper Results (4-6 hours)
```bash
# Option 1: Complete pipeline
python run_all_experiments.py --translation_pair bn-hi

# Option 2: Step by step
# 1. Train model
python emotion_semantic_nmt_enhanced.py  # Select 4
# 2. Evaluate
python emotion_semantic_nmt_enhanced.py  # Select 5
# 3. Visualize
python visualize_semantic_scores.py
```
**Result**: All metrics, ablation study, hyperparameter tuning

### For Both Language Pairs (8-12 hours)
```bash
# Run for bn-hi
python run_all_experiments.py --translation_pair bn-hi

# Run for bn-te
python run_all_experiments.py --translation_pair bn-te

# Compare
python visualize_semantic_scores.py
```
**Result**: Complete comparison, cross-family analysis

---

## 💡 Tips for Best Results

### 1. Use Colab Pro if Possible
- **Free tier**: Limited to 12 hours, may disconnect
- **Pro tier**: Longer sessions, better GPUs
- **Worth it** for this project!

### 2. Save Checkpoints Regularly
The code auto-saves checkpoints every epoch to `./checkpoints/`

### 3. Monitor Training
Watch the progress bars and loss values:
- Loss should decrease over time
- BLEU should increase

### 4. Handle Disconnections (Colab)
If Colab disconnects:
```python
# Check if checkpoint exists
import os
if os.path.exists('./checkpoints/best_model_nllb_bn-hi.pt'):
    print("Checkpoint found! Can resume evaluation")
```

### 5. Reduce Batch Size if OOM
If you get "CUDA Out of Memory":
```python
# Edit in emotion_semantic_nmt_enhanced.py
Config.BATCH_SIZE = 1  # Reduce from 2
Config.GRADIENT_ACCUMULATION_STEPS = 8  # Keep effective batch size
```

---

## 📥 Downloading Results

### From Colab
```python
# Option 1: Download specific file
from google.colab import files
files.download('./outputs/final_evaluation_nllb_bn-hi.json')

# Option 2: Download all results (recommended)
!zip -r esa_nmt_results.zip ./outputs ./checkpoints
files.download('esa_nmt_results.zip')
```

### From Local Machine
Results are saved in:
- `./outputs/` - JSON results and visualizations
- `./checkpoints/` - Model checkpoints
- `./models/` - Deployment-ready models

---

## 🔍 Verifying Results

### Check Training Completed
```bash
# Should see these files:
ls ./checkpoints/best_model_nllb_bn-hi.pt
ls ./outputs/final_evaluation_nllb_bn-hi.json
```

### Validate Metrics
```python
import json

with open('./outputs/final_evaluation_nllb_bn-hi.json', 'r') as f:
    results = json.load(f)

print(f"BLEU: {results['metrics']['bleu']:.2f}")
print(f"Emotion Accuracy: {results['metrics']['emotion_accuracy']:.2f}%")
print(f"Semantic Score: {results['metrics']['semantic_score']:.4f}")

# Expected ranges:
# BLEU: 25-35 (good), 35+ (excellent)
# Emotion Acc: 70-85%
# Semantic: 0.80-0.90
```

### View Visualizations
```python
from IPython.display import Image, display
display(Image('./outputs/semantic_scores_comparison.png'))
display(Image('./outputs/language_family_analysis.png'))
```

---

## ❓ Troubleshooting

### Issue: CUDA Out of Memory
**Solution:**
```python
# Reduce batch size
Config.BATCH_SIZE = 1
Config.MAX_LENGTH = 96
```

### Issue: Colab Disconnects
**Solution:**
- Use Colab Pro for longer sessions
- Save checkpoints frequently (already implemented)
- Can resume from checkpoint

### Issue: Slow Training
**Solution:**
```python
# Reduce epochs for quick test
Config.EPOCHS['phase1'] = 1
```

### Issue: ModuleNotFoundError
**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Dataset Not Found
**Solution:**
```bash
# Make sure you're in the right directory
cd ESA-NMT

# Check if file exists
ls -la BHT25_All.csv
```

---

## 📞 Getting Help

If you encounter issues:

1. **Check logs** in Colab output or terminal
2. **Review** `IMPLEMENTATION_SUMMARY.md` for details
3. **Check** GitHub Issues: https://github.com/SSanpui/ESA-NMT/issues
4. **Verify** all dependencies are installed
5. **Ensure** you have enough GPU memory

---

## 🎓 Summary: Easiest Path to Results

**Step 1**: Open Google Colab (https://colab.research.google.com/)

**Step 2**: Create new notebook or upload `ESA_NMT_Colab_Notebook.py`

**Step 3**: Change runtime to GPU (Runtime → Change runtime type → GPU)

**Step 4**: Run cells (Runtime → Run all)

**Step 5**: Wait 3-8 hours (depending on GPU)

**Step 6**: Download results:
```python
from google.colab import files
files.download('esa_nmt_results.zip')
```

**Step 7**: Extract and review:
- `outputs/*.json` - Metrics
- `outputs/*.png` - Visualizations
- `checkpoints/*.pt` - Model weights

---

## 📈 Expected Timeline

| Task | Duration | Output |
|------|----------|--------|
| Setup & Installation | 5-10 min | Dependencies ready |
| Quick Demo | 30-45 min | Basic metrics |
| Full Training (1 pair) | 3-4 hours | Complete metrics |
| Ablation Study | 5-7 hours | Component analysis |
| Hyperparameter Tuning | 4-6 hours | Optimal α, β, γ |
| Both Language Pairs | 8-12 hours | Full comparison |

**Recommendation**: Start with **Quick Demo** to verify everything works, then run **Full Training**.

---

**You're all set! Choose Colab Pro for best experience. Good luck with your experiments! 🚀**
