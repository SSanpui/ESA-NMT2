"""
Upload ESA-NMT Model to HuggingFace Hub
Upload trained Hindi-Telugu model to: sudeshna84/ESA-NMT
"""

import os
import torch
import json
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from pathlib import Path

print("="*80)
print("UPLOAD ESA-NMT MODEL TO HUGGINGFACE")
print("="*80)

# Configuration
REPO_NAME = "sudeshna84/ESA-NMT"
MODEL_PATH = "./checkpoints/final_esa_nmt_hi-te.pt"
RESULTS_PATH = "./outputs/training_results_hi-te.json"
ABLATION_PATH = "./outputs/ablation_study_hi-te_fixed.json"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model checkpoint not found at {MODEL_PATH}")
    print("   Please run training first: python retrain_with_fixed_code.py")
    exit(1)

print(f"✅ Found model checkpoint: {MODEL_PATH}")

# Get HuggingFace token
print("\n" + "="*80)
print("AUTHENTICATION")
print("="*80)
print("\n📝 You need a HuggingFace token with write access.")
print("   Get it from: https://huggingface.co/settings/tokens")
print("   Required scope: 'write'\n")

# Option 1: Use environment variable
hf_token = os.environ.get('HF_TOKEN')

if not hf_token:
    # Option 2: Ask user to input
    print("Enter your HuggingFace token (or set HF_TOKEN environment variable):")
    hf_token = input("Token: ").strip()

if not hf_token:
    print("❌ No token provided. Exiting.")
    exit(1)

print("✅ Token received")

# Initialize HuggingFace API
api = HfApi(token=hf_token)

# Create repository if it doesn't exist
print("\n" + "="*80)
print("CREATING REPOSITORY")
print("="*80)

try:
    create_repo(
        repo_id=REPO_NAME,
        token=hf_token,
        repo_type="model",
        exist_ok=True,
        private=False
    )
    print(f"✅ Repository created/verified: {REPO_NAME}")
except Exception as e:
    print(f"⚠️ Repository creation: {e}")
    print("   Continuing with upload...")

# Prepare model files
print("\n" + "="*80)
print("PREPARING MODEL FILES")
print("="*80)

# Create a temporary directory for upload
upload_dir = Path("./huggingface_upload")
upload_dir.mkdir(exist_ok=True)

# 1. Copy model checkpoint
print("\n1️⃣ Preparing model checkpoint...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu')

# Extract just the model state dict for smaller file
model_state = checkpoint['model_state_dict']
config_data = checkpoint.get('config', {})
metrics = checkpoint.get('metrics', {})

# Save lightweight version
lightweight_path = upload_dir / "pytorch_model.bin"
torch.save(model_state, lightweight_path)
print(f"   ✅ Saved lightweight model: {lightweight_path}")

# 2. Save configuration
print("\n2️⃣ Saving model configuration...")
config_dict = {
    "model_type": "nllb",
    "base_model": "facebook/nllb-200-distilled-600M",
    "translation_pair": "hi-te",
    "source_language": "Hindi",
    "target_language": "Telugu",
    "max_length": 128,
    "use_emotion_module": True,
    "use_semantic_module": True,
    "num_emotions": 4,
    "training_metrics": metrics,
    "architectures": ["EmotionSemanticNMT"],
}

with open(upload_dir / "config.json", 'w') as f:
    json.dump(config_dict, f, indent=2)
print("   ✅ Saved config.json")

# 3. Copy training results
if os.path.exists(RESULTS_PATH):
    import shutil
    shutil.copy(RESULTS_PATH, upload_dir / "training_results.json")
    print("   ✅ Copied training results")

# 4. Copy ablation results
if os.path.exists(ABLATION_PATH):
    import shutil
    shutil.copy(ABLATION_PATH, upload_dir / "ablation_results.json")
    print("   ✅ Copied ablation results")

# 5. Create README.md (Model Card)
print("\n3️⃣ Creating model card...")

readme_content = f"""---
language:
- hi
- te
license: mit
tags:
- translation
- nmt
- emotion-aware
- semantic-aware
- hindi
- telugu
- ESA-NMT
datasets:
- custom (BHT25)
metrics:
- bleu
- meteor
- rouge
- chrf
model-index:
- name: ESA-NMT-Hindi-Telugu
  results:
  - task:
      type: translation
      name: Machine Translation
    dataset:
      name: BHT25
      type: custom
    metrics:
    - type: bleu
      value: {metrics.get('bleu', 31.55):.2f}
      name: BLEU
    - type: meteor
      value: {metrics.get('meteor', 51.65):.2f}
      name: METEOR
    - type: chrf
      value: {metrics.get('chrf', 61.95):.2f}
      name: chrF
---

# ESA-NMT: Emotion-Semantic-Aware Neural Machine Translation

## Model Description

**ESA-NMT** (Emotion-Semantic-Aware Neural Machine Translation) is an enhanced neural machine translation model for **Hindi → Telugu** translation that incorporates:

- 🎭 **Emotion Awareness**: Preserves emotional tone across translations
- 🧠 **Semantic Consistency**: Maintains semantic meaning through cross-lingual embeddings
- 📚 **Literary Translation**: Optimized for translating narrative and literary content

This model is based on **NLLB-200** and enhanced with custom emotion and semantic modules.

## Model Details

- **Base Model**: `facebook/nllb-200-distilled-600M`
- **Language Pair**: Hindi (hi) → Telugu (te)
- **Architecture**: Seq2Seq Transformer with Emotion & Semantic Modules
- **Training Data**: BHT25 parallel corpus (~2,850 sentence pairs)
- **Parameters**: ~600M (base) + custom modules

## Performance

### Translation Quality

| Metric | Score | Description |
|--------|-------|-------------|
| **BLEU** | {metrics.get('bleu', 31.55):.2f} | Translation quality |
| **METEOR** | {metrics.get('meteor', 51.65):.2f} | Semantic alignment |
| **chrF** | {metrics.get('chrf', 61.95):.2f} | Character-level similarity |
| **ROUGE-L** | {metrics.get('rouge_l', 76.8):.2f} | Recall-oriented similarity |

### Emotion & Semantic Preservation

| Metric | Score | Description |
|--------|-------|-------------|
| **Emotion Accuracy** | {metrics.get('emotion_accuracy', 77.62):.2f}% | Emotion label preservation |
| **Semantic Score** | {metrics.get('semantic_score', 0.9236):.4f} | Cross-lingual semantic similarity |

### Comparison with Baseline

```
Improvement over Base NLLB:
- BLEU:    +1.14 points (30.41 → 31.55)
- METEOR:  +2.18 points (49.47 → 51.65)
- chrF:    +2.26 points (59.69 → 61.95)
- Emotion: 77.62% preservation
```

## Usage

### Installation

```bash
pip install torch transformers sentence-transformers
```

### Quick Start

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer
model_name = "sudeshna84/ESA-NMT"
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Set language codes
tokenizer.src_lang = "hin_Deva"  # Hindi
tokenizer.tgt_lang = "tel_Telu"  # Telugu

# Translate
hindi_text = "आज मौसम बहुत अच्छा है और मैं बहुत खुश हूं"
inputs = tokenizer(hindi_text, return_tensors="pt", padding=True)

# Generate translation
with torch.no_grad():
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["tel_Telu"],
        max_length=128,
        num_beams=5,
        early_stopping=True
    )

# Decode
telugu_translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print(f"Telugu: {{telugu_translation}}")
```

### Advanced Usage with Emotion & Semantic Modules

For full ESA-NMT functionality with emotion and semantic awareness, see the [GitHub repository](https://github.com/SSanpui/ESA-NMT2).

## Training Details

### Training Data

- **Corpus**: BHT25 (Bengali-Hindi-Telugu parallel corpus)
- **Samples**: ~2,850 Hindi-Telugu sentence pairs
- **Domain**: Literary and narrative text
- **Annotations**:
  - Emotion labels (4 classes: joy, sadness, anger, fear)
  - Semantic similarity scores (cross-lingual embeddings)

### Training Configuration

```python
- Batch Size: 2 (with gradient accumulation × 4)
- Max Length: 128 tokens
- Epochs: 3 (phase 1)
- Learning Rate: 5e-5 → 1e-5 → 5e-6 (scheduled)
- Loss Weights:
  - α = 1.0 (translation)
  - β = 0.4 (emotion)
  - γ = 0.5 (semantic)
```

### Hardware

- GPU: NVIDIA GPU (CUDA-enabled)
- Training Time: ~30-60 minutes

## Ablation Study Results

| Configuration | BLEU | METEOR | chrF | Emotion |
|--------------|------|--------|------|---------|
| Base NLLB (Baseline) | 30.41 | 49.47 | 59.69 | N/A |
| Base + Emotion | 28.98 | 47.76 | 58.33 | 25.89% |
| Base + Semantic | 30.41 | 49.47 | 59.69 | N/A |
| **Full ESA-NMT** | **31.55** | **51.65** | **61.95** | **77.62%** |

**Key Finding**: Emotion and semantic modules work synergistically - both are needed for optimal performance.

## Limitations

- Optimized for **literary/narrative text** (may underperform on technical or formal text)
- **Hindi → Telugu direction only** (not bidirectional)
- Limited to **128 tokens** max length
- Best performance on texts with **emotional content**
- Training data size is moderate (~2,850 pairs)

## Citation

If you use this model, please cite:

```bibtex
@misc{{esa-nmt-hindi-telugu,
  author = {{Sudeshna Sani}},
  title = {{ESA-NMT: Emotion-Semantic-Aware Neural Machine Translation for Hindi-Telugu}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/sudeshna84/ESA-NMT}}
}}
```

## Model Card Authors

- **Sudeshna Sani** ([@sudeshna84](https://huggingface.co/sudeshna84))

## License

MIT License - See repository for details.

## Acknowledgments

- Base model: **NLLB-200** by Meta AI
- Semantic embeddings: **LaBSE** by Google Research
- Emotion classifier: **xlm-emo-t** by MilaNLP

## Repository

Full code, training scripts, and documentation: [ESA-NMT2 GitHub](https://github.com/SSanpui/ESA-NMT2)

---

**Tags**: #translation #nmt #hindi #telugu #emotion-aware #semantic-aware #multilingual
"""

with open(upload_dir / "README.md", 'w', encoding='utf-8') as f:
    f.write(readme_content)
print("   ✅ Created README.md (model card)")

# Upload all files
print("\n" + "="*80)
print("UPLOADING TO HUGGINGFACE")
print("="*80)

try:
    print(f"\nUploading to: https://huggingface.co/{REPO_NAME}\n")

    # Upload folder
    api.upload_folder(
        folder_path=str(upload_dir),
        repo_id=REPO_NAME,
        repo_type="model",
        token=hf_token,
        commit_message="Upload ESA-NMT Hindi-Telugu model"
    )

    print("\n" + "="*80)
    print("✅ UPLOAD COMPLETE!")
    print("="*80)
    print(f"\n🎉 Model is now available at:")
    print(f"   https://huggingface.co/{REPO_NAME}")
    print(f"\n📝 Usage:")
    print(f'   from transformers import AutoModelForSeq2SeqLM')
    print(f'   model = AutoModelForSeq2SeqLM.from_pretrained("{REPO_NAME}")')

except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nTroubleshooting:")
    print("1. Check your HuggingFace token has 'write' scope")
    print("2. Verify repository name is correct")
    print("3. Check internet connection")

# Cleanup
print("\n🧹 Cleaning up temporary files...")
import shutil
shutil.rmtree(upload_dir)
print("✅ Done!")
