# HuggingFace Upload Guide

This guide will help you upload your trained ESA-NMT model to HuggingFace Hub.

## Prerequisites

1. **HuggingFace Account**: Sign up at https://huggingface.co/join
2. **Access Token**: Get from https://huggingface.co/settings/tokens
   - Click "New token"
   - Name: "ESA-NMT Upload"
   - Type: **Write** (important!)
   - Copy the token

3. **Trained Model**: You should have `./checkpoints/final_esa_nmt_hi-te.pt`

## Method 1: Using the Upload Script (Recommended)

### Step 1: Install Required Package

```bash
pip install huggingface_hub
```

### Step 2: Set Your Token (Choose one option)

**Option A: Environment Variable (Recommended)**
```bash
export HF_TOKEN="hf_your_token_here"
```

**Option B: Input When Prompted**
Just run the script and paste when asked.

### Step 3: Run Upload Script

```bash
cd /home/user/ESA-NMT2
python upload_to_huggingface.py
```

The script will:
- ✅ Create repository `sudeshna84/ESA-NMT`
- ✅ Upload model checkpoint
- ✅ Upload configuration
- ✅ Upload training/ablation results
- ✅ Create beautiful model card
- ✅ Make it publicly accessible

### Step 4: Verify Upload

Visit: https://huggingface.co/sudeshna84/ESA-NMT

---

## Method 2: Kaggle Notebook Upload

If running in Kaggle:

```python
# Install package
!pip install -q huggingface_hub

# Set token (in a new cell - keep it private!)
import os
os.environ['HF_TOKEN'] = 'hf_your_token_here'  # Replace with your token

# Run upload
%cd /kaggle/working/ESA-NMT2
!python upload_to_huggingface.py
```

---

## What Gets Uploaded

```
sudeshna84/ESA-NMT/
├── pytorch_model.bin          # Model weights (~600MB)
├── config.json                # Model configuration
├── README.md                  # Model card (auto-generated)
├── training_results.json      # Training metrics
└── ablation_results.json      # Ablation study results
```

---

## Using Your Model

Once uploaded, anyone can use your model:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("sudeshna84/ESA-NMT")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Set languages
tokenizer.src_lang = "hin_Deva"  # Hindi
tokenizer.tgt_lang = "tel_Telu"  # Telugu

# Translate
text = "नमस्ते, आज मौसम बहुत अच्छा है"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["tel_Telu"])
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translation)
```

---

## Model Card Features

Your auto-generated model card includes:

- 📊 **Performance Metrics**: BLEU, METEOR, chrF, ROUGE-L
- 🎭 **Emotion Preservation**: 77.62% accuracy
- 🧠 **Semantic Score**: 0.9236
- 📈 **Ablation Results**: Comparison of all configurations
- 💻 **Code Examples**: Ready-to-use snippets
- 📝 **Citation**: BibTeX format
- 🔗 **Links**: GitHub repository

---

## Troubleshooting

### Issue: "Repository not found"
**Solution**: Make sure you're logged in to the correct account
```bash
huggingface-cli login
```

### Issue: "Permission denied"
**Solution**: Check your token has **write** access (not just read)

### Issue: "Model file too large"
**Solution**: The script uses lightweight version (~600MB). If still too large:
```python
# In upload_to_huggingface.py, modify:
torch.save(model_state, lightweight_path, _use_new_zipfile_serialization=True)
```

### Issue: "Upload timeout"
**Solution**: Try again or upload in chunks using HuggingFace web UI

---

## Making Updates

To update your model later:

```bash
# Make changes to your model
# Then re-run:
python upload_to_huggingface.py
```

The script will overwrite existing files while keeping download statistics.

---

## Privacy Settings

**Default**: Public (anyone can use)

To make private:
1. Go to https://huggingface.co/sudeshna84/ESA-NMT/settings
2. Click "Make private"

---

## Next Steps

After upload:
1. ✅ Share the link: `https://huggingface.co/sudeshna84/ESA-NMT`
2. ✅ Test the model using the web inference widget
3. ✅ Monitor downloads and usage stats
4. ✅ Engage with community in discussions
5. ✅ Add to your research papers/CV

---

## Support

If you encounter issues:
- HuggingFace docs: https://huggingface.co/docs
- GitHub Issues: https://github.com/SSanpui/ESA-NMT2/issues
- HuggingFace Forum: https://discuss.huggingface.co/

---

**Ready to make your model publicly available! 🚀**
