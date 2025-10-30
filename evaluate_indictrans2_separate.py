"""
IndicTrans2 Separate Evaluation Script
Run this AFTER the main evaluation to add IndicTrans2 results

This script:
1. Uses your HuggingFace token
2. Loads IndicTrans2 directly (without custom modules)
3. Evaluates on test set
4. Saves results separately
"""

import os
os.chdir('/kaggle/working/ESA-NMT')

import torch
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import login
import sacrebleu
from rouge_score import rouge_scorer

# =============================================================================
# STEP 1: Login to HuggingFace
# =============================================================================

print("="*80)
print("IndicTrans2 Evaluation")
print("="*80)

print("\n🔐 HuggingFace Authentication")
print("Paste your HF token (starts with hf_...)")
HF_TOKEN = input("Token: ").strip()

if not HF_TOKEN:
    print("❌ No token provided. Exiting.")
    exit()

try:
    login(token=HF_TOKEN)
    print("✅ Logged in to HuggingFace")
except Exception as e:
    print(f"❌ Login failed: {e}")
    exit()

# =============================================================================
# STEP 2: Load IndicTrans2 Model
# =============================================================================

print("\n📥 Loading IndicTrans2 model...")
print("   This may take a few minutes...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    # IndicTrans2 model - use the correct model path
    # For Bengali to Indic languages (Hindi/Telugu)
    model_name = "ai4bharat/indictrans2-indic-indic-1B"  # For Indic-to-Indic

    print(f"   Loading: {model_name}")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_auth_token=HF_TOKEN
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_auth_token=HF_TOKEN
    )

    print(f"✅ IndicTrans2 loaded on {device}")

except Exception as e:
    print(f"❌ Failed to load IndicTrans2: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you requested access at:")
    print("   https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B")
    print("2. Wait for approval (usually instant)")
    print("3. Try again with correct token")
    exit()

# =============================================================================
# STEP 3: Setup METEOR
# =============================================================================

print("\n📦 Setting up METEOR...")
try:
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.translate.meteor_score import meteor_score
    METEOR_AVAILABLE = True
    print("✅ METEOR ready")
except:
    METEOR_AVAILABLE = False
    print("⚠️ METEOR not available")

# =============================================================================
# STEP 4: Load Test Dataset
# =============================================================================

print("\n📊 Loading test dataset...")

import pandas as pd

TRANSLATION_PAIR = input("\nEnter translation pair (bn-hi or bn-te): ").strip() or 'bn-hi'

df = pd.read_csv('BHT25_All_annotated.csv')
print(f"✅ Loaded {len(df)} samples")

# Get test split (15% of data, matching main evaluation)
test_start_idx = int(0.85 * len(df))
test_df = df.iloc[test_start_idx:].reset_index(drop=True)

print(f"✅ Test samples: {len(test_df)}")

# Get source and target columns
src_lang, tgt_lang = TRANSLATION_PAIR.split('-')
src_col = src_lang  # 'bn'
tgt_col = tgt_lang  # 'hi' or 'te'

# =============================================================================
# STEP 5: Translate with IndicTrans2
# =============================================================================

print(f"\n🔄 Translating {TRANSLATION_PAIR}...")
print("   This will take 10-15 minutes...")

predictions = []
references = []

model.eval()

# Language codes for IndicTrans2
# Bengali: ben_Beng, Hindi: hin_Deva, Telugu: tel_Telu
lang_codes = {
    'bn': 'ben_Beng',
    'hi': 'hin_Deva',
    'te': 'tel_Telu'
}

src_lang_code = lang_codes[src_col]
tgt_lang_code = lang_codes[tgt_col]

with torch.no_grad():
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Translating"):
        source_text = str(row[src_col]).strip()
        target_text = str(row[tgt_col]).strip()

        if len(source_text) < 3:
            continue

        try:
            # Prepare input with language tags
            input_text = f"{src_lang_code} {source_text} {tgt_lang_code}"

            # Tokenize
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)

            # Generate
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )

            # Decode
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

            predictions.append(translation)
            references.append(target_text)

        except Exception as e:
            # Skip problematic samples
            continue

print(f"\n✅ Translated {len(predictions)} samples")

# =============================================================================
# STEP 6: Calculate Metrics
# =============================================================================

print("\n📊 Calculating metrics...")

# BLEU
bleu = sacrebleu.corpus_bleu(predictions, [references])
bleu_score = bleu.score

# METEOR
if METEOR_AVAILABLE:
    try:
        meteor_scores = [
            meteor_score([ref.split()], pred.split())
            for ref, pred in zip(references, predictions)
        ]
        meteor_score_val = np.mean(meteor_scores) * 100
    except:
        meteor_score_val = 0.0
else:
    meteor_score_val = 0.0

# chrF
chrf = sacrebleu.corpus_chrf(predictions, [references])
chrf_score = chrf.score

# ROUGE-L
rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_scores = [
    rouge_scorer_obj.score(ref, pred)['rougeL'].fmeasure
    for ref, pred in zip(references, predictions)
]
rouge_l_score = np.mean(rouge_scores) * 100

# =============================================================================
# STEP 7: Display and Save Results
# =============================================================================

print("\n" + "="*80)
print(f"IndicTrans2 Results for {TRANSLATION_PAIR.upper()}")
print("="*80)

results = {
    'model': 'IndicTrans2',
    'translation_pair': TRANSLATION_PAIR,
    'bleu': float(bleu_score),
    'meteor': float(meteor_score_val),
    'chrf': float(chrf_score),
    'rouge_l': float(rouge_l_score),
    'num_samples': len(predictions)
}

print(f"\nBLEU:    {results['bleu']:.2f}")
print(f"METEOR:  {results['meteor']:.2f}")
print(f"chrF:    {results['chrf']:.2f}")
print(f"ROUGE-L: {results['rouge_l']:.2f}")
print(f"\nSamples: {results['num_samples']}")

# Save results
os.makedirs('./outputs', exist_ok=True)
output_file = f'./outputs/indictrans2_results_{TRANSLATION_PAIR}.json'

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Saved: {output_file}")

# Copy to /kaggle/working
import shutil
shutil.copy(output_file, f'/kaggle/working/indictrans2_results_{TRANSLATION_PAIR}.json')
print(f"💾 Copied to: /kaggle/working/indictrans2_results_{TRANSLATION_PAIR}.json")

print("\n" + "="*80)
print("✅ IndicTrans2 Evaluation Complete!")
print("="*80)

print("\n📥 Next Steps:")
print("1. Download indictrans2_results_{}.json".format(TRANSLATION_PAIR))
print("2. Combine with main evaluation results")
print("3. For other translation pair, run this script again")

print("\n💡 To create final comparison table:")
print("   Merge this result with comparison_3models_{}.json".format(TRANSLATION_PAIR))
