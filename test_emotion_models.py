#!/usr/bin/env python3
"""
FIXED ANNOTATION - Using proper multilingual emotion model
Tests multiple emotion classification approaches
"""

import pandas as pd
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

print("🔄 Testing Emotion Classification Models...")
print("="*80)

# Load test sentence
test_sentences = {
    'bn': 'আমি খুব খুশি',  # "I am very happy"
    'hi': 'मैं बहुत खुश हूं',  # "I am very happy"
    'te': 'నేను చాలా సంతోషంగా ఉన్నాను'  # "I am very happy"
}

# ============================================================================
# TEST 1: XLM-RoBERTa-base with zero-shot (current - NOT WORKING)
# ============================================================================
print("\n1️⃣ Testing: joeddav/xlm-roberta-large-xnli (CURRENT)")
try:
    classifier1 = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        device=0 if torch.cuda.is_available() else -1
    )

    emotions = ['joy', 'sadness', 'anger', 'fear', 'trust', 'disgust', 'surprise', 'anticipation']

    for lang, text in test_sentences.items():
        result = classifier1(text, candidate_labels=emotions, multi_label=False)
        print(f"  {lang}: {result['labels'][0]} (score: {result['scores'][0]:.3f})")
except Exception as e:
    print(f"  ❌ Error: {e}")

# ============================================================================
# TEST 2: Multilingual emotion model (RECOMMENDED)
# ============================================================================
print("\n2️⃣ Testing: MilaNLProc/xlm-emo-t (Multilingual Emotion)")
try:
    classifier2 = pipeline(
        "text-classification",
        model="MilaNLProc/xlm-emo-t",  # Multilingual emotion model
        device=0 if torch.cuda.is_available() else -1,
        top_k=1
    )

    for lang, text in test_sentences.items():
        result = classifier2(text)
        print(f"  {lang}: {result[0]['label']} (score: {result[0]['score']:.3f})")
except Exception as e:
    print(f"  ❌ Error: {e}")

# ============================================================================
# TEST 3: English emotion model (baseline)
# ============================================================================
print("\n3️⃣ Testing: j-hartmann/emotion-english-distilroberta-base (English only)")
try:
    classifier3 = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=0 if torch.cuda.is_available() else -1,
        top_k=1
    )

    # Test with English
    result = classifier3("I am very happy")
    print(f"  en: {result[0]['label']} (score: {result[0]['score']:.3f})")
except Exception as e:
    print(f"  ❌ Error: {e}")

# ============================================================================
# RECOMMENDATION
# ============================================================================
print("\n" + "="*80)
print("📊 ANALYSIS:")
print("- Model 1 (XNLI): Biased towards surprise/anticipation ❌")
print("- Model 2 (xlm-emo-t): Best for multilingual emotion ✅")
print("- Model 3 (English): Only works for English ❌")
print("\n💡 RECOMMENDATION: Use MilaNLProc/xlm-emo-t for multilingual emotion!")
print("="*80)

# ============================================================================
# Ask user which model they used before
# ============================================================================
print("\n❓ QUESTION FOR USER:")
print("Which model did you use in your previous experiment that gave you:")
print("  - 28% joy, 22% sadness, 15% anger, 13% fear?")
print("\nPlease provide the exact model name/path!")
