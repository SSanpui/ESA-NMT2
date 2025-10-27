#!/usr/bin/env python3
"""
Pre-annotate BHT25 dataset with emotion labels and semantic scores
Uses XLM-RoBERTa for cross-lingual emotion classification and LaBSE for semantic similarity

This creates a properly annotated dataset for training ESA-NMT
Supports Bengali, Hindi, and Telugu text
"""

import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import json

print("🔄 Loading annotation models...")
print("   Using multilingual emotion model for literary content...")

# Load emotion classifier - Multilingual emotion model for cross-lingual classification
# Using MilaNLProc/xlm-emo-t which supports Bengali, Hindi, Telugu
# Suitable for literary/narrative content
emotion_classifier = pipeline(
    "text-classification",
    model="MilaNLProc/xlm-emo-t",  # Multilingual emotion model (40+ languages)
    device=0 if torch.cuda.is_available() else -1,
    top_k=1
)

# Load semantic similarity model (LaBSE)
semantic_model = SentenceTransformer('sentence-transformers/LaBSE')
if torch.cuda.is_available():
    semantic_model = semantic_model.to('cuda')

print("✅ Models loaded!")

# Our target: 4 emotion classes (MilaNLProc/xlm-emo-t outputs)
# Based on basic emotion theory (joy, sadness, anger, fear)
EMOTION_NAMES = ['joy', 'sadness', 'anger', 'fear']

# Emotion label mapping for MilaNLProc/xlm-emo-t
EMOTION_MAP = {
    'joy': 0,
    'sadness': 1,
    'anger': 2,
    'fear': 3,
    # Alternative labels that might appear
    'happy': 0,      # → joy
    'happiness': 0,  # → joy
    'sad': 1,        # → sadness
    'sorrow': 1,     # → sadness
    'angry': 2,      # → anger
    'rage': 2,       # → anger
    'scared': 3,     # → fear
    'afraid': 3,     # → fear
    'anxiety': 3,    # → fear
}

def get_emotion_label(text):
    """
    Get emotion label using MilaNLProc/xlm-emo-t multilingual emotion classifier
    Works with Bengali, Hindi, Telugu text
    Suitable for literary/narrative content
    """
    try:
        # Classify emotion (returns top prediction)
        results = emotion_classifier(text[:512])  # Truncate to 512 chars

        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                # top_k returns nested list
                top_emotion = results[0][0]['label'].lower()
            else:
                # Single prediction
                top_emotion = results[0]['label'].lower()
        else:
            top_emotion = 'joy'  # Default

        # Clean label (remove LABEL_ prefix if present)
        top_emotion = top_emotion.replace('label_', '')

        # Map to our 8 classes
        return EMOTION_MAP.get(top_emotion, 0)

    except Exception as e:
        print(f"Error in emotion classification: {e}")
        return 0  # Default to joy

def get_semantic_similarity(text1, text2):
    """Calculate semantic similarity using LaBSE"""
    try:
        with torch.no_grad():
            embeddings = semantic_model.encode([text1, text2], convert_to_tensor=True)
            similarity = torch.nn.functional.cosine_similarity(
                embeddings[0].unsqueeze(0),
                embeddings[1].unsqueeze(0)
            ).item()
        return similarity
    except Exception as e:
        print(f"Error in semantic similarity: {e}")
        return 0.0

def annotate_dataset(csv_path, output_path):
    """Annotate BHT25 dataset with emotions and semantic scores"""

    print(f"\n📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace('﻿', '')

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Remove NaN
    df = df.dropna(subset=['bn', 'hi', 'te'])
    print(f"After removing NaN: {df.shape}")

    # Annotate each row
    print("\n🔄 Annotating dataset (this may take a while)...")

    annotations = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        bn_text = str(row['bn']).strip()
        hi_text = str(row['hi']).strip()
        te_text = str(row['te']).strip()

        # Skip empty
        if len(bn_text) < 3 or len(hi_text) < 3 or len(te_text) < 3:
            continue

        # Get emotion labels using XLM-RoBERTa (supports Bengali, Hindi, Telugu)
        emotion_bn = get_emotion_label(bn_text)
        emotion_hi = get_emotion_label(hi_text)
        emotion_te = get_emotion_label(te_text)

        # Get semantic similarities
        # bn-hi similarity
        semantic_bn_hi = get_semantic_similarity(bn_text, hi_text)

        # bn-te similarity
        semantic_bn_te = get_semantic_similarity(bn_text, te_text)

        # hi-te similarity (for reference)
        semantic_hi_te = get_semantic_similarity(hi_text, te_text)

        annotations.append({
            'bn': bn_text,
            'hi': hi_text,
            'te': te_text,
            'emotion_bn': emotion_bn,
            'emotion_hi': emotion_hi,
            'emotion_te': emotion_te,
            'semantic_bn_hi': semantic_bn_hi,
            'semantic_bn_te': semantic_bn_te,
            'semantic_hi_te': semantic_hi_te,
        })

        # Save intermediate results every 100 rows
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} rows...")
            temp_df = pd.DataFrame(annotations)
            temp_df.to_csv(output_path.replace('.csv', '_temp.csv'), index=False)

    # Create annotated dataframe
    annotated_df = pd.DataFrame(annotations)

    # Save
    annotated_df.to_csv(output_path, index=False)
    print(f"\n✅ Annotated dataset saved to: {output_path}")

    # Print statistics
    print("\n📊 Annotation Statistics:")
    print(f"Total samples: {len(annotated_df)}")
    print(f"\nEmotion distribution (Bengali):")
    print("MilaNLProc/xlm-emo-t outputs 4 primary emotions:")
    print("Expected for traditional literary content:")
    print("  - Joy: 30-40% (romantic moments, celebrations, happy endings)")
    print("  - Sadness: 20-30% (tragic events, separation, loss)")
    print("  - Anger: 15-25% (conflict, moral indignation, injustice)")
    print("  - Fear: 15-25% (suspense, uncertainty, danger)")
    print()
    print("Actual distribution:")
    emotion_counts = pd.Series([a['emotion_bn'] for a in annotations]).value_counts()
    for emotion_id in range(4):  # Only 4 emotions now
        count = emotion_counts.get(emotion_id, 0)
        percentage = (count / len(annotated_df) * 100) if len(annotated_df) > 0 else 0
        print(f"  {EMOTION_NAMES[emotion_id]:12s}: {count:4d} ({percentage:5.1f}%)")

    print(f"\nSemantic similarity (bn-hi):")
    print(f"  Mean: {annotated_df['semantic_bn_hi'].mean():.4f}")
    print(f"  Std:  {annotated_df['semantic_bn_hi'].std():.4f}")
    print(f"  Min:  {annotated_df['semantic_bn_hi'].min():.4f}")
    print(f"  Max:  {annotated_df['semantic_bn_hi'].max():.4f}")

    print(f"\nSemantic similarity (bn-te):")
    print(f"  Mean: {annotated_df['semantic_bn_te'].mean():.4f}")
    print(f"  Std:  {annotated_df['semantic_bn_te'].std():.4f}")
    print(f"  Min:  {annotated_df['semantic_bn_te'].min():.4f}")
    print(f"  Max:  {annotated_df['semantic_bn_te'].max():.4f}")

    return annotated_df

if __name__ == "__main__":
    # Annotate the dataset
    input_csv = "BHT25_All.csv"
    output_csv = "BHT25_All_annotated.csv"

    annotated_df = annotate_dataset(input_csv, output_csv)

    print("\n✅ Annotation complete!")
    print(f"Use '{output_csv}' for training your ESA-NMT model")
