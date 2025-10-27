#!/usr/bin/env python3
"""
Fix emotion labels in annotated CSV
Maps emotions 4-7 to 0-3 for 4-emotion model
"""

import pandas as pd
import shutil

print("🔧 Fixing emotion labels for 4-emotion model...")
print("="*60)

# Backup original file
shutil.copy('BHT25_All_annotated.csv', 'BHT25_All_annotated_backup.csv')
print("✅ Backup created: BHT25_All_annotated_backup.csv")

# Load CSV
df = pd.read_csv('BHT25_All_annotated.csv')
print(f"✅ Loaded {len(df)} rows")

# Show original distribution
print("\n📊 Original Emotion Distribution:")
emotion_counts_before = df['emotion_bn'].value_counts().sort_index()
emotion_names = ['joy', 'sadness', 'anger', 'fear', 'trust', 'disgust', 'surprise', 'anticipation']
for emotion_id in sorted(emotion_counts_before.index):
    count = emotion_counts_before[emotion_id]
    pct = count / len(df) * 100
    name = emotion_names[emotion_id] if emotion_id < 8 else 'UNKNOWN'
    print(f"  {emotion_id}: {name:12s} - {count:5d} ({pct:5.1f}%)")

# Map emotions 4-7 to 0-3
# Based on semantic similarity:
# 4 (trust) → 0 (joy) - both positive emotions
# 5 (disgust) → 2 (anger) - both negative, reactive emotions
# 6 (surprise) → 0 (joy) - often positive in literature
# 7 (anticipation) → 0 (joy) - forward-looking, often positive

emotion_mapping = {
    0: 0,  # joy → joy
    1: 1,  # sadness → sadness
    2: 2,  # anger → anger
    3: 3,  # fear → fear
    4: 0,  # trust → joy (positive social emotion)
    5: 2,  # disgust → anger (negative reactive emotion)
    6: 0,  # surprise → joy (unexpected, often positive)
    7: 0,  # anticipation → joy (hopeful, forward-looking)
}

print("\n🔄 Mapping Strategy:")
print("  4 (trust) → 0 (joy)")
print("  5 (disgust) → 2 (anger)")
print("  6 (surprise) → 0 (joy)")
print("  7 (anticipation) → 0 (joy)")

# Apply mapping
df['emotion_bn'] = df['emotion_bn'].map(emotion_mapping)
df['emotion_hi'] = df['emotion_hi'].map(emotion_mapping)
df['emotion_te'] = df['emotion_te'].map(emotion_mapping)

# Show new distribution
print("\n📊 New Emotion Distribution (4 emotions):")
emotion_counts_after = df['emotion_bn'].value_counts().sort_index()
emotion_names_4 = ['joy', 'sadness', 'anger', 'fear']
for emotion_id in range(4):
    count = emotion_counts_after.get(emotion_id, 0)
    pct = count / len(df) * 100
    print(f"  {emotion_id}: {emotion_names_4[emotion_id]:12s} - {count:5d} ({pct:5.1f}%)")

# Verify no invalid labels
max_emotion = df['emotion_bn'].max()
if max_emotion > 3:
    print(f"\n❌ ERROR: Still have emotion labels > 3!")
else:
    print(f"\n✅ All emotion labels now in valid range [0-3]")

# Save
df.to_csv('BHT25_All_annotated.csv', index=False)
print(f"\n✅ Fixed file saved: BHT25_All_annotated.csv")
print(f"   Original backed up to: BHT25_All_annotated_backup.csv")

print("\n" + "="*60)
print("✅ Fix complete! You can now train the model without CUDA errors.")
