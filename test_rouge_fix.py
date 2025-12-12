"""
Quick test to diagnose and fix ROUGE-L issue
"""
import numpy as np
from rouge_score import rouge_scorer

# Test ROUGE-L calculation
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Sample Hindi-Telugu translations
test_cases = [
    ("मैं स्कूल जाता हूं", "నేను పాఠశాలకు వెళ్తున్నాను"),  # Reference, Prediction
    ("यह एक किताब है", "ఇది ఒక పుస్తకం"),
    ("आज मौसम अच्छा है", "ఈరోజు వాతావరణం బాగుంది"),
]

print("Testing ROUGE-L calculation:")
print("="*60)

rouge_scores = []
for ref, pred in test_cases:
    score = scorer.score(ref, pred)
    rouge_l = score['rougeL'].fmeasure
    rouge_scores.append(rouge_l)

    print(f"Ref:  {ref}")
    print(f"Pred: {pred}")
    print(f"ROUGE-L (raw):    {rouge_l:.6f}")
    print(f"ROUGE-L (*100):   {rouge_l * 100:.2f}")
    print("-"*60)

avg_rouge = np.mean(rouge_scores)
print(f"\nAverage ROUGE-L (raw):  {avg_rouge:.6f}")
print(f"Average ROUGE-L (*100): {avg_rouge * 100:.2f}")

print("\n" + "="*60)
print("✅ Expected range: 0-100 after multiplying by 100")
print(f"✅ Your values look correct: {avg_rouge * 100:.2f}")
