"""
Fix for CUDA Error: t >= 0 && t < n_classes failed
This error occurs because padding tokens in labels are not masked.
"""

# The error happens because:
# 1. target_input_ids contains padding tokens (e.g., pad_token_id = 1)
# 2. These are used as labels in CrossEntropyLoss
# 3. If pad_token_id >= vocab_size or is invalid, we get out-of-bounds error

# Solution: Mask padding positions in labels with -100 (ignored by CrossEntropyLoss)

# ==============================================================================
# FIX 1: Modify forward() method in EmotionSemanticNMT class
# ==============================================================================

# REPLACE THIS (emotion_semantic_nmt_enhanced.py line 522-531):
"""
    if target_input_ids is not None:
        # Training mode
        outputs = self.base_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            decoder_input_ids=target_input_ids[:, :-1],
            decoder_attention_mask=target_attention_mask[:, :-1] if target_attention_mask is not None else None,
            labels=target_input_ids[:, 1:].contiguous(),
            output_hidden_states=True
        )
"""

# WITH THIS:
"""
    if target_input_ids is not None:
        # Training mode

        # Prepare labels: mask padding positions with -100
        labels = target_input_ids[:, 1:].contiguous()

        # Mask padding tokens (set to -100 so they're ignored in loss)
        if target_attention_mask is not None:
            labels_attention = target_attention_mask[:, 1:]
            labels = labels.masked_fill(labels_attention == 0, -100)

        # Also explicitly mask pad_token_id
        if self.tokenizer.pad_token_id is not None:
            labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

        outputs = self.base_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            decoder_input_ids=target_input_ids[:, :-1],
            decoder_attention_mask=target_attention_mask[:, :-1] if target_attention_mask is not None else None,
            labels=labels,  # ← Use masked labels
            output_hidden_states=True
        )
"""

# ==============================================================================
# FIX 2: Add token validation in dataset
# ==============================================================================

# Add this validation in dataset_with_annotations.py after tokenization:

"""
# Validate token IDs are in valid range
vocab_size = self.tokenizer.vocab_size
source_max = source_tokens['input_ids'].max().item()
target_max = target_tokens['input_ids'].max().item()

if source_max >= vocab_size:
    print(f"⚠️ WARNING: Source token ID {source_max} >= vocab_size {vocab_size}")
    source_tokens['input_ids'] = torch.clamp(source_tokens['input_ids'], 0, vocab_size - 1)

if target_max >= vocab_size:
    print(f"⚠️ WARNING: Target token ID {target_max} >= vocab_size {vocab_size}")
    target_tokens['input_ids'] = torch.clamp(target_tokens['input_ids'], 0, vocab_size - 1)
"""

print(__doc__)
