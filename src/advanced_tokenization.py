"""
Chapter 3: Advanced Tokenization Examples
These are the tokenization examples from Chapter 3 that were deferred to Chapter 5
"""

import time

from tabulate import tabulate
from transformers import AutoTokenizer

from config import logger


def advanced_padding_examples():
    """Advanced padding examples with different strategies"""
    logger.info("=== Advanced Padding Examples ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    texts = [
        "Short text.",
        "This is a medium length sentence that demonstrates padding.",
        "This is a much longer sentence that will show how padding works with "
        "multiple sentences of different lengths in a batch.",
    ]

    # Padding to max length in batch
    batch_encoding = tokenizer(texts, padding=True, return_tensors="pt")
    logger.info(f"Original texts lengths: {[len(text.split()) for text in texts]}")
    logger.info(f"Padded sequence lengths: {batch_encoding['input_ids'].shape}")
    logger.info(f"Attention mask shape: {batch_encoding['attention_mask'].shape}")

    logger.info("\nAttention masks (1=real token, 0=padding):")
    for i, mask in enumerate(batch_encoding["attention_mask"]):
        logger.info(f"  Text {i+1}: {mask.tolist()}")

    # Different padding strategies
    logger.info("\n=== Padding Strategies ===")

    # Dynamic padding (to longest in batch)
    dynamic_padding = tokenizer(texts, padding="longest", return_tensors="pt")
    logger.info(f"Dynamic padding shape: {dynamic_padding['input_ids'].shape}")

    # Fixed padding to specific length
    fixed_padding = tokenizer(
        texts, padding="max_length", max_length=30, return_tensors="pt"
    )
    logger.info(
        f"Fixed padding shape (max_length=30): {fixed_padding['input_ids'].shape}"
    )

    # No padding
    no_padding = tokenizer(texts, padding=False)
    logger.info(f"No padding: {[len(ids) for ids in no_padding['input_ids']]}")


def advanced_truncation_examples():
    """Advanced truncation examples"""
    logger.info("\n=== Advanced Truncation Examples ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Create a very long text
    long_text = " ".join(["This is a very long sentence."] * 50)

    # Without truncation
    tokens_no_trunc = tokenizer.tokenize(long_text)
    logger.info(f"Without truncation: {len(tokens_no_trunc)} tokens")

    # With truncation to max_length
    tokens_with_trunc = tokenizer(
        long_text, truncation=True, max_length=20, return_tensors="pt"
    )
    logger.info(
        f"With truncation (max_length=20): "
        f"{tokens_with_trunc['input_ids'].shape[1]} tokens"
    )

    # Show truncated tokens
    truncated_tokens = tokenizer.convert_ids_to_tokens(
        tokens_with_trunc["input_ids"][0].tolist()
    )
    logger.info(f"Truncated tokens: {truncated_tokens}")

    # Truncation strategies for sentence pairs
    logger.info("\n=== Truncation Strategies for Sentence Pairs ===")
    question = "What is the capital of France?"
    context = " ".join(["Paris is the capital and most populous city of France."] * 10)

    # Strategy: 'only_second' - truncate only the context
    encoding_only_second = tokenizer(
        question, context, truncation="only_second", max_length=50, return_tensors="pt"
    )
    logger.info(f"'only_second' strategy: {encoding_only_second['input_ids'].shape}")

    # Strategy: 'longest_first' - truncate the longest sequence first
    encoding_longest_first = tokenizer(
        question,
        context,
        truncation="longest_first",
        max_length=50,
        return_tensors="pt",
    )
    logger.info(
        f"'longest_first' strategy: {encoding_longest_first['input_ids'].shape}"
    )


def multiple_sequences_handling():
    """Handling multiple sequences for question-answering"""
    logger.info("\n=== Multiple Sequences (Question-Answering) ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    question = "What is tokenization?"
    context = (
        "Tokenization is the process of breaking down text into smaller units "
        "called tokens. These tokens can be words, subwords, or even characters. "
        "In NLP, tokenization is a crucial preprocessing step that converts "
        "raw text into a format that machine learning models can understand."
    )

    # Encode question and context together
    qa_encoding = tokenizer(
        question, context, padding=True, truncation=True, return_tensors="pt"
    )

    # Convert to tokens to visualize
    tokens = tokenizer.convert_ids_to_tokens(qa_encoding["input_ids"][0])
    token_type_ids = qa_encoding["token_type_ids"][0].tolist()

    logger.info(f"Question: {question}")
    logger.info(f"Context: {context[:100]}...")
    logger.info(f"\nCombined tokens (first 20): {tokens[:20]}...")
    logger.info("\nToken type IDs visualization:")
    logger.info("  0 = Question/First sequence")
    logger.info("  1 = Context/Second sequence")

    # Visualize token types
    for i in range(min(20, len(tokens))):
        logger.info(f"  Token {i:2d}: '{tokens[i]:15}' -> Type {token_type_ids[i]}")

    # Find where question ends and context begins
    sep_positions = [i for i, token in enumerate(tokens) if token == "[SEP]"]
    logger.info(f"\n[SEP] token positions: {sep_positions}")
    logger.info(f"Question ends at position: {sep_positions[0]}")
    logger.info(f"Context starts at position: {sep_positions[0] + 1}")


def offset_mapping_examples():
    """Token-to-character offset mapping examples"""
    logger.info("\n=== Token-to-Character Offset Mapping ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = "Hugging Face's tokenizers are extremely powerful!"
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]

    logger.info(f"Original text: '{text}'")
    logger.info("\nToken to character mapping:")
    logger.info("-" * 60)
    logger.info(f"{'Token':15} {'Text':20} {'Start':>6} {'End':>6}")
    logger.info("-" * 60)

    for token, (start, end) in zip(tokens, offsets, strict=False):
        if start == end:  # Special tokens
            logger.info(f"{token:15} {'[SPECIAL TOKEN]':20} {start:6} {end:6}")
        else:
            original_text = text[start:end]
            logger.info(f"{token:15} {original_text:20} {start:6} {end:6}")

    # Practical use case: Entity extraction
    logger.info("\n=== Practical Example: Entity Extraction ===")

    # Simulate NER predictions (token indices for "Hugging Face")
    entity_token_indices = [1, 2]  # Tokens at positions 1 and 2

    logger.info("Detected entity tokens:")
    entity_chars = []
    for idx in entity_token_indices:
        token = tokens[idx]
        start, end = offsets[idx]
        entity_chars.extend(range(start, end))
        logger.info(f"  Token '{token}' -> '{text[start:end]}'")

    # Reconstruct the entity
    min_char = min(entity_chars)
    max_char = max(entity_chars) + 1
    entity_text = text[min_char:max_char]
    logger.info(f"\nExtracted entity: '{entity_text}'")


def subword_tokenization_comparison():
    """Compare different subword tokenization methods"""
    logger.info("\n=== Subword Tokenization Methods Comparison ===")

    text = (
        "Tokenization is fundamental to NLP. Let's explore BPE, WordPiece, and "
        "SentencePiece algorithms!"
    )

    # Load different tokenizers
    tokenizers = {
        "GPT-2 (BPE)": AutoTokenizer.from_pretrained("gpt2"),
        "BERT (WordPiece)": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "T5 (SentencePiece)": AutoTokenizer.from_pretrained("t5-small"),
        "RoBERTa (BPE)": AutoTokenizer.from_pretrained("roberta-base"),
    }

    logger.info(f"Text: {text}")
    logger.info("\nTokenization comparison:")

    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(text.lower() if "uncased" in name else text)
        logger.info(f"\n{name}:")
        logger.info(f"  Tokens: {tokens}")
        logger.info(f"  Token count: {len(tokens)}")
        logger.info(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Handling rare words
    logger.info("\n=== Handling Rare/Unknown Words ===")
    rare_word = "Supercalifragilisticexpialidocious"

    logger.info(f"\nRare word: '{rare_word}'")
    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(
            rare_word.lower() if "uncased" in name else rare_word
        )
        logger.info(f"{name}: {tokens}")


def oov_handling_examples():
    """Out-of-vocabulary word handling examples"""
    logger.info("\n=== Out-of-Vocabulary (OOV) Word Handling ===")

    # Test with made-up and rare words
    test_texts = [
        "The flibbertigibbet jumped over the moon.",
        "Pneumonoultramicroscopicsilicovolcanoconiosis is a lung disease.",
        "The 🦄 and 🌈 are beautiful.",
        "Contact us at support@企业.com",
    ]

    tokenizers = {
        "BERT": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "GPT-2": AutoTokenizer.from_pretrained("gpt2"),
        "RoBERTa": AutoTokenizer.from_pretrained("roberta-base"),
    }

    for text in test_texts:
        logger.info(f"\nText: '{text}'")
        logger.info("-" * 70)

        for name, tokenizer in tokenizers.items():
            # Get UNK token
            unk_token = getattr(tokenizer, "unk_token", None)

            # Tokenize
            if name == "BERT":
                tokens = tokenizer.tokenize(text.lower())
            else:
                tokens = tokenizer.tokenize(text)

            # Check for UNK tokens
            unk_count = tokens.count(unk_token) if unk_token else 0

            info_msg = f"{name:10} ({len(tokens):2} tokens): "
            if unk_count > 0:
                info_msg += f"⚠️  {unk_count} UNK token(s)! "

            # Show first few tokens
            display_tokens = tokens[:8] + ["..."] if len(tokens) > 8 else tokens
            logger.info(info_msg + str(display_tokens))

    # Key insight
    logger.info("\n💡 Key Insight:")
    logger.info("- BERT uses [UNK] tokens for unknown words/characters")
    logger.info(
        "- GPT-2 and RoBERTa use BPE to break down any word into known subwords"
    )
    logger.info("- This is why BPE-based models handle OOV words better!")


def tiktoken_comparison():
    """Compare HuggingFace tokenizers with TikToken (GPT-3.5/4)"""
    logger.info("\n=== Comparing with TikToken (GPT-3.5/4) ===")

    try:
        import tiktoken
    except ImportError:
        logger.warning("TikToken not installed. Install with: pip install tiktoken")
        return

    # Initialize tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 encoding

    # Test texts
    test_texts = [
        "Hello world!",
        "The transformer architecture revolutionized NLP in 2017.",
        "def tokenize(text): return text.split()",
        "Email: user@example.com, URL: https://example.com",
    ]

    # Load HF tokenizers
    hf_tokenizers = {
        "GPT-2": AutoTokenizer.from_pretrained("gpt2"),
        "BERT": AutoTokenizer.from_pretrained("bert-base-uncased"),
    }

    for text in test_texts:
        logger.info(f"\nText: '{text}'")
        logger.info("-" * 60)

        # TikToken
        tiktoken_ids = encoding.encode(text)
        tiktoken_tokens = [encoding.decode([tid]) for tid in tiktoken_ids]
        logger.info(
            f"TikToken (GPT-3.5/4): {tiktoken_tokens} ({len(tiktoken_tokens)} tokens)"
        )

        # HuggingFace tokenizers
        for name, tokenizer in hf_tokenizers.items():
            tokens = tokenizer.tokenize(text.lower() if "uncased" in name else text)
            logger.info(f"{name}: {tokens} ({len(tokens)} tokens)")

    # Vocabulary size comparison
    logger.info("\n=== Vocabulary Size Comparison ===")
    logger.info(f"TikToken (cl100k_base): {encoding.n_vocab:,} tokens")
    for name, tokenizer in hf_tokenizers.items():
        logger.info(f"{name}: {tokenizer.vocab_size:,} tokens")


def performance_comparison():
    """Compare performance of different tokenizers"""
    logger.info("\n=== Tokenizer Performance Comparison ===")

    # Create test data
    short_text = "Hello, world!"
    medium_text = " ".join(["This is a medium length sentence."] * 10)
    long_text = " ".join(["This is a very long text for performance testing."] * 100)

    test_cases = [
        ("Short (13 chars)", short_text),
        ("Medium (350 chars)", medium_text),
        ("Long (5000 chars)", long_text),
    ]

    # Tokenizers to test
    tokenizers = {
        "BERT": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "GPT-2": AutoTokenizer.from_pretrained("gpt2"),
        "T5": AutoTokenizer.from_pretrained("t5-small"),
    }

    results = []

    for text_name, text in test_cases:
        logger.info(f"\nTesting: {text_name}")

        for tok_name, tokenizer in tokenizers.items():
            # Time the tokenization
            start = time.time()
            for _ in range(100):
                encoded = tokenizer(text, return_tensors=None)
            elapsed = time.time() - start

            num_tokens = len(encoded["input_ids"])
            tokens_per_sec = (num_tokens * 100) / elapsed

            results.append(
                [
                    text_name,
                    tok_name,
                    num_tokens,
                    f"{elapsed:.3f}s",
                    f"{tokens_per_sec:,.0f}",
                ]
            )

            logger.info(f"  {tok_name}: {num_tokens} tokens in {elapsed:.3f}s")

    # Display results as table
    logger.info("\n=== Performance Summary ===")
    headers = ["Text", "Tokenizer", "Tokens", "Time (100 runs)", "Tokens/sec"]
    logger.info("\n" + tabulate(results, headers=headers, tablefmt="grid"))

    logger.info("\n💡 Performance Insights:")
    logger.info("- Fast tokenizers (Rust-based) are significantly faster")
    logger.info("- Token count varies by algorithm (affects model cost/speed)")
    logger.info("- Always use 'fast' tokenizers when available")


def batch_encoding_strategies():
    """Different batch encoding strategies"""
    logger.info("\n=== Batch Encoding Strategies ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    batch_texts = [
        "First sentence.",
        "Second sentence is a bit longer.",
        "Third sentence is the longest of all the sentences in this batch.",
    ]

    # Strategy 1: Dynamic padding (to longest in batch)
    dynamic_padding = tokenizer(batch_texts, padding="longest", return_tensors="pt")
    logger.info(f"Dynamic padding shape: {dynamic_padding['input_ids'].shape}")

    # Strategy 2: Fixed padding (to specific length)
    fixed_padding = tokenizer(
        batch_texts, padding="max_length", max_length=30, return_tensors="pt"
    )
    logger.info(
        f"Fixed padding shape (max_length=30): {fixed_padding['input_ids'].shape}"
    )

    # Strategy 3: No padding (returns list)
    no_padding = tokenizer(batch_texts, padding=False)
    logger.info(f"No padding lengths: {[len(ids) for ids in no_padding['input_ids']]}")

    # Show attention masks
    logger.info("\nAttention masks for dynamic padding:")
    for i, mask in enumerate(dynamic_padding["attention_mask"]):
        logger.info(f"  Sentence {i+1}: {mask.tolist()}")


def special_tokens_deep_dive():
    """Deep dive into special tokens handling"""
    logger.info("\n=== Special Tokens Deep Dive ===")

    # Compare special tokens across models
    models = ["bert-base-uncased", "gpt2", "roberta-base", "t5-small"]

    logger.info("Special tokens comparison:")
    logger.info("-" * 80)

    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"\n{model_name}:")

        # List all special tokens
        special_tokens = {
            "PAD": tokenizer.pad_token,
            "UNK": tokenizer.unk_token,
            "CLS": getattr(tokenizer, "cls_token", None),
            "SEP": getattr(tokenizer, "sep_token", None),
            "MASK": getattr(tokenizer, "mask_token", None),
            "BOS": getattr(tokenizer, "bos_token", None),
            "EOS": getattr(tokenizer, "eos_token", None),
        }

        for name, token in special_tokens.items():
            if token:
                token_id = tokenizer.convert_tokens_to_ids(token)
                logger.info(f"  {name}: '{token}' (ID: {token_id})")

    # Example: Adding custom special tokens
    logger.info("\n=== Adding Custom Special Tokens ===")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Add domain-specific special tokens
    special_tokens_dict = {
        "additional_special_tokens": ["<MEDICAL>", "<DIAGNOSIS>", "<TREATMENT>"]
    }
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added} special tokens")

    # Test with custom tokens
    text = "<DIAGNOSIS> Patient has <MEDICAL> pneumonia <TREATMENT> antibiotics"
    tokens = tokenizer.tokenize(text)
    logger.info(f"\nText with custom tokens: {text}")
    logger.info(f"Tokens: {tokens}")


def run_all_examples():
    """Run all Chapter 3 advanced tokenization examples"""
    logger.info("Running Chapter 3 Advanced Tokenization Examples")

    # Padding and truncation
    advanced_padding_examples()
    advanced_truncation_examples()

    # Multiple sequences
    multiple_sequences_handling()

    # Offset mapping
    offset_mapping_examples()

    # Subword tokenization
    subword_tokenization_comparison()

    # OOV handling
    oov_handling_examples()

    # TikToken comparison
    tiktoken_comparison()

    # Performance
    performance_comparison()

    # Batch encoding strategies
    batch_encoding_strategies()

    # Special tokens
    special_tokens_deep_dive()

    logger.info("\n✅ All Chapter 3 advanced tokenization examples completed!")


if __name__ == "__main__":
    run_all_examples()
