"""
Chapter 5: Tokenization Debugging
Debugging and visualizing tokenization issues
"""

from transformers import AutoModel, AutoTokenizer

from config import logger


def visualize_tokenization_with_offsets():
    """Visualize tokenization output and alignment (from article5.md lines 809-820)"""
    logger.info("=== Visualizing Tokenization with Offsets ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = "Let's test: 🤖 transformers!"
    output = tokenizer(text, return_offsets_mapping=True, return_tensors=None)
    tokens = tokenizer.convert_ids_to_tokens(output["input_ids"])
    offsets = output["offset_mapping"]

    logger.info(f"Original text: '{text}'")
    logger.info("\nToken breakdown:")
    for token, (start, end) in zip(tokens, offsets, strict=False):
        if start == end:  # Special tokens
            logger.info(f"  {token}: [SPECIAL TOKEN]")
        else:
            logger.info(f"  {token}: [{start}, {end}] -> '{text[start:end]}'")

    return tokens, offsets


def detect_tokenizer_model_mismatch():
    """Detect tokenizer-model mismatch (from article5.md lines 359-375)"""
    logger.info("\n=== Detecting Tokenizer-Model Mismatch ===")

    # Intentionally use mismatched tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("roberta-base")

    text = "Tokenization mismatch!"
    inputs = tokenizer(text, return_tensors="pt")

    try:
        # This will likely fail or produce unexpected results
        outputs = model(**inputs)
        logger.warning("Model processed mismatched input (this is unusual!)")
        logger.info(f"Output shape: {outputs.last_hidden_state.shape}")
    except Exception as e:
        logger.error(f"Error due to mismatched tokenizer and model: {e}")
        logger.info("✅ Successfully caught the mismatch error!")

    # Show the correct pairing
    logger.info("\nCorrect pairing:")
    correct_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    correct_inputs = correct_tokenizer(text, return_tensors="pt")
    correct_outputs = model(**correct_inputs)
    logger.info(f"Correct output shape: {correct_outputs.last_hidden_state.shape}")


def debug_special_tokens():
    """Debug special token handling"""
    logger.info("\n=== Debugging Special Tokens ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Check all special tokens
    logger.info("Special tokens mapping:")
    for token_name, token in tokenizer.special_tokens_map.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.info(f"  {token_name}: '{token}' -> ID: {token_id}")

    # Test different text formats
    test_cases = [
        "Simple sentence",
        "[CLS] Manual special tokens [SEP]",
        "Multiple. Sentences. Here.",
        "",  # Empty string
    ]

    for text in test_cases:
        logger.info(f"\nTokenizing: '{text}'")
        encoded = tokenizer(text, add_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        logger.info(f"  Tokens: {tokens}")

        # Check if special tokens are in expected positions
        if tokens:
            logger.info(f"  First token: '{tokens[0]}' (should be [CLS])")
            logger.info(f"  Last token: '{tokens[-1]}' (should be [SEP])")


def analyze_unknown_tokens():
    """Analyze unknown token generation"""
    logger.info("\n=== Analyzing Unknown Tokens ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Text with potentially unknown tokens
    test_texts = [
        "Normal English text",
        "Emojis: 😀 🚀 🤖",
        "Special chars: ™ © ® µ",
        "Mixed: Hello世界Bonjour",
        "Medical: pneumonoultramicroscopicsilicovolcanoconiosis",
        "Code: def foo(x): return x**2",
        "Email: user@example.com",
        "URL: https://example.com/path",
    ]

    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Check for unknown tokens
        unk_token_id = tokenizer.unk_token_id
        unk_count = token_ids.count(unk_token_id)

        logger.info(f"\nText: '{text}'")
        logger.info(f"  Tokens: {tokens}")
        logger.info(f"  Unknown tokens: {unk_count}")

        if unk_count > 0:
            unk_positions = [
                i for i, tid in enumerate(token_ids) if tid == unk_token_id
            ]
            logger.info(f"  Unknown at positions: {unk_positions}")


def debug_padding_and_truncation():
    """Debug padding and truncation behavior"""
    logger.info("\n=== Debugging Padding and Truncation ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Sentences of different lengths
    sentences = [
        "Short",
        "This is a medium length sentence.",
        (
            "This is a much longer sentence that will definitely exceed the "
            "maximum length limit we set for truncation testing purposes."
        ),
    ]

    max_length = 10

    for strategy in ["longest", "max_length", "do_not_pad"]:
        logger.info(f"\nPadding strategy: {strategy}")

        try:
            if strategy == "do_not_pad":
                encoded = tokenizer(
                    sentences,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_tensors=None,
                )
            else:
                encoded = tokenizer(
                    sentences,
                    padding=strategy,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

            for i, sent in enumerate(sentences):
                if strategy == "do_not_pad":
                    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][i])
                    logger.info(
                        f"  '{sent[:20]}...': {len(encoded['input_ids'][i])} tokens"
                    )
                else:
                    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][i])
                    logger.info(f"  '{sent[:20]}...': {tokens}")

        except Exception as e:
            logger.error(f"  Error with strategy '{strategy}': {e}")


def debug_tokenization_edge_cases():
    """Debug various edge cases in tokenization"""
    logger.info("\n=== Debugging Edge Cases ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    edge_cases = {
        "Empty string": "",
        "Only spaces": "     ",
        "Only punctuation": "!!!???...",
        "Mixed scripts": "Hello世界مرحبا",
        "Numbers": "123.456 1e-10 0xFF",
        "Repeated chars": "aaaaaaaaaa",
        "Unicode": "café naïve résumé",
        "Newlines": "Line1\nLine2\nLine3",
        "Tabs": "Col1\tCol2\tCol3",
        "HTML": "<div>Hello</div>",
        "Very long word": "a" * 100,
    }

    for name, text in edge_cases.items():
        logger.info(f"\n{name}: '{text[:50]}{'...' if len(text) > 50 else ''}'")

        try:
            tokens = tokenizer.tokenize(text)
            encoded = tokenizer(text, add_special_tokens=True)

            logger.info(f"  Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            logger.info(f"  Token count: {len(tokens)}")
            logger.info(f"  With special tokens: {len(encoded['input_ids'])}")

        except Exception as e:
            logger.error(f"  Error: {e}")


def compare_tokenizer_settings():
    """Compare different tokenizer settings and their effects"""
    logger.info("\n=== Comparing Tokenizer Settings ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "Hello, world! How are you doing today?"

    settings = [
        {"add_special_tokens": True, "name": "With special tokens"},
        {"add_special_tokens": False, "name": "Without special tokens"},
        {"max_length": 8, "truncation": True, "name": "Truncated to 8"},
        {"return_token_type_ids": True, "name": "With token type IDs"},
        {
            "return_attention_mask": True,
            "padding": "max_length",
            "max_length": 15,
            "name": "Padded to 15",
        },
    ]

    for setting_dict in settings:
        name = setting_dict.pop("name")
        logger.info(f"\n{name}:")

        encoded = tokenizer(text, **setting_dict)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

        logger.info(f"  Tokens: {tokens}")
        logger.info(f"  Input IDs: {encoded['input_ids']}")

        if "token_type_ids" in encoded:
            logger.info(f"  Token type IDs: {encoded['token_type_ids']}")
        if "attention_mask" in encoded:
            logger.info(f"  Attention mask: {encoded['attention_mask']}")


def run_all_examples():
    """Run all tokenization debugging examples"""
    logger.info("Running Tokenization Debugging Examples from Chapter 5")

    # Visualize with offsets
    visualize_tokenization_with_offsets()

    # Detect mismatches
    detect_tokenizer_model_mismatch()

    # Debug special tokens
    debug_special_tokens()

    # Analyze unknown tokens
    analyze_unknown_tokens()

    # Debug padding/truncation
    debug_padding_and_truncation()

    # Edge cases
    debug_tokenization_edge_cases()

    # Compare settings
    compare_tokenizer_settings()

    logger.info("\n✅ All tokenization debugging examples completed!")


if __name__ == "__main__":
    run_all_examples()
