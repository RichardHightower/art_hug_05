"""
Chapter 5: Tokenization Examples
Basic tokenization examples from article5.md
"""

from transformers import AutoTokenizer

from config import logger


def basic_tokenization_example():
    """Basic tokenization example with BERT (from article5.md lines 149-166)"""
    logger.info("=== Basic Tokenization Example ===")

    # Load a pre-trained fast tokenizer (BERT)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = "Transformers are revolutionizing AI!"

    # Tokenize and prepare model inputs in one step
    encoded = tokenizer(text)
    logger.info(f'Input IDs: {encoded["input_ids"]}')
    logger.info(f'Tokens: {tokenizer.convert_ids_to_tokens(encoded["input_ids"])}')

    # For direct tensor output (e.g., for PyTorch models):
    tensor_inputs = tokenizer(text, return_tensors="pt")
    logger.info(f'Tensor Input IDs: {tensor_inputs["input_ids"]}')

    return encoded, tensor_inputs


def multilingual_tokenization_example():
    """Multilingual tokenization with emojis (from article5.md lines 250-268)"""
    logger.info("\n=== Multilingual Tokenization Example ===")

    # Always use the tokenizer that matches your model version
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    text = "Transformers están revolucionando la IA! 🚀"

    # Tokenize and map to IDs in one step (recommended)
    encoded = tokenizer(text, return_tensors="pt")
    logger.info(f'Input IDs: {encoded["input_ids"]}')
    logger.info(f'Tokens: {tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])}')

    # Inspect special tokens
    logger.info(f"Special tokens: {tokenizer.special_tokens_map}")

    return encoded


def batch_tokenization_with_alignment():
    """Batch tokenization with padding and alignment (from article5.md lines 666-689)"""
    logger.info("\n=== Batch Tokenization with Alignment ===")

    # Load the BERT tokenizer (downloads vocab and config)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    sentences = ["Tokenization is fun!", "Let's build smarter models."]

    # Tokenize the batch, including alignment info
    encoded = tokenizer(
        sentences,
        padding=True,  # Pad to the longest sentence
        truncation=True,  # Truncate if too long
        return_tensors="pt",  # PyTorch tensors ('tf' for TensorFlow)
        return_offsets_mapping=True,  # Get character-to-token alignment
    )

    logger.info(f'Input IDs: {encoded["input_ids"]}')
    logger.info(f'Attention Mask: {encoded["attention_mask"]}')
    logger.info(f'Offsets: {encoded["offset_mapping"]}')

    # Show tokens for each sentence
    for i, _ in enumerate(sentences):
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][i])
        logger.info(f"Sentence {i+1} tokens: {tokens}")

    return encoded


def tensorflow_output_example():
    """TensorFlow tensor output example (from article5.md lines 707-717)"""
    logger.info("\n=== TensorFlow Output Example ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    sentences = ["Tokenization is fun!", "Let's build smarter models."]

    # For TensorFlow tensors, use:
    encoded_tf = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="tf",
        return_offsets_mapping=True,
    )
    logger.info(f'TF Input IDs: {encoded_tf["input_ids"]}')

    return encoded_tf


def special_token_handling():
    """Special token inspection and handling (from article5.md lines 969-984)"""
    logger.info("\n=== Special Token Handling ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Inspect current special tokens
    logger.info(f"Special tokens: {tokenizer.special_tokens_map}")

    # Add custom special tokens if needed
    special_tokens_dict = {"additional_special_tokens": ["<CUSTOM>"]}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added} special tokens.")

    # Visualize tokenization with special tokens
    text = "Classify this sentence."
    encoded = tokenizer(text)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    logger.info(f"Tokens with Special Tokens: {tokens}")
    # Example output: ['[CLS]', 'classify', 'this', 'sentence', '.', '[SEP]']

    return tokenizer, encoded


def production_batch_example():
    """Production-ready batch tokenization (from article5.md lines 931-951)"""
    logger.info("\n=== Production Batch Example ===")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    sentences = ["Transformers are powerful.", "Tokenization makes them work!"]

    # Batch tokenize, pad, and truncate
    encoded_batch = tokenizer(
        sentences,
        padding=True,  # Pad to same length
        truncation=True,  # Truncate if too long
        return_tensors="pt",  # PyTorch tensors
    )

    # For production or API use: convert tensors to lists for JSON serialization
    input_ids_list = encoded_batch["input_ids"].cpu().tolist()
    attention_mask_list = encoded_batch["attention_mask"].cpu().tolist()

    logger.info(f"Batch Input IDs (as lists): {input_ids_list}")
    logger.info(f"Batch Attention Masks: {attention_mask_list}")

    return input_ids_list, attention_mask_list


def run_all_examples():
    """Run all tokenization examples"""
    logger.info("Running Tokenization Examples from Chapter 5")

    # Basic tokenization
    basic_tokenization_example()

    # Multilingual with emojis
    multilingual_tokenization_example()

    # Batch processing with alignment
    batch_tokenization_with_alignment()

    # TensorFlow output
    try:
        tensorflow_output_example()
    except ImportError:
        logger.warning("TensorFlow not installed, skipping TF example")

    # Special token handling
    special_token_handling()

    # Production batch processing
    production_batch_example()

    logger.info("\n✅ All tokenization examples completed!")


if __name__ == "__main__":
    run_all_examples()
