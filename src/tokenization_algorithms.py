"""
Chapter 5: Tokenization Algorithms
Comparing BPE, WordPiece, and Unigram tokenization algorithms
"""

from transformers import AutoTokenizer

from config import logger


def bpe_tokenization_example():
    """BPE tokenization with RoBERTa (from article5.md lines 468-478)"""
    logger.info("=== BPE Tokenization (RoBERTa) ===")

    # Load RoBERTa's BPE tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    text = "unhappiness"
    tokens = tokenizer.tokenize(text)
    logger.info(f"BPE Tokens: {tokens}")  # Example output: ['un', 'happi', 'ness']

    # Show token IDs
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    logger.info(f"Token IDs: {token_ids}")

    return tokens, token_ids


def wordpiece_tokenization_example():
    """WordPiece tokenization with BERT (from article5.md lines 502-514)"""
    logger.info("\n=== WordPiece Tokenization (BERT) ===")

    # Load BERT's WordPiece tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "unhappiness"
    tokens = tokenizer.tokenize(text)
    logger.info(
        f"WordPiece Tokens: {tokens}"
    )  # Example output: ['un', '##happi', '##ness']

    # Show token IDs
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    logger.info(f"Token IDs: {token_ids}")

    return tokens, token_ids


def unigram_tokenization_example():
    """Unigram tokenization with XLM-RoBERTa (from article5.md lines 530-545)"""
    logger.info("\n=== Unigram Tokenization (XLM-RoBERTa) ===")

    # Load XLM-RoBERTa's Unigram tokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    text = "unhappiness"
    tokens = tokenizer.tokenize(text)
    logger.info(
        f"Unigram Tokens: {tokens}"
    )  # Example output: ['un', 'happiness'] or ['un', 'happi', 'ness']

    # Show token IDs
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    logger.info(f"Token IDs: {token_ids}")

    return tokens, token_ids


def compare_tokenization_algorithms():
    """Compare all three algorithms on the same text"""
    logger.info("\n=== Comparing Tokenization Algorithms ===")

    test_words = ["unhappiness", "tokenization", "transformers", "preprocessing"]

    # Load all tokenizers
    bpe_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    wordpiece_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    unigram_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    for word in test_words:
        logger.info(f"\nTokenizing: '{word}'")

        # BPE
        bpe_tokens = bpe_tokenizer.tokenize(word)
        logger.info(f"  BPE (RoBERTa): {bpe_tokens}")

        # WordPiece
        wordpiece_tokens = wordpiece_tokenizer.tokenize(word)
        logger.info(f"  WordPiece (BERT): {wordpiece_tokens}")

        # Unigram
        unigram_tokens = unigram_tokenizer.tokenize(word)
        logger.info(f"  Unigram (XLM-RoBERTa): {unigram_tokens}")


def tokenization_on_special_text():
    """Test tokenization on special cases: emojis, numbers, mixed text"""
    logger.info("\n=== Tokenization on Special Text ===")

    special_texts = [
        "I love pizza! 🍕🔥",
        "COVID-19 pandemic",
        "user@example.com",
        "Price: $99.99",
        "C++ programming",
    ]

    tokenizers = {
        "BPE (RoBERTa)": AutoTokenizer.from_pretrained("roberta-base"),
        "WordPiece (BERT)": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "Unigram (XLM-RoBERTa)": AutoTokenizer.from_pretrained("xlm-roberta-base"),
    }

    for text in special_texts:
        logger.info(f"\nTokenizing: '{text}'")
        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.tokenize(text)
            logger.info(f"  {name}: {tokens}")


def algorithm_strengths_weaknesses():
    """Demonstrate strengths and weaknesses of each algorithm"""
    logger.info("\n=== Algorithm Strengths and Weaknesses ===")

    # Technical/domain terms
    technical_terms = ["myocardial", "infarction", "pneumothorax", "cryptocurrency"]

    # Noisy text with typos
    noisy_text = ["helo", "wrold", "teh", "recieve"]

    # Compound words
    compound_words = ["firewall", "smartphone", "bookkeeper", "downstream"]

    tokenizers = {
        "BPE": AutoTokenizer.from_pretrained("roberta-base"),
        "WordPiece": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "Unigram": AutoTokenizer.from_pretrained("xlm-roberta-base"),
    }

    logger.info("\nTechnical Terms:")
    for term in technical_terms:
        logger.info(f"\n  '{term}':")
        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.tokenize(term)
            logger.info(f"    {name}: {tokens} (length: {len(tokens)})")

    logger.info("\nNoisy Text (Typos):")
    for word in noisy_text:
        logger.info(f"\n  '{word}':")
        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.tokenize(word)
            logger.info(f"    {name}: {tokens}")

    logger.info("\nCompound Words:")
    for word in compound_words:
        logger.info(f"\n  '{word}':")
        for name, tokenizer in tokenizers.items():
            tokens = tokenizer.tokenize(word)
            logger.info(f"    {name}: {tokens}")


def run_all_examples():
    """Run all tokenization algorithm examples"""
    logger.info("Running Tokenization Algorithm Examples from Chapter 5")

    # Individual algorithm examples
    bpe_tokenization_example()
    wordpiece_tokenization_example()
    unigram_tokenization_example()

    # Compare algorithms
    compare_tokenization_algorithms()

    # Special text handling
    tokenization_on_special_text()

    # Strengths and weaknesses
    algorithm_strengths_weaknesses()

    logger.info("\n✅ All tokenization algorithm examples completed!")


if __name__ == "__main__":
    run_all_examples()
