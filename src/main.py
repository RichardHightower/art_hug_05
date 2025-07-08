"""Main entry point for all examples."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))


from advanced_tokenization import run_all_examples as run_advanced_examples
from custom_tokenization import run_all_examples as run_custom_examples
from model_loading import run_model_loading_examples
from multimodal_tokenization import run_all_examples as run_multimodal_examples
from tokenization_algorithms import run_all_examples as run_algorithm_examples
from tokenization_debugging import run_all_examples as run_debugging_examples
from tokenization_examples import run_all_examples as run_tokenization_examples


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def main():
    """Run all examples."""
    print_section("TOKENIZATION: CHAPTERS 3 & 5 COMPLETE EXAMPLES")
    print("Welcome! This script demonstrates all tokenization concepts.")
    print(
        "Includes examples from Chapter 5 (article5.md) and advanced examples "
        "from Chapter 3.\n"
    )

    # Core tokenization examples
    print_section("1. BASIC TOKENIZATION EXAMPLES")
    run_tokenization_examples()

    print_section("2. TOKENIZATION ALGORITHMS (BPE, WordPiece, Unigram)")
    run_algorithm_examples()

    print_section("3. CUSTOM TOKENIZATION")
    run_custom_examples()

    print_section("4. TOKENIZATION DEBUGGING")
    run_debugging_examples()

    print_section("5. MULTIMODAL TOKENIZATION")
    run_multimodal_examples()

    print_section("6. ADVANCED TOKENIZATION")
    run_advanced_examples()

    # Legacy examples (kept for compatibility)
    print_section("7. MODEL LOADING (Legacy)")
    run_model_loading_examples()

    print_section("CONCLUSION")
    print("These examples demonstrate tokenization concepts from Chapter 5.")
    print("Tokenization is the critical first step in every transformer pipeline!")
    print(
        "Try modifying the code to experiment with different tokenizers and settings."
    )


if __name__ == "__main__":
    main()
