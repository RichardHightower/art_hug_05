# Tokenization - Converting Text to Numbers for Neural Networks

This project contains working examples for Article 5: Tokenization from the Hugging Face Transformers series.

🔗 GitHub Repository: [https://github.com/RichardHightower/art_hug_05](https://github.com/RichardHightower/art_hug_05)

## Overview

Learn how to implement and understand:
- How tokenization converts text into numerical representations
- Three major tokenization algorithms: BPE, WordPiece, and Unigram
- Implementation using Hugging Face's transformers library
- Handling common edge cases in production systems
- Debugging tokenization issues effectively
- Building custom tokenizers for specialized domains

## Prerequisites

- Python 3.12 (managed via pyenv)
- Poetry for dependency management
- Go Task for build automation
- API keys for any required services (see .env.example)

## Setup

1. Clone this repository:
   ```bash
   git clone git@github.com:RichardHightower/art_hug_05.git
   cd art_hug_05
   ```
2. Run the setup task:
   ```bash
   task setup
   ```
3. Copy `.env.example` to `.env` and configure as needed

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration and utilities
│   ├── main.py                # Entry point with all examples
│   ├── tokenization_examples.py       # Basic tokenization examples
│   ├── tokenization_algorithms.py     # BPE, WordPiece, and Unigram comparison
│   ├── custom_tokenization.py         # Training custom tokenizers
│   ├── tokenization_debugging.py      # Debugging and visualization tools
│   ├── multimodal_tokenization.py     # Image and CLIP tokenization
│   ├── advanced_tokenization.py       # Advanced tokenization techniques
│   ├── model_loading.py               # Model loading examples
│   └── utils.py               # Utility functions
├── tests/
│   └── test_examples.py       # Unit tests
├── .env.example               # Environment template
├── Taskfile.yml               # Task automation
└── pyproject.toml             # Poetry configuration
```

## Running Examples

Run all examples:
```bash
task run
```

Or run individual modules:
```bash
task run-tokenization          # Run basic tokenization examples
task run-algorithms            # Run tokenization algorithms comparison
task run-custom                # Run custom tokenizer training
task run-debugging             # Run tokenization debugging tools
task run-multimodal            # Run multimodal tokenization
task run-advanced              # Run advanced tokenization techniques
task run-medical               # Run medical tokenization comparison
task run-model-loading         # Run model loading examples
```

## Loading Notebooks

To launch Jupyter notebooks:
```bash
task notebook
```

This will start a Jupyter server where you can:
- Create interactive notebooks for experimentation
- Run code cells step by step
- Visualize tokenization results
- Test different tokenizers interactively

## Available Tasks

- `task setup` - Set up Python environment and install dependencies
- `task run` - Run all examples
- `task run-tokenization` - Run basic tokenization examples
- `task run-algorithms` - Run algorithm comparison examples
- `task run-custom` - Run custom tokenizer training
- `task run-debugging` - Run debugging and visualization tools
- `task run-multimodal` - Run multimodal tokenization examples
- `task run-advanced` - Run advanced tokenization techniques
- `task run-medical` - Run medical tokenization comparison (MedCPT)
- `task run-model-loading` - Run model loading examples
- `task notebook` - Launch Jupyter notebook server
- `task test` - Run unit tests
- `task format` - Format code with Black and Ruff
- `task lint` - Run linting checks (Black, Ruff, mypy)
- `task clean` - Clean up generated files

## Learn More

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Library](https://github.com/huggingface/transformers)
- [Book Resources](https://example.com/book-resources)
