# Working with Pre-trained Models

This project contains working examples for Chapter 05 of the Hugging Face Transformers book.

## Overview

Learn how to implement and understand:

## Prerequisites

- Python 3.12 (managed via pyenv)
- Poetry for dependency management
- Go Task for build automation
- API keys for any required services (see .env.example)

## Setup

1. Clone this repository
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
│   ├── model_loading.py        # Model Loading implementation
│   ├── fine_tuning.py        # Fine Tuning implementation
│   ├── model_comparison.py        # Model Comparison implementation
│   ├── inference_optimization.py        # Inference Optimization implementation
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
task run-model-loading    # Run model loading
task run-fine-tuning    # Run fine tuning
task run-model-comparison    # Run model comparison
```

## Available Tasks

- `task setup` - Set up Python environment and install dependencies
- `task run` - Run all examples
- `task test` - Run unit tests
- `task format` - Format code with Black and Ruff
- `task clean` - Clean up generated files

## Learn More

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Library](https://github.com/huggingface/transformers)
- [Book Resources](https://example.com/book-resources)
