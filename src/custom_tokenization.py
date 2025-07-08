"""
Chapter 5: Custom Tokenization
Training custom tokenizers for domain-specific data
"""

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

from config import DATA_DIR, logger


def train_custom_tokenizer_simple():
    """Train custom tokenizer using train_new_from_iterator.

    (from article5.md lines 731-760)
    """
    logger.info("=== Training Custom Tokenizer (Simple Method) ===")

    # Example domain-specific medical texts
    texts = [
        "Patient exhibits signs of pneumothorax.",
        "CT scan reveals bilateral infiltrates.",
        "Myocardial infarction confirmed via ECG.",
        "Administered 5mg of morphine for pain management.",
        "Post-operative recovery progressing normally.",
        "CBC shows elevated white blood cell count.",
        "MRI indicates herniated disc at L4-L5.",
        "Patient history includes hypertension and diabetes.",
        "Prescribed antibiotics for bacterial infection.",
        "Radiology report shows no acute findings.",
    ]

    # Start with a base tokenizer as template
    base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Train a new tokenizer on domain data
    new_tokenizer = base_tokenizer.train_new_from_iterator(
        texts,
        vocab_size=5000,
    )

    # Save the custom tokenizer
    save_path = DATA_DIR / "custom_medical_tokenizer"
    save_path.mkdir(parents=True, exist_ok=True)
    new_tokenizer.save_pretrained(str(save_path))

    logger.info(f"Custom tokenizer saved to: {save_path}")

    # Test the custom tokenizer
    test_text = "Patient exhibits signs of pneumothorax."
    tokens = new_tokenizer.tokenize(test_text)
    logger.info(f"Custom tokenization: {tokens}")

    return new_tokenizer


def load_and_use_custom_tokenizer():
    """Load and use a custom tokenizer (from article5.md lines 765-785)"""
    logger.info("\n=== Loading and Using Custom Tokenizer ===")

    save_path = DATA_DIR / "custom_medical_tokenizer"

    # Check if custom tokenizer exists
    if not save_path.exists():
        logger.info("Custom tokenizer not found, training one first...")
        train_custom_tokenizer_simple()

    # Load the custom tokenizer
    custom_tokenizer = AutoTokenizer.from_pretrained(str(save_path))

    sample = "Patient exhibits signs of pneumothorax."

    # Tokenize and get alignment info
    tokens = custom_tokenizer.tokenize(sample)
    ids = custom_tokenizer.encode(sample)
    offsets = custom_tokenizer(
        sample, return_offsets_mapping=True, return_tensors=None
    )["offset_mapping"]

    logger.info(f"Custom Tokens: {tokens}")
    logger.info(f"Token IDs: {ids}")
    logger.info(f"Offsets: {offsets}")

    return custom_tokenizer


def train_custom_bpe_tokenizer():
    """Train a custom BPE tokenizer using the tokenizers library"""
    logger.info("\n=== Training Custom BPE Tokenizer (Advanced) ===")

    # Create training corpus
    corpus = [
        "The patient presented with acute myocardial infarction.",
        "Pneumothorax was diagnosed via chest X-ray.",
        "HbA1c levels indicate well-controlled diabetes.",
        "Bilateral pleural effusion observed on CT scan.",
        "Post-operative complications include wound infection.",
        "Electrocardiogram shows atrial fibrillation.",
        "Magnetic resonance imaging of the lumbar spine.",
        "Chronic obstructive pulmonary disease exacerbation.",
        "Intravenous antibiotics administered for sepsis.",
        "Echocardiogram reveals left ventricular hypertrophy.",
    ]

    # Initialize a tokenizer with BPE model
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenization (splitting on whitespace and punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Train the tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=1000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # Add post-processing for BERT-style tokens
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 2),
            ("[SEP]", 3),
        ],
    )

    # Test the tokenizer
    test_text = "Patient with myocardial infarction"
    encoding = tokenizer.encode(test_text)
    logger.info(f"BPE tokens: {encoding.tokens}")
    logger.info(f"BPE IDs: {encoding.ids}")

    return tokenizer


def compare_domain_tokenization():
    """Compare general vs domain-specific tokenization"""
    logger.info("\n=== Comparing General vs Domain-Specific Tokenization ===")

    # Medical terms that might be split differently
    medical_terms = [
        "pneumothorax",
        "myocardial",
        "electrocardiogram",
        "thrombocytopenia",
        "cholecystectomy",
    ]

    # Load general tokenizer
    general_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load or create custom tokenizer
    save_path = DATA_DIR / "custom_medical_tokenizer"
    if save_path.exists():
        custom_tokenizer = AutoTokenizer.from_pretrained(str(save_path))
    else:
        custom_tokenizer = train_custom_tokenizer_simple()

    logger.info("\nComparing tokenization of medical terms:")
    for term in medical_terms:
        general_tokens = general_tokenizer.tokenize(term)
        custom_tokens = custom_tokenizer.tokenize(term)

        logger.info(f"\n'{term}':")
        logger.info(f"  General: {general_tokens} (length: {len(general_tokens)})")
        logger.info(f"  Custom:  {custom_tokens} (length: {len(custom_tokens)})")


def domain_specific_batch_processing():
    """Process domain-specific batch with custom tokenizer"""
    logger.info("\n=== Domain-Specific Batch Processing ===")

    # Load custom tokenizer
    save_path = DATA_DIR / "custom_medical_tokenizer"
    if not save_path.exists():
        train_custom_tokenizer_simple()

    custom_tokenizer = AutoTokenizer.from_pretrained(str(save_path))

    # Medical reports batch
    medical_reports = [
        "Patient presents with acute chest pain and shortness of breath.",
        "ECG shows ST-elevation myocardial infarction.",
        "Recommended immediate cardiac catheterization.",
        "Post-procedure: successful stent placement in LAD.",
    ]

    # Batch tokenize with padding and truncation
    encoded_batch = custom_tokenizer(
        medical_reports,
        padding=True,
        truncation=True,
        max_length=50,
        return_tensors="pt",
    )

    logger.info(f"Batch shape: {encoded_batch['input_ids'].shape}")
    first_tokens = custom_tokenizer.convert_ids_to_tokens(encoded_batch["input_ids"][0])
    logger.info(f"First report tokens: {first_tokens}")

    return encoded_batch


def run_all_examples():
    """Run all custom tokenization examples"""
    logger.info("Running Custom Tokenization Examples from Chapter 5")

    # Train simple custom tokenizer
    train_custom_tokenizer_simple()

    # Load and use custom tokenizer
    load_and_use_custom_tokenizer()

    # Train advanced BPE tokenizer
    train_custom_bpe_tokenizer()

    # Compare tokenization
    compare_domain_tokenization()

    # Domain-specific batch processing
    domain_specific_batch_processing()

    logger.info("\n✅ All custom tokenization examples completed!")


if __name__ == "__main__":
    run_all_examples()
