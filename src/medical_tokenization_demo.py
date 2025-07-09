"""
Medical Tokenization Demo
Standalone script to run medical tokenization examples
"""

from transformers import AutoTokenizer, AutoModel
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_medical_tokenization():
    """Shows advantage of domain-specific tokenization."""
    # Generic tokenizer
    generic = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Medical terms that generic tokenizers fragment
    medical_terms = [
        "pneumonoultramicroscopicsilicovolcanoconiosis",
        "electroencephalography",
        "thrombocytopenia",
        "gastroesophageal"
    ]

    logger.info("\n=== Generic vs Domain Tokenization ===")

    for term in medical_terms:
        generic_tokens = generic.tokenize(term)

        logger.info(f"\n'{term}':")
        logger.info(f"  Generic: {generic_tokens} ({len(generic_tokens)} tokens)")
        # Custom tokenizer would show fewer tokens

        # Calculate efficiency loss
        if len(generic_tokens) > 3:
            logger.warning(f"  ⚠️ Excessive fragmentation: {len(generic_tokens)} pieces")


def medcpt_encoder_example():
    """Demonstrates MedCPT encoder for biomedical text embeddings."""
    logger.info("\n=== MedCPT Biomedical Text Encoder Example ===")
    
    try:
        # Load MedCPT Article Encoder
        logger.info("Loading MedCPT Article Encoder...")
        model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        
        # Example medical articles
        articles = [
            [
                "Diagnosis and Management of Central Diabetes Insipidus in Adults",
                "Central diabetes insipidus (CDI) is a clinical syndrome which results from loss or impaired function of vasopressinergic neurons in the hypothalamus/posterior pituitary, resulting in impaired synthesis and/or secretion of arginine vasopressin (AVP).",
            ],
            [
                "Adipsic diabetes insipidus",
                "Adipsic diabetes insipidus (ADI) is a rare but devastating disorder of water balance with significant associated morbidity and mortality. Most patients develop the disease as a result of hypothalamic destruction from a variety of underlying etiologies.",
            ],
            [
                "Nephrogenic diabetes insipidus: a comprehensive overview",
                "Nephrogenic diabetes insipidus (NDI) is characterized by the inability to concentrate urine that results in polyuria and polydipsia, despite having normal or elevated plasma concentrations of arginine vasopressin (AVP).",
            ],
        ]
        
        # Format articles for the model
        formatted_articles = [f"{title}. {abstract}" for title, abstract in articles]
        
        with torch.no_grad():
            # Tokenize the articles
            encoded = tokenizer(
                formatted_articles, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=512,
            )
            
            # Encode the articles
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            
            logger.info(f"\nEmbedding shape: {embeds.shape}")
            logger.info(f"Embedding dimension: {embeds.shape[1]}")
            
            # Show tokenization comparison for medical terms
            logger.info("\n=== MedCPT Tokenization of Medical Terms ===")
            
            medical_terms = [
                "diabetes insipidus",
                "vasopressinergic neurons",
                "hypothalamic destruction",
                "polyuria and polydipsia"
            ]
            
            for term in medical_terms:
                tokens = tokenizer.tokenize(term)
                logger.info(f"\n'{term}':")
                logger.info(f"  Tokens: {tokens} ({len(tokens)} tokens)")
            
            # Compare with generic BERT tokenizer
            generic = AutoTokenizer.from_pretrained('bert-base-uncased')
            logger.info("\n=== Comparison with Generic BERT ===")
            
            for term in medical_terms:
                medcpt_tokens = tokenizer.tokenize(term)
                generic_tokens = generic.tokenize(term)
                
                logger.info(f"\n'{term}':")
                logger.info(f"  MedCPT: {len(medcpt_tokens)} tokens")
                logger.info(f"  Generic BERT: {len(generic_tokens)} tokens")
                
                if len(generic_tokens) > len(medcpt_tokens):
                    logger.info(f"  ✅ MedCPT is {len(generic_tokens) - len(medcpt_tokens)} tokens more efficient")
                    
    except Exception as e:
        logger.error(f"Error loading MedCPT model: {e}")
        logger.info("Install with: pip install transformers torch")
        logger.info("Note: MedCPT model requires downloading ~440MB")


def main():
    """Run medical tokenization examples."""
    logger.info("🏥 Medical Tokenization Examples")
    logger.info("=" * 50)
    
    # Run generic vs domain comparison
    compare_medical_tokenization()
    
    # Run MedCPT encoder example
    medcpt_encoder_example()
    
    logger.info("\n✅ Medical tokenization examples completed!")


if __name__ == "__main__":
    main()