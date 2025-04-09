import os
import json
import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm


def load_perplexity_model(model_name: str = "gpt2"):
    """
    Load the language model and tokenizer for perplexity computation.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model


def compute_perplexity(text: str, tokenizer, model) -> float:
    """
    Compute the perplexity of a single text string.
    """
    # Tokenize input (truncate to model max length)
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    input_ids = encodings.input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    # Compute negative log-likelihood
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss.item() * input_ids.size(1)

    # Perplexity = exp(average negative log-likelihood per token)
    return math.exp(neg_log_likelihood / input_ids.size(1))



def select_top_texts_by_perplexity(
    texts,
    top_k: int = 5000,
    model_name: str = "gpt2",
    output_file: str = os.path.join("data", "top_texts.json")
) -> list:
    """
    Select the top_k text pieces by highest perplexity and save them.

    Args:
        texts (List[str]): List of text strings to evaluate.
        top_k (int): Number of top items to select.
        model_name (str): Pretrained model name for perplexity.
        output_file (str): Path to save the selected texts as JSON list.

    Returns:
        List[str]: The top_k texts sorted by descending perplexity.
    """
    # Prepare output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load model
    tokenizer, model = load_perplexity_model(model_name)

    # Compute perplexities
    perp_pairs = []
    for text in tqdm(texts, desc="Computing perplexity"):
        try:
            ppl = compute_perplexity(text, tokenizer, model)
        except Exception:
            ppl = float("inf")
        perp_pairs.append((text, ppl))

    # Sort by perplexity descending
    perp_pairs.sort(key=lambda x: x[1], reverse=True)

    # Select top_k texts
    top_texts = [text for text, _ in perp_pairs[:top_k]]

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(top_texts, f, ensure_ascii=False, indent=2)

    return top_texts
