import os
import json
import math
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
from blingfire import text_to_words
from cached_path import cached_path

# URL for Google 1T unigram counts
GOOGLE_1T_CORPUS = (
    "https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/lucas/google-1T-unigram/unigram_freq.csv"
)

class UnigramPerplexityPredictor:
    """
    Predicts the surprisal (negative average log2-probability) of a passage
    based on unigram distribution from a large corpus.
    """
    UNK = "<unk>"

    def __init__(self, word_counts_path: str = GOOGLE_1T_CORPUS):
        local_path = cached_path(word_counts_path)
        with open(local_path, encoding="utf-8") as f:
            word_counts = {
                word: int(count)
                for word, count in (line.strip().split(",", 1) for line in f)
                if count.isnumeric()
            }
        total = sum(word_counts.values())
        log_total = math.log2(total)
        # log2 probability per word
        self.words_logp = {
            w: math.log2(c) - log_total for w, c in word_counts.items()
        }
        # Unknown token probability
        self.words_logp[self.UNK] = math.log2(math.sqrt(len(self.words_logp)) + 1) - log_total

    def log_p(self, word: str) -> float:
        return self.words_logp.get(word.lower(), self.words_logp[self.UNK])

    def predict(self, text: str) -> float:
        """
        Returns negative average log2-probability (surprisal) per token.
        Higher means more surprising (informative).
        """
        tokens = text_to_words(text).split()
        if not tokens:
            return 0.0
        avg_logp = sum(self.log_p(w) for w in tokens) / len(tokens)
        # surprisal = -avg_logp
        return -avg_logp


def load_perplexity_model(model_name: str = "gpt2"):
    """
    Load the language model and tokenizer for GPT2-based perplexity.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model


def compute_perplexity(text: str, tokenizer, model) -> float:
    """
    Compute GPT2-based perplexity of a single text string.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                     max_length=tokenizer.model_max_length)
    input_ids = enc.input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # negative log-likelihood sum
        nll = outputs.loss.item() * input_ids.size(1)
    return math.exp(nll / input_ids.size(1))


def select_top_texts_by_perplexity(
    texts,
    top_k: int = 5000,
    method: str = "model",
    model_name: str = "gpt2",
    unigram_counts_path: str = GOOGLE_1T_CORPUS,
    output_file: str = os.path.join("data", "top_texts.json")
) -> list:
    """
    Select top_k texts by highest score from chosen method.

    Args:
        texts (List[str]): Texts to evaluate.
        top_k (int): Number to select.
        method (str): 'model' for GPT2 perplexity, 'unigram' for unigram surprisal.
        model_name (str): GPT2 model name if method is 'model'.
        output_file (str): Path to save selected texts JSON.
        unigram_counts_path (str): URL or path for unigram counts if 'unigram'.

    Returns:
        List[str]: Top_k texts sorted by descending score.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize scorer
    if method == "model":
        tokenizer, model = load_perplexity_model(model_name)
        scorer = lambda txt: compute_perplexity(txt, tokenizer, model)
    elif method == "unigram":
        upp = UnigramPerplexityPredictor(word_counts_path=unigram_counts_path)
        scorer = lambda txt: upp.predict(txt)
    else:
        raise ValueError(f"Unknown method '{method}'; choose 'model' or 'unigram'.")

    # Score texts
    scored = []
    for txt in tqdm(texts, desc=f"Scoring texts ({method})"):
        try:
            score = scorer(txt)
        except Exception:
            score = float('-inf')
        scored.append((txt, score))

    # Sort descending by score
    scored.sort(key=lambda x: x[1], reverse=True)
    top_texts = [txt for txt, _ in scored[:top_k]]

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(top_texts, f, ensure_ascii=False, indent=2)

    return top_texts
