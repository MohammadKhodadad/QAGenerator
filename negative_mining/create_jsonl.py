#!/usr/bin/env python3
"""
create_jsonl.py

Loads the BASF-AI/ChemRxiv-Train-CC-BY dataset, embeds queries and paragraphs
with a HuggingFace model, and writes a JSONL (gzipped) file where each record
has:
  {
    "question": <query>,
    "positive_ctxs": <paragraph>,
    "hard_negative_ctxs": [<negative1>, …]
  }

Negatives are sampled by:
 1. Computing cosine similarities between each query and *all* paragraphs.
 2. Keeping only the top `rate` fraction by similarity (e.g. rate=0.5 → top 50%).
 3. Randomly sampling `num_negatives` documents from that subset.

Usage:
    python create_jsonl.py --output out.jsonl.gz --split train \
        --model bert-base-uncased --rate 0.5 --num-negatives 5
"""
import gzip
import json
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def mean_pooling(model_output, attention_mask):
    tokens = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(tokens.size()).float()
    return torch.sum(tokens * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def embed_texts(texts, model, tokenizer, device, batch_size=32):
    """
    Embed a list of strings with mean pooling and L2-normalization.
    Returns a (N, D) torch.FloatTensor on CPU.
    """
    embeddings = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = model(**enc)
            pooled = mean_pooling(out, enc["attention_mask"])
            normed = F.normalize(pooled, p=2, dim=1)
        embeddings.append(normed.cpu())

    return torch.cat(embeddings, dim=0)


def prepare_jsonl_with_negatives(
    output_path: str,
    split: str = "train",
    model_name: str = "bert-base-uncased",
    rate: float = 0.5,
    num_negatives: int = 5,
    batch_size: int = 32,
):
    # 1) Load dataset
    ds = load_dataset("BASF-AI/ChemRxiv-Train-CC-BY", split=split)
    questions = [str(ex.get("generated_query", "")) for ex in ds]
    paragraphs = [str(ex.get("paragraph", "")) for ex in ds]

    # 2) Load model & tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # 3) Embed all queries & paragraphs
    print("⏳ Embedding queries...")
    q_emb = embed_texts(questions, model, tokenizer, device, batch_size)
    print("⏳ Embedding paragraphs...")
    p_emb = embed_texts(paragraphs, model, tokenizer, device, batch_size)

    # 4) Compute full cosine similarity matrix (N x N)
    # Because embeddings are L2-normalized, inner product = cosine similarity.
    sim = torch.mm(q_emb, p_emb.t())  # (num_examples, num_examples)
    num_docs = len(paragraphs)
    cutoff = int(num_docs * rate)

    # 5) Write out JSONL with negatives
    out_path = Path(output_path)
    print(f"✍️  Writing records with {num_negatives} negatives each to {out_path}")
    with gzip.open(out_path, "wt") as fout:
        for idx, ex in enumerate(tqdm(ds, desc="Sampling negatives")):
            sims = sim[idx]  # (num_docs,)
            # rank paragraphs by descending similarity
            ranked = torch.argsort(sims, descending=True).tolist()
            # exclude the true positive at the same index
            ranked = [i for i in ranked if i != idx]
            # keep only the top `cutoff` fraction
            pool = ranked[:cutoff]
            # sample negatives randomly from that pool
            chosen = random.sample(pool, min(num_negatives, len(pool)))
            neg_texts = [paragraphs[i] for i in chosen]

            record = {
                "question": questions[idx],
                "positive_ctxs": paragraphs[idx],
                "hard_negative_ctxs": neg_texts,
            }
            fout.write(json.dumps(record) + "\n")

    print("✅ Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="chemrxiv_train.jsonl.gz")
    parser.add_argument("--split", "-s", default="train")
    parser.add_argument("--model", "-m", default="bert-base-uncased")
    parser.add_argument("--rate", "-r", type=float, default=0.5,
                        help="Fraction of top-similarity docs to keep before sampling negatives")
    parser.add_argument("--num-negatives", "-n", type=int, default=5)
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    args = parser.parse_args()

    prepare_jsonl_with_negatives(
        output_path=args.output,
        split=args.split,
        model_name=args.model,
        rate=args.rate,
        num_negatives=args.num_negatives,
        batch_size=args.batch_size,
    )
