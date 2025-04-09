import os
import json
import dotenv
from openai import OpenAI
from utils.qa_generator import generate_bulk_qa
from utils.chemrxiv import load_chemrxiv_texts
from utils.perplexity import select_top_texts_by_perplexity


def main():
    TOPK=500
    METHOD="model"
    # Phase 1: Load the 'cc-by-nc' train split of ChemRxiv-Paragraphs

    texts = load_chemrxiv_texts(config="cc-by-nc", split="train", cache_dir="data")
    print(f"Loaded {len(texts)} texts from ChemRxiv-Paragraphs 'cc-by-nc' split 'train'.")

    # Phase 1.5: Select top TOPK texts by perplexity

    top_texts_file = os.path.join("created_data", f"top{TOPK}_texts_{METHOD}.json")
    print("Selecting top TOPK most informative texts by perplexity...")
    top_texts = select_top_texts_by_perplexity(
        texts=texts,
        method=METHOD,
        top_k=TOPK,
        model_name="gpt2",
        output_file=top_texts_file
    )
    print(f"Selected {len(top_texts)} texts and saved to {top_texts_file}.")
    
    # Load already saved file
    # with open(top_texts_file, "r", encoding="utf-8") as f:
    #     top_texts = json.load(f)



    # Phase 2: Generate QA pairs for the top texts and save to JSON
    output_file = os.path.join("created_data", f"chemrxiv_qa_top{TOPK}_{METHOD}.json")
    print(f"Generating QA pairs for {len(top_texts)} texts...")
    dotenv.load_dotenv()

    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    client = OpenAI(api_key=api_key)
    qa_list = generate_bulk_qa(
        texts=top_texts,
        client=client,
        output_file=output_file,
        max_workers=None,
        temperature=0.7
    )
    print(f"Saved {len(qa_list)} QA pairs to {output_file}.")


if __name__ == "__main__":
    main()
