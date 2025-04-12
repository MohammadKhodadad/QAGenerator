import os
from openai import OpenAI
from utils.question_generator import generate_bulk_qa
from utils.chemrxiv import load_chemrxiv_texts


def main():


    texts = load_chemrxiv_texts(config="cc-by-nc", split="train", cache_dir="data")
    print(f"Loaded {len(texts)} texts from ChemRxiv-Paragraphs 'cc-by-nc' split 'train'.")


    output_file = os.path.join("data","chemrxiv_questions_v1.json")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    client = OpenAI(api_key=api_key)
    qa_list = generate_bulk_qa(
        texts=texts[:10],
        client=client,
        output_file=output_file,
        low_difficulty_prob=1.0,
    )
    print(f"Saved {len(qa_list)} QA pairs to {output_file}.")


if __name__ == "__main__":
    main()
