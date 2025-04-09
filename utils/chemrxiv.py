from datasets import load_dataset


def load_chemrxiv_texts(
    config: str = "cc-by-nc",
    split: str = "train",
    cache_dir: str = "data"
) -> list:
    """
    Load and return text paragraphs from the ChemRxiv-Paragraphs dataset.

    Args:
        config (str): Dataset config name ("cc-by" or "cc-by-nc").
        split (str): Split name ("train", "validation", or "test").
        cache_dir (str): Directory to cache the dataset.

    Returns:
        List[str]: A list of paragraph strings.
    """
    ds = load_dataset(
        "BASF-AI/ChemRxiv-Paragraphs",
        config,
        split=split,
        cache_dir=cache_dir
    )
    # Assume the first column is the text
    return ds['paragraph']



if __name__ == "__main__":
    texts = load_chemrxiv_texts()
    print(f"Loaded {len(texts)} paragraphs.")
