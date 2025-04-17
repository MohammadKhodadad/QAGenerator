# Running Data Preparation and Negative Sampling

This guide describes how to generate the training dataset and sample negatives using the provided scripts.

## Prerequisites

- Python 3.8+  
- PyTorch  
- `torchrun` (included with PyTorch)  
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Generate JSONL Training Data

```bash
python create_jsonl.py \
  --output data/chemrxiv_train.jsonl.gz \
  --split train
```

This command will create a gzipped JSONL file at `data/chemrxiv_train.jsonl.gz`.

## Generate Negative Samples

```bash
torchrun \
  --nproc_per_node=1 \
  --master_port=12345 \
  get_negatives.py \
  --dataset data/chemrxiv_train.jsonl.gz \
  --output_dir data/outputs/
```

The negative samples will be written to the `data/outputs/` directory.
