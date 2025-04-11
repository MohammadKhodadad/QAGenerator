import tiktoken
from tooldantic import OpenAiResponseFormatBaseModel as BaseModel
from openai import OpenAI
import pandas as pd
import os
from typing import Union
import tqdm
import json
import gzip


class TokenValidation(BaseModel):
    is_valid: bool
    reason: str


class TextValidation(BaseModel):
    is_valid: bool
    reason: str


QUERY_TOKEN = """We are working on adapting large language models for chemistry by enhancing their vocabulary with chemical-specific tokens. A 'token' here is defined as the smallest unit of information that carries meaningful chemical context (for example, 'hydro' from 'hydroxide' or 'acetyl' from 'acetylsalicylic'). For each token extracted from chemical compound names, please evaluate whether it is chemistry-related. Answer in one sentence starting with 'Yes' or 'No,' followed by a brief explanation.
For example, for the token 'acetyl,' you might say: 'Yes, because it represents a common chemical group found in many organic compounds.' Similarly, for the token 'cyclic,' you might say: 'Yes, as it describes a structural feature prevalent in ring compounds.' If the token does not carry chemical-specific meaning, indicate that it is not chemistry-related. 

Token: {token}"""

QUERY_CHEMRXIV_CLEANING = """You are given a text field extracted from a chemistry research paper. Your task is to decide if this text represents a meaningful paragraph from the paper's body rather than metadata, author names, figure/table captions, truncated text, or text containing extraneous elements like URLs or formulas. 
For instance:
- "Heng Liu , Hao Zheng , Zhenhe Jia , Binghui Zhou..." should be flagged as an author list.
- "https://doi.org/10.26434/chemrxiv-2023-bsft2 ORCID: ..." should be flagged for containing URLs.
- "Figure 1. Schematic illustration of the Sabatier principle..." should be flagged as a figure caption.
- In contrast, a paragraph like "The catalytic volcano activity models are the quantified and visualized tools of the Sabatier principle for heterogeneous catalysis..." is a valid body text.

Note: The invalid cases are not limited to the examples provided. If you determine that a text is invalid for any reason, mark it as invalid and provide your specific reasoning.
Please analyze the input text and provide a JSON response with the fields "is_valid" (true or false) and "reason" (a short string indicating the main issue, e.g., 'short', 'url', 'author list', 'figure caption', 'truncated', etc).

Text: {text}
"""


def get_client() -> OpenAI:
    """
    Returns an instance of the OpenAI client.

    Returns:
        OpenAI: OpenAI client instance.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OPENAI_API_KEY found in the environment variables.")
    return OpenAI(api_key=api_key)


def generate_pubchem_queries(data: pd.DataFrame,
                             output_path: str,
                             id_column: str,
                             token_column: str,
                             template: str,
                             system_prompt: str,
                             model: str,
                             response_format: Union[BaseModel, dict],
                             temperature: float = 0.2,
                             shard_size: int = 50_000):
    """
    Generates PubChem queries for the given data and saves them to the output file.

    Args:
        data (pd.DataFrame): DataFrame containing the PubChem data.
        output_path (str): Output file path to save the generated queries.
        id_column (str): Column name for the ID in the DataFrame.
        token_column (str): Column name for the token in the DataFrame.
        template (str): Template for the query.
        system_prompt (str): System prompt for the query.
        response_format (BaseModel or str): Response format model or string.
        temperature (float): Temperature for the model.
        model (str): Model name to use for the query generation.
        shard_size (int): Number of samples to process in each shard.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    in_tokens = 0

    for shard_start in tqdm.tqdm(range(0, len(data), shard_size)):
        dataset_slice = data.iloc[shard_start:shard_start + shard_size]
        shard_num = shard_start // shard_size
        for i, row in dataset_slice.iterrows():
            prompt = template.format(text=row[token_column])
            with open(os.path.join(output_path, f"shard-{shard_num:03d}.jsonl"), 'a') as f:
                task = {
                    "custom_id": row[id_column],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": temperature,
                        "response_format": response_format if isinstance(response_format, dict) else response_format.model_json_schema()
                    }
                }
                if model in ['o3', 'o3-mini', 'o1', 'o1-mini']:
                    task['body'].pop('temperature')
                f.write(json.dumps(task) + "\n")
                in_tokens = in_tokens + count_tokens(prompt) + count_tokens(system_prompt)

    print(f"Total tokens: {in_tokens}")
    print(f"Total shards: {shard_num + 1}")
    return in_tokens


def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in a given text.

    Args:
        text (str): Text to count tokens.

    Returns:
        int: Number of tokens in the text.
    """
    enc = tiktoken.get_encoding("o200k_base")
    tokens = enc.encode(text)
    return len(tokens)


def merge_json_files(file_paths: Union[str, list], output_file=None) -> list:
    """
    Merges multiple json files into a single json file.

    Args:
        file_paths (Union[str, list]): List of file paths to merge or directory path.
        output_file (str): Output file path. If None, returns the merged json as a list.

    Returns:
        list: Merged jsons as a single list.
    """
    if isinstance(file_paths, str):
        file_paths = [os.path.join(file_paths, f)
                      for f in os.listdir(file_paths) if f.endswith('.jsonl')]

    merged_lines = []
    for path in file_paths:
        with open(path, 'r') as infile:
            merged_lines.extend(infile.readlines())

    if output_file:
        with open(output_file, 'w') as outfile:
            outfile.writelines(merged_lines)
    return merged_lines


def query_generator(data_path: str,
                    output_path: str,
                    query: str,
                    model: str,
                    response_format: BaseModel,
                    text_column: str,
                    id_column: str):

    df = pd.read_csv(data_path)

    # system_prompt = """You are a chemical vocabulary expert tasked with evaluating tokens."""
    system_prompt = "You are a chemist tasked with cleaning and validating extracted text fields from scientific papers for further AI model training."

    generate_pubchem_queries(df,
                             output_path,
                             id_column=id_column,
                             token_column=text_column,
                             template=query,
                             system_prompt=system_prompt,
                             model=model,
                             response_format=response_format)


def generate_batch_job(jsonl_path: str) -> str:
    """
    Generates a batch job for the given JSONL file.

    Args:
        jsonl_path (str): Path to the JSONL file.

    Returns:
        str: Batch job ID.
    """
    client = get_client()
    batch_file = client.files.create(
        file=open(jsonl_path, 'rb'),
        purpose='batch'
    )
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    return batch_job.id


def generate_batch_jobs(input_path: str,
                        json_output_path: str):
    """
    Generates batch jobs for the given input path.

    Args:
        input_path (str): Path to the input files.
        json_output_path (str): Path to save the merged JSON files.
    """
    shards = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.jsonl')]
    file_to_batch_id = {}
    for shard in shards:
        batch_id = generate_batch_job(shard)
        file_to_batch_id[shard] = batch_id

    with open(os.path.join(json_output_path), 'w') as f:
        json.dump(file_to_batch_id, f)


def extract_responses(jsonl_list: list, response_format: BaseModel) -> pd.DataFrame:
    """
    Extracts responses from the given list of JSONL files.

    Args:
        jsonl_list (list): List of JSONL files.
        response_format (BaseModel): Response format model.

    Returns:
        pd.DataFrame: Extracted responses as a DataFrame.
    """
    fields = response_format.model_fields.keys()

    responses = []
    for jsonl in jsonl_list:
        loaded = json.loads(jsonl)
        if loaded['error']:
            continue
        custom_id = loaded['custom_id']
        response = loaded['response']['body']['choices'][0]['message']['content']
        response = json.loads(response)
        responses.append({**{field: response[field] for field in fields}, "custom_id": custom_id})

    responses_df = pd.DataFrame(responses)
    return responses_df


def download_outputs(batch_job_ids: str, output_dir: str):
    """
    Downloads the batch outputs to a given directory.

    Args:
        batch_job_ids (str): Path to the JSON file containing the batch job IDs.
        output_dir (str): Path to save the outputs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(batch_job_ids, 'r') as f:
        file_to_batch_id = json.load(f)

    client = get_client()

    for file, batch_id in tqdm.tqdm(file_to_batch_id.items()):
        batch = client.batches.retrieve(batch_id)
        output_file_id = batch.output_file_id
        response_bytes = client.files.content(output_file_id).content

        base_name = os.path.basename(file)
        base_name, ext = os.path.splitext(base_name)
        response_name = f'{base_name}-response{ext}'
        output_file = os.path.join(output_dir, response_name)

        with open(output_file, 'wb') as f:
            f.write(response_bytes)


def df_to_pairs(data: pd.DataFrame,
                output_dir: str,
                query_col: str,
                document_col: str,
                shard_size: int = 100_000):
    """
    Convert a DataFrame to a JSONL file with the format expected by the training script.

    Args:
        data: The DataFrame to convert.
        output_dir: The directory to write the output JSONL files to.
        query_col: The name of the column in the DataFrame containing the queries.
        document_col: The name of the column in the DataFrame containing the documents.
        shard_size: The number of rows to write to each shard. The output will have one JSONL file per shard.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metadata = {
        'objective': {'self': [], 'paired': [['query', 'document']], 'triplet': []}
    }
    for shard_start in tqdm.tqdm(range(0, len(data), shard_size)):
        dataset_slice = data[shard_start:shard_start + shard_size]
        shard_num = shard_start // shard_size
        with gzip.open(os.path.join(output_dir, f"shard-{shard_num:05d}.jsonl.gz"), 'wt') as f:
            for i, row in dataset_slice.iterrows():
                line = {}
                line['query'] = row[query_col]
                line['document'] = row[document_col]
                line['metadata'] = metadata
                f.write(json.dumps(line) + '\n')


def oai_batch_response_to_df(responses_path: str,
                             input_df: pd.DataFrame,
                             id_column: str,
                             output_path) -> pd.DataFrame:
    """
    Converts the OpenAI batch response JSONL file to a DataFrame.

    Args:
        responses_path (str): Path to the responses JSONL file.
    Returns:
        pd.DataFrame: DataFrame containing the responses.
    """
    columns = set()
    responses = []
    merged_jsonl = merge_json_files(responses_path)
    refusals = 0

    for line in merged_jsonl:
        response = json.loads(line)
        custom_id = response['custom_id']
        message = response['response']['body']['choices'][0]['message']
        if message['refusal'] is None:
            content = json.loads(message['content'])
            responses.append({id_column: custom_id,
                              **content})
            if not columns:
                columns = set(content.keys())
        elif message['refusal'] is not None and message['content'] is None:
            refusals += 1
            refusal = json.loads(message['refusal'])
            if not all(key in refusal for key in columns):
                continue
            responses.append({id_column: custom_id,
                              **refusal})
    print(f"Refusals: {refusals}")
    responses_df = pd.DataFrame(responses)
    merged_df = pd.merge(input_df, responses_df, on=id_column)
    merged_df.to_csv(output_path, index=False)
    return merged_df


if __name__ == "__main__":
    data_path = "data/ChemRxiv_Text_Fields.csv"
    requests_path = "chemrxiv/cleaning/requests"
    responses_path = "chemrxiv/cleaning/responses"
    batch_jons_path = "chemrxiv/cleaning/batch_jobs.json"

    # query_generator(data_path,
    #                 requests_path,
    #                 query=QUERY_CHEMRXIV_CLEANING,
    #                 model='gpt-4o-mini',
    #                 response_format=TextValidation,
    #                 text_column='text',
    #                 id_column='uuid')

    # Create batch jobs

    # generate_batch_jobs(requests_path,
    #                     batch_jons_path)

    # Download outputs

    download_outputs(batch_jons_path,
                     responses_path)

    # Convert responses to DataFrame
    oai_batch_response_to_df(
        responses_path,
        pd.read_csv(data_path),
        id_column='uuid',
        output_path='chemrxiv/cleaning/cleaned_text.csv'
    )
