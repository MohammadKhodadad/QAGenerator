import json
import os

def ask_openai_batch(client, prompts, temperature: float = 0.7) -> list:
    """
    Send a batch of prompts to the OpenAI client using the batch API
    and return a list of response strings.

    Args:
        client: Initialized OpenAI client.
        prompts (list of str): A list of prompt strings.
        temperature (float): Sampling temperature.

    Returns:
        list of str: A list containing each response's content.
    """
    # Prepare a list of messages from prompts
    # Here we assume that the batch API takes a list of messages where each item 
    # is a dictionary with the role and content. The batch API is assumed to be
    # similar to the normal API call but accepts multiple prompts at once.
    batch_messages = [{"role": "user", "content": prompt} for prompt in prompts]

    # Call the hypothetical batch endpoint
    responses = client.chat.completions.create_batch(
        messages=batch_messages,
        model="o3-mini",
        temperature=temperature
    )

    # Extract and return the content from each response
    return [response.choices[0].message.content.strip() for response in responses]


def generate_qa(text: str, client, temperature: float = 0.7) -> dict:
    """
    Generate a question-answer pair for a given text piece using the batch API.

    Args:
        text (str): The input text from which to generate the QA pair.
        client: Initialized OpenAI client.
        temperature (float): Sampling temperature.

    Returns:
        dict: A dictionary containing keys 'question', 'answer', 'context',
              and an error indicator (if any).
    """
    prompt = (
        "You are provided with a text in chemistry. "
        "Your task is to generate a single, focused question and answer pair about the text. "
        "The question should imitate a real question for retrieval-augmented generation. "
        "The answer must be explicitly found in the description. "
        "Do not include phrases such as 'based on the text' or 'according to the description'. "
        "Return a JSON object with keys 'question' and 'answer'.\n\n"
        f"Text:\n{text}\n"
    )
    response_list = ask_openai_batch(client, [prompt], temperature=temperature)
    content = response_list[0]

    try:
        qa = json.loads(content)
        qa['context'] = text
        qa['error'] = None
    except json.JSONDecodeError:
        qa = {"context": text, "error": "json decode error"}
    return qa


def generate_bulk_qa(
    texts,
    client,
    output_file: str = os.path.join("data", "qa_data.json"),
    temperature: float = 0.7
) -> list:
    """
    Generate QA pairs for multiple texts using a single batch request
    and save the results to a JSON file.

    Args:
        texts (list of str): A list of text pieces to generate QA pairs for.
        client: Initialized OpenAI client.
        output_file (str): File path to store the resulting JSON data.
        temperature (float): Sampling temperature.

    Returns:
        list: A list of dictionaries, each containing the generated QA pair and context.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Build one prompt per text
    prompts = []
    for text in texts:
        prompt = (
            "You are provided with a text in chemistry. "
            "Your task is to generate a single, focused question about the text. "
            "The question should imitate a real question for retrieval-augmented generation. "
            "Do not include phrases such as 'based on the text' or 'according to the description'. "
            "Return a JSON object with 'question' as key.\n\n"
            "AS an example:"
            "Text: In organic chemistry, nucleophilic substitution reactions are vital for transforming molecules and are mainly classified into two mechanisms: SN1 and SN2. The SN1 mechanism is a two-step process where the initial step involves the departure of a leaving group, resulting in the formation of a carbocation intermediate. This intermediate, being planar, can be attacked from either side by a nucleophile, often leading to a racemic mixture. Tertiary alkyl halides, for instance, frequently follow the SN1 pathway because their corresponding carbocations are significantly stabilized by hyperconjugation and inductive effects. In contrast, the SN2 mechanism is a concerted, one-step process in which the nucleophile attacks the electrophilic carbon from the opposite side of the leaving group, leading to an inversion of stereochemistry at the reaction center. This mechanism is favored in primary alkyl halides, where steric hindrance is minimal, allowing easier access for the nucleophile. Understanding these mechanistic differences is essential for predicting reaction rates and stereochemical outcomes in organic synthesis."
            "Question: Explain why tertiary alkyl halides are more likely to undergo an SN1 mechanism rather than an SN2 mechanism. In your answer, discuss the roles of carbocation stability and steric hindrance, and describe how these factors influence the stereochemical outcome of the reaction?"
            f"Text:\n{text}\n"
        )
        prompts.append(prompt)

    # Make one batch call for all texts
    responses = ask_openai_batch(client, prompts, temperature=temperature)

    results = []
    for text, content in zip(texts, responses):
        try:
            qa = json.loads(content)
            qa['context'] = text
            qa['error'] = None
        except json.JSONDecodeError:
            qa = {"context": text, "error": "json decode error"}
        results.append(qa)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results
