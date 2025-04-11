import json
import os
import random
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
    temperature: float = 0.7,
    low_difficulty_prob: float = 0.5
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
    low_difficulty_prompt = (
            "You are given a paragraph extracted from a chemistry research paper. Your task is to create one clear, focused question that targets a single chemistry aspect mentioned in the text."
            "The goal is to imitate information retrival task, meaning that you are provided with a text and you have to generate a question (query) for the text. Therefore the answer of your question should be explicitly found in the text. "
            "Do not include phrases such as 'based on the text' or 'according to the description'. "
            "Example 1: "
            "Text: Fluorescent dyes with extended conjugation typically exhibit shifted absorption and emission wavelengths compared to their less-conjugated counterparts. This extended π-electron network allows for a more efficient delocalization of electrons, which generally results in lower energy transitions and red-shifted spectra."
            "Correct Question: How does extended conjugation in a fluorescent dye influence its absorption and emission wavelengths?"

            "Keep your questions simple. "
            "As an example: " 
            "Text: Once the system is in the diffusive region, the MSD graph follows a straight-line behavior, and the diffusion coefficient, D r , can be calculated from the slope of the straight line against the time difference. It is worth noting that if there is a correlated diffusion, the self-diffusion differs from the collective diffusion. The self-and collective diffusion coefficients are succinctly linked by Haven's ratio (H R ), given by, H R = D c /D r ."
            "Correct Question: What connects self-diffusion and collective diffusion coefficients in a system?"
            "However, there are some cases that should be avoided. "
            "Do not generate questions that rely on vague references."
            "For example:"
            "Text: As is widely known in the community and seen in our previous studies, B3LYP-GD3BJ with small Pople-style basis sets and implicit solvation with CPCM systematically underestimates the free energies of activation of enzyme mechanisms compared to the experimental kinetic value. A focus on the quality of the quantum chemical level of theory is purposefully avoided in this work, to instead efficiently provide insight about QM-cluster model building approaches."
            "Wrong Question: Why does the work intentionally avoid focusing on the quality of the quantum chemical level of theory?"
            "In this question 'the work' term, make the question unclear without the original text. The question should be meaningful independently while still having an answer that is fully contained within the paragraph."
            "This kind of questions should be avoided."
            "Since the questions should mimic the real world scnearios, the question should have meaning without the text. However, its answer should be inside the text. Do not generate genral questions about the paper."
            "Also, since the paragraphs are extracted from papers and they are not preprocessed, the paragraph lacks any chemistry-relevant content."
            "In such scnearios, pass an empty string for the question. "
            "Now its your time to answer. think step by step"
            "When finished, write final_output then return a JSON object with 'question' as key.\n\n"
            f"Text:\n{text}\n"
        )
    high_difficulty_prompt = (
            "You are given a paragraph extracted from a chemistry research paper. Your task is to create one clear, focused question that targets a single chemistry aspect mentioned in the text."
            "The goal is to imitate information retrival task, meaning that you are provided with a text and you have to generate a question (query) for the text. Therefore the answer of your question should be explicitly found in the text. "
            "Do not include phrases such as 'based on the text' or 'according to the description'. "
            "Example 1: "
            "Text: Fluorescent dyes with extended conjugation typically exhibit shifted absorption and emission wavelengths compared to their less-conjugated counterparts. This extended π-electron network allows for a more efficient delocalization of electrons, which generally results in lower energy transitions and red-shifted spectra."
            "Correct Question: How does extended conjugation in a fluorescent dye influence its absorption and emission wavelengths?"

            "Keep your questions complex and specific to the text."
            "As an example: "
            "Text: Once the system is in the diffusive region, the MSD graph follows a straight-line behavior, and the diffusion coefficient, D r , can be calculated from the slope of the straight line against the time difference. It is worth noting that if there is a correlated diffusion, the self-diffusion differs from the collective diffusion. The self-and collective diffusion coefficients are succinctly linked by Haven's ratio (H R ), given by, H R = D c /D r ." 
            "Correct Question: What underlying mechanism causes correlated diffusion to produce different self- and collective diffusion coefficients as reflected by Haven’s ratio?"
            "However, there are some cases that should be avoided. "
            "Do not generate questions that rely on vague references."
            "For example:"
            "Text: As is widely known in the community and seen in our previous studies, B3LYP-GD3BJ with small Pople-style basis sets and implicit solvation with CPCM systematically underestimates the free energies of activation of enzyme mechanisms compared to the experimental kinetic value. A focus on the quality of the quantum chemical level of theory is purposefully avoided in this work, to instead efficiently provide insight about QM-cluster model building approaches."
            "Wrong Question: Why does the work intentionally avoid focusing on the quality of the quantum chemical level of theory?"
            "In this question 'the work' term, make the question unclear without the original text. The question should be meaningful independently while still having an answer that is fully contained within the paragraph."
            "This kind of questions should be avoided."
            "Since the questions should mimic the real world scnearios, the question should have meaning without the text. However, its answer should be inside the text. Do not generate genral questions about the paper."
            "Also, since the paragraphs are extracted from papers and they are not preprocessed, the paragraph lacks any chemistry-relevant content."
            "In such scnearios, pass an empty string for the question. "
            "Now its your time to answer. think step by step"
            "When finished, write final_output then return a JSON object with 'question' as key.\n\n"
            f"Text:\n{text}\n"
        )
    prompt_candidates = [low_difficulty_prompt, high_difficulty_prompt]
    weights = [low_difficulty_prob, 1 - low_difficulty_prob]

    
    prompts = []
    for text in texts:
        prompt = random.choice(prompt_candidates, weights=weights)[0]

        prompts.append(prompt)

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
