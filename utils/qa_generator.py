import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import tqdm

def ask_openai(client: OpenAI, prompt: str) -> str:
    """
    Send a prompt to the OpenAI client and return the response content.

    Args:
        client (OpenAI): Initialized OpenAI client.
        prompt (str): The prompt to send.

    Returns:
        str: The assistant's response.
    """
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="o3-mini"
    )
    return response.choices[0].message.content.strip()


def generate_qa(text: str,client:OpenAI, temperature: float = 0.7) -> dict:
    """
    Generate a question-answer pair for a given text piece using the OpenAI client.

    Args:
        text (str): The input text to generate QA from.
        temperature (float): Sampling temperature.

    Returns:
        dict: A dictionary with keys 'question' and 'answer', or raw content if parsing fails.
    """
    prompt = (
        "You are provided with a text in chemistry"
        "Your task is to generate a single, focused question and answer pair about the text."
        "The question should imitate a real questions for RAG" 
        "The answer to your question should be explicitly found in the description."
        "Make sure you do not point to a text in, such as 'based on text', or 'according to the description'"
        "Return a JSON object with keys 'question' and 'answer'.\n\n"
        f"Text:\n{text}\n"
    )
    content = ask_openai(client, prompt)
    try:
        qa = json.loads(content)
        qa['context']= text
        qa['error']=None
    except json.JSONDecodeError:
        qa = {"context": text,'error': "json decode error"}
    return qa


def generate_bulk_qa(
    texts,
    client,
    output_file: str = os.path.join("data", "qa_data.json"),
    max_workers: int = None,
    temperature: float = 0.7
) -> list:
    """
    Generate QA pairs for multiple texts in parallel and save to a JSON file.

    Args:
        texts (List[str]): A list of text pieces to generate QA for.
        output_file (str): Path to save the resulting JSON data.
        max_workers (int): Number of parallel workers. Defaults to None (max threads).
        temperature (float): Sampling temperature.

    Returns:
        list: A list of QA dictionaries.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_qa, text, client, temperature): text for text in texts}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Generating QA"):  
            text = futures[future]
            try:
                qa = future.result()
                results.append(qa)
            except Exception as e:
                results.append({"error": str(e), "text": text})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    import dotenv
    # Load environment variables from .env file
    dotenv.load_dotenv()

    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    client = OpenAI(api_key=api_key)

    # Example usage
    sample_texts = [
        "In this section, we put forth a set of potential characteristics for an ideal fast ion conductor based on our comprehensive understanding of various material classes and the underlying theoretical advancements in fast ion conduction. In order to explore these features, we have classified the materials from both the cationic and anionic perspectives, recognizing the pivotal role played by the immobile framework in facilitating mobile ion conduction. Firstly, it is generally observed that stable lattice sites exhibit higher coordination environments. Consequently, these sites foster stronger interactions, resulting in deeper energy minima, as depicted in Fig. . However, it is important to note that such deeper energy minima can also give rise to larger migration energy barriers, impeding cationic diffusion. Conversely, if we design a crystal structure wherein the coordination number along the ion migration path remains relatively constant, we can achieve a flatter energy landscape that facilitates faster ionic diffusion. Secondly, the presence of a three-dimensional ion migration path is crucial for achieving fast ion conduction. This architectural arrangement minimizes the probability of ion transport blockages due to the presence of grain boundaries. Thirdly, it has been established that the concentration of mobile ions directly affects ion conduction, as evident in the conduction equation. However, it is worth highlighting that both mobile ion concentration and vacancies play significant roles in this context. Therefore, the optimal concentration of mobile ions can provide the ideal vacancy concentration, ultimately leading to enhanced ionic conductivity. Fourthly, ion-ion correlation can also significantly reduce the activation energy barrier. Another important factor is anion rotation, which can introduce structural frustration, enabling a flatter energy landscape for faster cationic diffusion.",
        "First synthesized in 1963 by van Tamelen, Dewar benzene consists of two fused strained cyclobutenes. Suitably substituted derivatives are promising as energy storage materials due to the reversibility of the Dewar benzene formation. For example, hexafluorobenzene is selectively and in high yield photoisomerized to its high energy Dewar isomer, whereas the Dewar isomer of hexamethylbenzene is sufficiently stable to release thermal energy only on demand (Figure ). Dewar benzene derivatives have also been utilized in holographic 3D-information storage, taking advantage of quantum-amplification effects of the photoisomerization, and they have been embedded into polymers to achieve new reconfigurable materials that undergo main-chain structural transformations via valence isomerization. The materials discussed above consist solely of carbon atoms in their backbone, and heteroaromatic analogs of Dewar benzene remain exceedingly rare. Heteroarenes with B-N units embedded in the aromatic framework provoke ever increasing interest due to the extensive applications that are emerging in biochemistry and pharmacology, materials science, and catalysis. The isosteric replacement of C=C for B-N units in benzene to furnish 1,2-azaborinines has proved to be a particularly effective approach. Importantly, azaborinines exhibit significant differences in the aromatic delocalization from benzene. As a consequence, they present distinct reactivity at different ring positions that allows for selective functionalization. In a remarkable recent development, Bettinger and Liu discovered that 1,2-dihydro-1-tert-butyldimethylsilyl-2-mesityl-1,2-azaborinine (A) undergoes photoconversion into the corresponding Dewar valence isomer (B) upon irradiation with UV light (> 280 nm) (Figure ). The kinetically stable isomer B can be converted back to A by a thermal electrocyclic ring-opening reaction that requires an activation energy of (27.0 ± 1.2) kcal mol -1 (half-life of 25 min at 100 ºC). In the presence of Wilkinson's catalyst, the ring-opening occurs rapidly and exothermically even at room temperature. Pursuing new synthetic pathways that take advantage of the facile formation of highly functional BN-Dewar benzene derivatives, Liu and coworkers also developed a strategy to 1,2-substituted cyclobutane derivatives via hydrogenation and subsequent ring-opening of the 4-membered B-N heterocycle. Inspired by these results, we hypothesized that the presence of a strained cyclobutene ring system in BN Dewar"
    ]
    print("Generating QA pairs...")
    qa_list = generate_bulk_qa(sample_texts,client)
    print(f"Saved {len(qa_list)} QA pairs to {os.path.join('data', 'qa_data.json')}")
