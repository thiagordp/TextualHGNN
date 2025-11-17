import json
import logging
import os
from typing import List, Dict

import torch
from dotenv import load_dotenv
from together import Together

from src.models.graph_explainability.base_oracle import BaseOracle


class LLMOracle(BaseOracle):
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                 system_prompt: str = "", language="english"):
        """
        Initialize the LLMOracle with an API key and the model to use.
        """
        self.client = Together(api_key=api_key)
        self.system_prompt = open(system_prompt, "r").read().strip()
        self.model = model
        self.language = language

    def concept_grounding_from_embeddings(self, embeddings: torch.Tensor, num_terms_to_retrieve: int = 1) -> Dict:
        # Not used here.
        pass

    def concept_grounding_from_words(self, words: List[str], num_terms_to_retrieve: int = 1,
                                     similarity_threshold: float = 0.9999) -> Dict:
        """
        Call the Together AI API with the given system and user prompts, and retrieve the best fitting terms.
        """

        # Hardcoded for now. Change later
        context = {
            "english": "Evaluations of movies",
            "italian": "Decisioni della 'Corte di Cassazione' sulla custodia cautelare.",
            "portuguese_voto": "Decisões de Habeas Corpus contra prisão preventivas do Supremo Tribunal Federal."
        }

        words = ", ".join(words)

        user_prompt = f"""
# Context
{context[self.language]}

# Language
{self.language}

# Terms to retrieve
{num_terms_to_retrieve} 

# Words to analyse
{words}
            """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt.strip()}
        ]

        output = None
        tentative = 0
        json_output = None
        while output is None and tentative < 5:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000
            )

            try:
                output = response.choices[0].message.content.strip().lower()

                json_output = self.extract_json(output)
            except Exception as e:
                logging.info(f"Attempt {tentative + 1}: Error fetching response - {e}")
                tentative += 1
                output = None

        if not json_output:
            logging.info("No valid output received from the model.")
            return {}

        return json_output

    @staticmethod
    def extract_json(output: str) -> Dict:
        """
        Extract and parse a JSON object from a mixed text output.

        :param output: The raw string output from the model.
        :return: A dictionary if JSON is successfully parsed, otherwise an empty dict.
        """
        try:
            # Find the first and last curly braces to isolate the JSON part
            start = output.find('{')
            end = output.rfind('}')

            if start != -1 and end != -1:
                json_str = output[start:end + 1]
                return json.loads(json_str)

        except json.JSONDecodeError as e:
            raise Exception(f"JSON parsing error: {e}")

        return {}


def main():
    # CODE For testing purposes only

    # Load environment variables from the .env file
    load_dotenv()

    PROMPT_LLM_ORACLE_PATH = "data/prompts/llm_oracle.txt"

    # Retrieve the API key from environment variables for security
    api_key = os.getenv("TOGETHER_API_KEY")

    if not api_key:
        logging.info("API Key not found. Please set the TOGETHER_API_KEY environment variable.")
        return

    # Instantiate the LLMOracle class with the API key
    oracle = LLMOracle(
        api_key=api_key,
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        system_prompt=open(PROMPT_LLM_ORACLE_PATH, 'r').read().strip(),
        language="italian"
    )

    words = ['Corte di Appello', 'Napoli', 'ordinanza']

    # Get the best fitting terms
    try:
        terms = oracle.concept_grounding_from_words(words)
        logging.info("Received terms:", terms)
    except Exception as e:
        logging.info("Error occurred while fetching terms:", e)


if __name__ == "__main__":
    main()
