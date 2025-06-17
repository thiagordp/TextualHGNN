# data_preprocessing.py

import unicodedata
import re
import contractions
import spacy

# Load spaCy model once for efficiency
nlp = spacy.load('en_core_web_lg')


def preprocess_text(text: str) -> str:
    """
    A comprehensive text cleaning pipeline to normalize, clean, and standardize raw text
    before it is passed to the graph builder.

    Args:
        text (str): The raw input document string.

    Returns:
        str: A fully cleaned and standardized document string.
    """

    def _replace_special_chars(text: str) -> str:
        """Replaces a curated list of symbols and special characters."""
        # Note: Extensive replacement rules can be managed in a separate config file.
        replacements = {
            "<br />": "\n", "\ufeff": "",
            "&": " and ", "%": " percent ", "--": " - ", "²": " squared ",
            "®": " registered ", "™": " trademark ", "°": " degrees ",
            "½": " half ", "¼": " quarter ", "¾": " three quarters ",
            "–": "-", "—": "-", "‘": "'", "’": "'", "“": "\"", "”": "\"",
            "´": "'", "`": "'", "¨": "\"", "…": "...", "€": " euro ",
            "£": " pound ", "$": " dollar "
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove any remaining simple HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        return text

    def _normalize_unicode(text: str) -> str:
        """Handles unicode normalization and removes non-ASCII characters."""
        text = unicodedata.normalize('NFKD', text)
        return text.encode('ascii', 'ignore').decode('utf-8', 'ignore')

    def _expand_contractions(text):
        # Expand contractions using the contractions library
        return contractions.fix(text)

    # --- Preprocessing Pipeline ---
    # 1. Initial character and symbol replacement
    clean_text = _replace_special_chars(text)

    # 2. Normalize Unicode to get closer to a standard ASCII set
    clean_text = _normalize_unicode(clean_text)

    # 3. Expand contractions (e.g., "don't" -> "do not")
    clean_text = _expand_contractions(clean_text)

    # 4. Use spaCy for robust token-level processing and rejoining.
    # This is more robust than complex regex for spacing.
    doc = nlp(clean_text)
    processed_tokens = []
    for token in doc:
        # Preserve "I" and proper nouns, lowercase the rest
        if token.text == "I" or token.pos_ == "PROPN":
            processed_tokens.append(token.text_with_ws)
        else:
            processed_tokens.append(token.text_with_ws.lower())

    # 5. Rejoin into a single string and clean up whitespace
    final_text = "".join(processed_tokens)
    final_text = re.sub(r'\s+', ' ', final_text).strip()

    return final_text