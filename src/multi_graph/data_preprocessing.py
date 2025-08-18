# data_preprocessing.py

import unicodedata
import re
import contractions
import spacy

def preprocess_text(text: str, nlp) -> str:
    """
    A comprehensive text cleaning pipeline to normalize, clean, and standardize raw text
    before it is passed to the graph builder.

    Args:
        text (str): The raw input document string.

    Returns:
        str: A fully cleaned and standardized document string.
    """

    def _replace_special_chars(target: str) -> str:
        """Replaces a curated list of symbols and special characters."""
        # Note: Extensive replacement rules can be managed in a separate config file.
        replacements = {
            "<br />": "\n", "\ufeff": "",
            "&": " and ", "%": " percent ", "--": " - ", "²": " squared ",
            "®": " registered ", "™": " trademark ", "°": " degrees ",
            "½": " half ", "¼": " quarter ", "¾": " three quarters ",
            "–": "-", "—": "-", "‘": "'", "’": "'", "“": "\"", "”": "\"",
            "´": "'", "`": "'", "¨": "\"", "…": "...", "...": "... ", "€": " euro ",
            "£": " pound ", "$": " dollar ", "**": "*"
        }

        for old, new in replacements.items():
            target = target.replace(old, new)

        # Remove any remaining simple HTML tags
        target = re.sub(r'<[^>]+>', '', target)
        return target

    def _normalize_unicode(target: str) -> str:
        """Handles Unicode normalization and removes non-ASCII characters."""
        target = unicodedata.normalize('NFKD', target)
        return target.encode('ascii', 'ignore').decode('utf-8', 'ignore')

    def _expand_contractions(target):
        # Expand contractions using the contractions library
        return contractions.fix(target)

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


def preprocessing_legal_pt(text: str, nlp_spacy):
    """
    Preprocesses legal documents written in Portuguese.

    This function performs a series of cleaning and normalization steps
    tailored to the characteristics of legal texts in Portuguese.

    The pipeline is as follows:
    0.  Isolates the report by truncating after the *last* "É o relatório.".
    1.  Removes common header and footer patterns from the remaining text.
    2.  Replaces special characters, symbols, and URLs with standardized equivalents.
    3.  Standardizes common legal references (e.g., 'Art. 99' -> 'ART_99').
    4.  Tokenizes the text into individual sentences.
    5.  Processes each sentence to:
        - Lowercase words while preserving proper nouns and standardized entities.
        - Join the processed words back into a clean sentence.
    6.  Performs a final cosmetic cleanup of spacing around punctuation.

    Args:
        text: A string containing the Portuguese legal document.

    Returns:
        A list of preprocessed sentence strings.
    """

    def _replace_special_chars_pt(target: str):
        """
           Performs safe, high-confidence character normalization and noise removal.
           This should be run on the raw string BEFORE processing with spaCy.
           """
        # 1. Standardize punctuation for consistency.
        # This helps reduce the vocabulary size for the model.
        target = target.replace('–', '-').replace('—', '-').replace('―', '-')
        target = target.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        target = target.replace('…', '...')

        # 2. Remove non-semantic characters (pure noise).
        target = target.replace('•', '').replace('·', '')
        target = target.replace('\ufeff', '')  # Remove invisible BOM character

        # 3. Standardize URLs to a single token.
        # Note: The specific authentication block is handled separately. This is for general URLs.
        target = re.sub(r'https?://\S+|www\.\S+', '[URL]', target)
        return target

    def _standardize_legal_entities_v2(target: str) -> str:
        """
        Standardizes simple and complex legal entities using programmatic replacement.
        Handles single articles, lists ("e"), and ranges ("a").
        """
        target = target.replace('§', 'paragrafo')

        # Keyword we are looking for (singular and plural)
        keyword = "artigo"
        keywords_plural = "artigos"

        # --- 1. Handler for ranges like "Artigos 3 a 6" ---
        def expand_range(match):
            start_num = int(match.group(1))
            end_num = int(match.group(2))
            # Generate the numbers in the range
            numbers = range(start_num, end_num + 1)
            # Create the replacement string: "ARTIGO_3, ARTIGO_4, ..."
            return ', '.join([f"{keyword.upper()}_{num}" for num in numbers])

        range_pattern = re.compile(
            fr'\b{keywords_plural}\s+(\d+)\s+a\s+(\d+)\b',
            re.IGNORECASE
        )
        target = range_pattern.sub(expand_range, target)

        # --- 2. Handler for lists like "Artigos 4 e 5" or "3, 4 e 5" ---
        def expand_list(match):
            # Extract the full string of numbers and connectors, e.g., "3, 4 e 5"
            list_str = match.group(1)
            # Find all numbers in that string
            numbers = re.findall(r'\d+', list_str)
            # Create the replacement string
            return ', '.join([f"{keyword.upper()}_{num}" for num in numbers])

        list_pattern = re.compile(
            fr'\b{keywords_plural}\s+((?:\d+,\s*)*\d+\s+e\s+\d+)\b',
            re.IGNORECASE
        )
        target = list_pattern.sub(expand_list, target)

        # --- 3. Handler for simple cases like "Artigo 5" ---
        # This is your original pattern, slightly adapted.
        simple_pattern = re.compile(
            fr'\b({keyword})\.?\s*([\d./-]+)\b',
            re.IGNORECASE
        )
        target = simple_pattern.sub(
            lambda m: f"{m.group(1).upper().replace('.', '')}_{m.group(2).replace('.', '')}",
            target
        )

        return target


    def __isolate_report_section(target: str) -> str:
        """
        Isolates the report section by truncating the text after the *last*
        occurrence of the phrase "É o relatório.".

        This method is designed to capture the entire summary of facts while
        reliably excluding the subsequent vote and reasoning sections.

        Args:
            text: The full text of the legal document.

        Returns:
            The isolated report section of the text. If the marker is not found,
            it returns the original text.
        """
        # This regex finds "É o relatório" as a whole phrase, allowing for an
        # optional period at the end.
        report_end_marker = r'\bÉ o relatório\.?\b'

        # re.finditer finds all non-overlapping matches and returns an iterator.
        # We convert it to a list to easily access the last one.
        matches = list(re.finditer(report_end_marker, target, re.IGNORECASE | re.DOTALL))

        if matches:
            # If one or more matches are found, get the last one from the list.
            last_match = matches[-1]

            # Truncate the text at the end position of this last match.
            return target[:last_match.end()]
        else:
            # If the marker is never found, we return the original text.
            # This prevents accidental deletion of the entire document content.
            # You could add a warning here if you want to track such cases.
            # print("Warning: Report end marker not found. Returning original text.")
            return text

    def _join_broken_lines(target: str) -> str:
        """Joins lines that were likely broken during PDF extraction."""
        lines = target.split('\n')
        reconstructed_lines = []
        buffer = ""
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # If the buffer is not empty and the current line looks like a continuation, append it.
            # Heuristic: The previous line (in the buffer) does not end with sentence-terminating punctuation.
            if buffer and not buffer.endswith(('.', '!', '?', ';', ':')):
                buffer += " " + stripped_line
            else:
                # If the buffer has content, it's a complete line/sentence.
                if buffer:
                    reconstructed_lines.append(buffer)
                buffer = stripped_line

        # Add the last buffered line
        if buffer:
            reconstructed_lines.append(buffer)

        return "\n".join(reconstructed_lines)  # Return text with sentences on new lines

    def __remove_authentication_block(target: str) -> str:
        """
        Finds and removes the multi-line document authentication block from STF texts.

        This pattern often looks like:
        http://www.stf.jus.br/portal/autenticacao/autenticarDocumento.asp sob o código
        48E5-EFAF-DDED-A9F6 e senha A040-0268-651A-9F60

        The function is designed to handle this pattern even when it's broken
        across multiple lines.

        Args:
            text: The text containing the pattern.

        Returns:
            The text with the authentication block removed.
        """
        # Regex to find the entire block, allowing for any whitespace (\s+) in between.
        auth_pattern = re.compile(
            r'https?://www\.stf\.jus\.br/portal/autenticacao/autenticarDocumento\.asp'
            r'\s+sob\s+o\s+código\s+[\w-]+\s+e\s+senha\s+[\w-]+',
            flags=re.IGNORECASE
        )

        return auth_pattern.sub('', target)

    def tokenize_and_normalize_spacy(target: str, nlp_spacy) -> str:
        """
        Processes the full text using spaCy to perform sentence segmentation,
        tokenization, and linguistically-aware normalization (lemmatization).

        This function is the core of the spaCy-centric pipeline. It avoids the
        destructive "Doc -> string -> regex" cycle.

        Args:
            text: The cleaned document text.
            nlp_spacy: The loaded spaCy model for Portuguese.

        Returns:
            A list of sentences, where each sentence is a list of processed tokens (lemmas).
            Example: [['o', 'réu', 'ser', 'condenado'], ['a', 'defesa', 'apelar']]
        """
        # 1. Process the ENTIRE document text at once.
        # spaCy handles sentence segmentation much more accurately than regex/NLTK's tokenizer.
        doc = nlp_spacy(target)
        processed_tokens = []
        for token in doc:
            # Preserve "I" and proper nouns, lowercase the rest
            if token.pos_ == "PROPN":
                processed_tokens.append(token.text_with_ws)
            else:
                processed_tokens.append(token.text_with_ws.lower())

        # 5. Rejoin into a single string and clean up whitespace
        final_text = "".join(processed_tokens)
        final_text = re.sub(r'\s+', ' ', final_text).strip()

        return final_text

    def __remove_noise_patterns(target: str) -> str:
        """
        Remove common headers, footers, and other formatting artifacts from legal texts.

        This function uses a curated list of high-confidence regular expressions to
        clean the text line by line, minimizing the risk of removing substantive content.

        Args:
            text: The text to be cleaned.

        Returns:
            The cleaned text.
        """

        # List of high-confidence regex patterns for noise lines.
        # Each pattern is compiled for efficiency and targets a specific type of noise.
        noise_patterns = [
            # Matches document identifiers like 'HC 115002 / SP' or 'REsp 1.276.871-SP'
            # This is more specific and safer than the original.
            re.compile(r'^\s*(HC|REsp|AgRg)\s+[\d.-]+\s*/\s*\w{2}\s*$', re.IGNORECASE),

            # Matches explicit page numbers like 'Página 1 de 10' or 'fls. 2'.
            # We REMOVED the risky `|[\d\s]+` part.
            re.compile(r'^\s*(página|pag|fls)\.?\s*\d+(\s*de\s*\d+)?\s*$', re.IGNORECASE),

            # Matches section titles ONLY if they appear in ALL CAPS, which is a common
            # formatting for titles and reduces the risk of removing content words.
            re.compile(r'^\s*(RELATÓRIO|VOTO|EMENTA|ACÓRDÃO|DECISÃO)\s*$', re.IGNORECASE),

            # Matches final signature/footer lines with a date and court division.
            re.compile(r'^\s*\d{2}/\d{2}/\d{4}\s+(PRIMEIRA|SEGUNDA)\s+(TURMA|C[ÂA]MARA)\s*$', re.IGNORECASE),

            # Matches lines that ONLY contain the court division (e.g., 'SEGUNDA TURMA').
            re.compile(r'^\s*(PRIMEIRA|SEGUNDA)\s+(TURMA|C[ÂA]MARA)\s*$', re.IGNORECASE),
        ]

        # Process the text line by line
        cleaned_lines = []
        for line in target.split('\n'):
            # If a line is empty or matches any of the noise patterns, it's skipped.
            if not line.strip():
                continue
            if any(pattern.fullmatch(line.strip()) for pattern in noise_patterns):
                continue

            cleaned_lines.append(line)

        # Join the cleaned lines back into a single text block.
        return "\n".join(cleaned_lines)


    # Step 0: Truncates the document at the beginning of the reasoning/vote section.
    text = __isolate_report_section(text)

    # Step 1: Remove lines that look like headers or footers from the remaining text.
    text = __remove_noise_patterns(text)


    # Step 2: Join lines broken by PDF formatting.
    text = _join_broken_lines(text)

    # Remove URLs
    text = __remove_authentication_block(text)

    # Step 3: Replace special characters, symbols,
    text = _replace_special_chars_pt(text)

    # # Step 4: Standardize legal entities like 'Art. 99' to 'ART_99'.
    # text = _standardize_legal_entities_v2(text)

    return  tokenize_and_normalize_spacy(text, nlp_spacy)