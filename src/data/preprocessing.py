import re
import spacy


def preprocessing_legal_pt(text: str, nlp_spacy):
    """
    Realiza o pré-processamento de documentos jurídicos em português.

    Esta função executa uma série de etapas de limpeza e normalização
    adaptadas às características de textos jurídicos em português.

    A pipeline é a seguinte:
    0.  Isola a seção do relatório, truncando o texto após a última
        ocorrência de "É o relatório.".
    1.  Remove padrões comuns de cabeçalho e rodapé.
    2.  Une linhas que foram quebradas durante a extração de texto (ex: de PDFs).
    3.  Remove blocos de ruído específicos, como os de autenticação de documentos.
    4.  Substitui caracteres especiais, símbolos, URLs e expande abreviações/contrações
        comuns (ex: 'nº' e suas variações -> 'número', 'à' -> 'para a').
    5.  Expande números ordinais (ex: '1º', '2ª') para palavras completas ('primeiro', 'segunda').
    6.  Padroniza referências jurídicas comuns (ex: 'Art. 99' -> 'ARTIGO_99').
    7.  Usa o spaCy para tokenizar, normalizar (lematizar) e realizar o processamento linguístico.
    8.  Executa uma limpeza cosmética final no espaçamento.

    Args:
        text: Uma string contendo o documento jurídico em português.
        nlp_spacy: Um modelo spaCy carregado para a língua portuguesa.

    Returns:
        O texto pré-processado como uma única string.
    """

    def __remove_icp_brasil_footer(target: str) -> str:
        """
        Remove o bloco de texto da assinatura digital ICP-Brasil e seus resíduos.
        Atua em dois estágios para garantir a limpeza completa.
        """
        # Estágio 1: Remove o bloco principal, da "Infraestrutura" até o primeiro
        # "ICP-Brasil". A expressão é flexível para lidar com texto concatenado.
        # Usamos re.DOTALL para garantir que funcione mesmo que a junção de linhas falhe.
        main_pattern = re.compile(
            r"Infraestrutura\s+de.*?ICP-Brasil\.?",
            re.IGNORECASE | re.DOTALL
        )
        target = main_pattern.sub('', target)

        # Estágio 2: Remove o resíduo específico "ICP-Brasil.OO" que pode ter
        # sobrado após a primeira remoção.
        remnant_pattern = re.compile(r"ICP-Brasil\.OO", re.IGNORECASE)
        target = remnant_pattern.sub('', target)

        return target

    def _replace_special_chars_pt(target: str) -> str:
        """
        Executa normalização de caracteres e remoção de ruídos com alta confiança.
        Inclui a expansão de abreviações e contrações comuns em português.
        """
        # 1. Padroniza pontuação e remove caracteres de ruído
        target = target.replace('–', '-').replace('—', '-').replace('―', '-')
        target = target.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        target = target.replace('…', '...').replace('•', '').replace('·', '')
        target = target.replace('\ufeff', '')  # Remove o caractere BOM invisível

        # 2. Expande contrações
        target = re.sub(r'\bà\b', 'para a', target, flags=re.IGNORECASE)
        target = re.sub(r'\bàs\b', 'para as', target, flags=re.IGNORECASE)

        # 3. Expande a abreviação "número" com uma abordagem segura de duas etapas
        # Etapa 3.1: Substitui casos inequívocos que NUNCA são preposições.
        # Trata n°, n.°, nº, n.º, n0, n.0
        target = re.sub(r'\bn\.?[º°0]\b', 'número', target, flags=re.IGNORECASE)

        # Etapa 3.2: Substitui o ambíguo "no" ou "n.o" APENAS se seguido por um dígito.
        # Usa um "positive lookahead" (?=\s*\d) para verificar o dígito sem consumi-lo.
        target = re.sub(r'\bn\.?o\b(?=\s*\d)', 'número', target, flags=re.IGNORECASE)

        # 4. Padroniza URLs
        target = re.sub(r'https?://\S+|www\.\S+', '[URL]', target)

        return target

    def _expand_ordinals_pt(target: str) -> str:
        """Expande números ordinais para palavras completas (ex: '1º' -> 'primeiro')."""
        ordinal_map = {
            1: "primeir", 2: "segund", 3: "terceir", 4: "quart",
            5: "quint", 6: "sext", 7: "sétim", 8: "oitav",
            9: "non", 10: "décim"
        }

        def _replacer(match):
            num_str, indicator = match.group(1), match.group(2).lower()
            num = int(num_str)
            stem = ordinal_map.get(num)
            if not stem: return match.group(0)
            if indicator in ['o', 'º']: return stem + 'o'
            if indicator in ['a', 'ª']: return stem + 'a'
            return match.group(0)

        pattern = re.compile(r'\b(\d+)([ºªoa])\b', re.IGNORECASE)
        return pattern.sub(_replacer, target)

    def _standardize_legal_entities_v2(target: str) -> str:
        """Padroniza entidades jurídicas como Artigos, lidando com intervalos e listas."""
        target = target.replace('§', 'paragrafo')
        keyword, keywords_plural = "artigo", "artigos"

        def expand_range(match):
            start, end = int(match.group(1)), int(match.group(2))
            return ', '.join([f"{keyword.upper()}_{num}" for num in range(start, end + 1)])

        target = re.sub(fr'\b{keywords_plural}\s+(\d+)\s+a\s+(\d+)\b', expand_range, target, flags=re.IGNORECASE)

        def expand_list(match):
            numbers = re.findall(r'\d+', match.group(1))
            return ', '.join([f"{keyword.upper()}_{num}" for num in numbers])

        target = re.sub(fr'\b{keywords_plural}\s+((?:\d+,\s*)*\d+\s+e\s+\d+)\b', expand_list, target,
                        flags=re.IGNORECASE)
        target = re.sub(fr'\b({keyword})\.?\s*([\d./-]+)\b',
                        lambda m: f"{m.group(1).upper().replace('.', '')}_{m.group(2).replace('.', '')}", target,
                        flags=re.IGNORECASE)
        return target

    def __isolate_report_section(target: str) -> str:
        """Isola o texto até a última ocorrência de 'É o relatório.'."""
        matches = list(re.finditer(r'\bÉ o relatório\.?\b', target, re.IGNORECASE | re.DOTALL))
        if matches: return target[:matches[-1].end()]
        return target

    def _join_broken_lines(target: str) -> str:
        """Une linhas que foram quebradas incorretamente."""
        lines, reconstructed_lines, buffer = target.split('\n'), [], ""
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line: continue
            if buffer and not buffer.endswith(('.', '!', '?', ';', ':')):
                buffer += " " + stripped_line
            else:
                if buffer: reconstructed_lines.append(buffer)
                buffer = stripped_line
        if buffer: reconstructed_lines.append(buffer)
        return "\n".join(reconstructed_lines)

    def __remove_line_noise(target: str) -> str:
        """
        Remove linhas inteiras que contêm padrões de ruído de rodapé.
        Executar ANTES de _join_broken_lines para evitar a concatenação
        incorreta de ruído com texto legítimo.
        """
        # Palavras-chave que, se presentes, marcam a linha inteira como ruído.
        noise_keywords = [
            'ICP-Brasil',
            'Infraestrutura de',
            'deChaves',
            'ChavesPúblicas',
            'PúblicasBrasileira',
            'stf.jus.br/portal/autenticacao/'
        ]

        lines = target.split('\n')
        # Mantém apenas as linhas que NÃO contêm nenhuma das palavras-chave de ruído.
        clean_lines = [line for line in lines if not any(keyword in line for keyword in noise_keywords)]

        return "\n".join(clean_lines)

    def __remove_authentication_block(target: str) -> str:
        """Remove o bloco de autenticação de documentos do STF."""
        return re.sub(r'https?://www\.stf\.jus\.br/portal/autenticacao/.*', '', target, flags=re.IGNORECASE | re.DOTALL)

    def __remove_noise_patterns(target: str) -> str:
        """Remove linhas inteiras que correspondem a padrões de ruído (cabeçalhos, etc.)."""
        noise_patterns = [
            re.compile(r'^\s*(HC|REsp|AgRg)\s+[\d.-]+\s*/\s*\w{2}\s*$', re.IGNORECASE),
            re.compile(r'^\s*(página|pag|fls)\.?\s*\d+(\s*de\s*\d+)?\s*$', re.IGNORECASE),
            re.compile(r'^\s*(RELATÓRIO|VOTO|EMENTA|ACÓRDÃO|DECISÃO)\s*$', re.IGNORECASE),
            re.compile(r'^\s*\d{2}/\d{2}/\d{4}\s+(PRIMEIRA|SEGUNDA)\s+(TURMA|C[ÂA]MARA)\s*$', re.IGNORECASE),
            re.compile(r'^\s*(PRIMEIRA|SEGUNDA)\s+(TURMA|C[ÂA]MARA)\s*$', re.IGNORECASE),
        ]
        return "\n".join([line for line in target.split('\n') if
                          line.strip() and not any(p.fullmatch(line.strip()) for p in noise_patterns)])

    def tokenize_and_normalize_spacy(target: str) -> str:
        """Usa spaCy para tokenização, normalização e preservação de entidades/propns."""
        doc, processed_tokens = nlp_spacy(target), []
        for token in doc:
            if '_' in token.text or token.pos_ == "PROPN" or token.text == '[URL]':
                processed_tokens.append(token.text_with_ws)
            else:
                processed_tokens.append(token.text_with_ws.lower())
        final_text = "".join(processed_tokens)
        final_text = re.sub(r'\s+', ' ', final_text).strip()
        return re.sub(r'\s([.,;:])', r'\1', final_text)

    # --- PIPELINE DE EXECUÇÃO ---
    text = __isolate_report_section(text)
    text = __remove_noise_patterns(text)

    # ORDEM CORRIGIDA E DEFINITIVA
    # 1. Remove as linhas de ruído ANTES de qualquer outra coisa.
    text = __remove_line_noise(text)

    # 2. Agora, junta as linhas quebradas com segurança.
    text = _join_broken_lines(text)

    # 3. Executa o resto da limpeza.
    text = _replace_special_chars_pt(text)
    text = _expand_ordinals_pt(text)
    text = _standardize_legal_entities_v2(text)

    # Limpeza cosmética final para remover espaços em branco excessivos.
    text = re.sub(r'\s{2,}', ' ', text).strip()
    text = re.sub(r'(\n\s*)+\n', '\n\n', text).strip()

    return text


import re
from typing import List, Tuple


# --- Funções Auxiliares (Nível de Módulo) ---

def _create_spaced_pattern(word: str) -> str:
    """
    Cria um padrão regex para uma palavra com espaços opcionais entre as letras.
    Isso lida com formatações como 'R E L A T Ó R I O'.
    """
    escaped_word = re.escape(word)
    return r'\s*'.join(list(escaped_word))


def _remove_noise_patterns(target: str) -> str:
    """Remove linhas que são apenas cabeçalhos de seção, marcadores de página ou metadados gerais."""
    # Padrões para os títulos que, após usados como marcadores, devem ser removidos do corpo do texto.
    relatorio_spaced = _create_spaced_pattern("RELATÓRIO")
    voto_spaced = _create_spaced_pattern("VOTO")
    vista_spaced = _create_spaced_pattern("VISTA")
    ementa_spaced = _create_spaced_pattern("EMENTA")
    acordao_spaced = _create_spaced_pattern("ACÓRDÃO")
    decisao_spaced = _create_spaced_pattern("DECISÃO")

    noise_patterns = [
        # Remove os próprios títulos das seções
        re.compile(
            fr'^\s*({relatorio_spaced}|{voto_spaced}(?:[ –-]\s*{vista_spaced})?(?: - MIN\. [\w\s()]+)?|{ementa_spaced}|{acordao_spaced}|{decisao_spaced})\s*$',
            re.IGNORECASE),
        # Remove identificadores de processo (ex: HC 123.456 / SP)
        re.compile(r'^\s*(HC|REsp|AgRg|RHC)\s+[\d.-]+\s*/\s*\w{2}\s*$', re.IGNORECASE),
        # Remove marcadores de página (ex: fls. 123)
        re.compile(r'^\s*(página|pag|fls)\.?\s*\d+(\s*de\s*\d+)?\s*$', re.IGNORECASE),
        # Remove cabeçalhos de data e turma
        re.compile(r'^\s*\d{2}/\d{2}/\d{4}\s+(PRIMEIRA|SEGUNDA)\s+TURMA\s*$', re.IGNORECASE),
        re.compile(r'^\s*(PRIMEIRA|SEGUNDA)\s+TURMA\s*$', re.IGNORECASE),
        re.compile(r'^\s*(PLENÁRIO)\s*$', re.IGNORECASE),
        # Remove assinaturas de ministros
        re.compile(
            r'^\s*Ministro\s+[\w\s]+(\s*-\s*(?:Relator(?:a)?|Presidente e Relator|Redator(?:a)? para o acórdão))?\s*$',
            re.IGNORECASE)
    ]

    clean_lines = []
    for line in target.split('\n'):
        stripped_line = line.strip()
        # Mantém a linha apenas se ela não for vazia e não corresponder a nenhum padrão de ruído
        if stripped_line and not any(p.fullmatch(stripped_line) for p in noise_patterns):
            clean_lines.append(line)

    return "\n".join(clean_lines)


def _remove_section_metadata(target: str) -> str:
    """Remove o bloco de metadados (RELATOR, PACTE, etc.) do início de uma seção."""
    metadata_pattern = re.compile(
        r'^\s*((?:(?:RELATOR|PACTE\.\(S\)|IMPTE\.\(S\)|COATOR\(A/S\)\(ES\)|ADV\.\(A/S\)|PROC\.\(A/S\)\(ES\))\s*:.+?\n)+)',
        re.IGNORECASE | re.MULTILINE
    )
    return metadata_pattern.sub('', target).strip()


def _remove_footers_and_auth(target: str) -> str:
    """Remove blocos de autenticação e rodapés de assinatura digital."""
    auth_pattern = re.compile(r'http://www\.stf\.jus\.br/portal/autenticacao/.*?(?:\n|$)', re.IGNORECASE | re.DOTALL)
    target = auth_pattern.sub('', target)
    icp_pattern = re.compile(
        r'.*?(?:ICP-Brasil|Documento\s+assinado\s+digitalmente|Infraestrutura\s+de\s+Chaves\s+Públicas\s+Brasileira).*?(?:\n|$)',
        re.IGNORECASE)
    target = icp_pattern.sub('', target)
    return target.strip()


def _join_broken_lines_and_paragraphs(target: str) -> str:
    """Junta linhas quebradas no meio da frase, mas preserva parágrafos (linhas em branco)."""
    # Substitui quebras de linha únicas (dentro de um parágrafo) por espaço
    # Preserva quebras de linha duplas (entre parágrafos)
    return re.sub(r'(?<!\n)\n(?!\n)', ' ', target)


def _replace_special_chars_pt(target: str) -> str:
    """Normaliza caracteres, expande abreviações e remove ruídos."""
    target = target.replace('–', '-').replace('—', '-').replace('―', '-')
    target = target.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    target = target.replace('…', '...').replace('•', '').replace('·', '')
    target = re.sub(r'\bn\.?[º°0]\b', 'número', target, flags=re.IGNORECASE)
    target = re.sub(r'\bn\.?o\b(?=\s*\d)', 'número', target, flags=re.IGNORECASE)
    return target


def _standardize_legal_entities(target: str) -> str:
    """Padroniza referências jurídicas como Artigos, parágrafos, etc., para facilitar a análise."""
    target = target.replace('§', 'paragrafo')
    target = re.sub(r'\b(art|arts)\.?\s*([\d./-]+)\b', lambda m: f"ARTIGO_{m.group(2).replace('.', '')}", target,
                    flags=re.IGNORECASE)
    return target


def _extract_and_structure_sections(text: str) -> List[Tuple[str, str]]:
    """
    Função interna para extrair o conteúdo das seções 'RELATÓRIO' e 'VOTO' de forma estruturada.
    Retorna uma lista de tuplas: (título_da_seção, texto_da_seção).
    """
    relatorio_spaced = _create_spaced_pattern("RELATÓRIO")
    voto_spaced = _create_spaced_pattern("VOTO")
    vista_spaced = _create_spaced_pattern("VISTA")

    # Aprimorado para aceitar parênteses no nome do Ministro, ex: (RELATOR)
    start_pattern = re.compile(
        fr"^\s*({relatorio_spaced}|{voto_spaced}(?:[ –-]\s*{vista_spaced})?(?: - MIN\. [\w\s()]+)?)\s*$",
        re.IGNORECASE | re.MULTILINE
    )

    end_markers = [
        "Extrato de Ata", "Decisão de Julgamento", "DEBATE", "EXPLICAÇÃO",
        "NOTAS PARA O VOTO", "ANTECIPAÇÃO AO VOTO", "RETIFICAÇÃO DE VOTO",
        "CONFIRMAÇÃO DE VOTO", "ESCLARECIMENTO"
    ]
    end_pattern_str = "|".join([_create_spaced_pattern(marker) for marker in end_markers])
    end_pattern = re.compile(fr"^\s*({end_pattern_str})", re.IGNORECASE | re.MULTILINE)

    start_matches = list(start_pattern.finditer(text))
    if not start_matches:
        return []

    end_match = end_pattern.search(text, start_matches[0].start())
    end_index = end_match.start() if end_match else len(text)

    structured_sections = []

    for i, current_match in enumerate(start_matches):
        section_title = current_match.group(1).strip()
        section_title = re.sub(r'\s+', ' ', section_title)  # Normaliza espaçamento no título

        section_start_index = current_match.end()
        is_last_section = (i + 1) == len(start_matches)

        section_end_index = start_matches[i + 1].start() if not is_last_section else end_index

        if section_end_index > end_index:
            section_end_index = end_index

        section_text = text[section_start_index:section_end_index].strip()
        if section_text:
            structured_sections.append((section_title, section_text))

    return structured_sections


# --- FUNÇÃO PRINCIPAL PARA O USUÁRIO ---

def preprocessing_legal_pt_voto_relatorio(text: str) -> str:
    """
    Realiza o pré-processamento de acórdãos, extraindo e limpando o relatório e
    os votos, e retorna o resultado como uma string formatada e legível.

    A pipeline executa os seguintes passos:
    1.  Extrai as seções "RELATÓRIO" e "VOTO" de forma estruturada.
    2.  Para cada seção extraída, aplica uma série de limpezas:
        - Remove metadados (RELATOR, PACTE, etc.).
        - Remove rodapés e links de autenticação.
        - Remove ruídos de linha (cabeçalhos, paginação).
        - Junta linhas quebradas, preservando os parágrafos.
        - Normaliza caracteres e expande abreviações.
        - Padroniza entidades jurídicas (ex: Art. 99 -> ARTIGO_99).
    3.  Formata a saída final como um texto único, com títulos de seção
        claramente identificados.

    Args:
        text: Uma string contendo o documento jurídico completo.

    Returns:
        Uma única string com o conteúdo pré-processado e formatado, ou uma
        string vazia se nenhuma seção relevante for encontrada.
    """
    structured_sections = _extract_and_structure_sections(text)
    if not structured_sections:
        return ""

    formatted_output = []
    for title, section_text in structured_sections:
        # Pipeline de limpeza aplicada a cada seção individualmente
        clean_text = _remove_section_metadata(section_text)
        clean_text = _remove_footers_and_auth(clean_text)
        clean_text = _remove_noise_patterns(clean_text)
        clean_text = _join_broken_lines_and_paragraphs(clean_text)
        clean_text = _replace_special_chars_pt(clean_text)
        clean_text = _standardize_legal_entities(clean_text)

        # Limpeza cosmética final
        clean_text = re.sub(r'\s{2,}', ' ', clean_text).strip()

        # Formata a seção com seu título e adiciona à lista de saída
        if clean_text:
            # Capitaliza o título para um formato mais limpo e consistente
            formatted_title = title.upper().replace("MIN.", "MINISTRO")
            formatted_section = f"{formatted_title}:\n{clean_text}"
            formatted_output.append(formatted_section)

    # Junta todas as seções formatadas com duas quebras de linha
    return "\n\n".join(formatted_output)
