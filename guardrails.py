"""
RF04: Guardrails de entrada e saída.

- Entrada: valida escopo, tamanho e detecta tentativas de prompt injection.
- Saída: verifica se a resposta está fundamentada nos documentos recuperados.
"""

import re

# Palavras-chave do domínio agroclimático
_DOMAIN_KEYWORDS = {
    "safra", "cultura", "plantio", "colheita", "risco", "clima",
    "solo", "irrigado", "sequeiro", "municipio", "município", "estado",
    "uf", "brasil", "agro", "agricultur", "produtividade", "cevada",
    "soja", "milho", "arroz", "feijão", "trigo", "algodão", "café",
    "decada", "décade", "portaria", "zoneamento", "agroclimático",
    "sinistro", "manejo", "ciclo", "proagro", "lavoura", "semear",
    "fazenda", "campo", "cultura", "grão", "grãos",
}

# Padrões de injeção de prompt
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"esquece\s+(todas\s+)?as\s+instru",
    r"act\s+as\s+(?!an?\s+agr)",
    r"you\s+are\s+now\s+(?!an?\s+agr)",
    r"pretend\s+(to\s+be|you\s+are)",
    r"jailbreak",
    r"<\s*script",
    r"system\s*:\s*(you|ignore|forget)",
    r"prompt\s+injection",
    r"ignore\s+o\s+contexto",
    r"responda\s+sem\s+contexto",
    r"instrução\s+do\s+sistema",
    r"reveal\s+your\s+(system\s+)?prompt",
    r"mostre\s+o\s+prompt",
]

_MIN_LENGTH = 5
_MAX_LENGTH = 1000


class GuardrailError(Exception):
    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


def validate_input(query: str) -> str:
    """
    Valida a mensagem de entrada.
    Retorna a query limpa ou lança GuardrailError.
    """
    query = query.strip()

    if len(query) < _MIN_LENGTH:
        raise GuardrailError("Consulta muito curta.", reason="query_too_short")

    if len(query) > _MAX_LENGTH:
        raise GuardrailError("Consulta muito longa (máx. 1000 caracteres).", reason="query_too_long")

    lower = query.lower()

    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            raise GuardrailError(
                "Solicitação bloqueada: tentativa de manipulação detectada.",
                reason="prompt_injection",
            )

    words = set(re.findall(r"[a-záéíóúãõâêîôûçà]+", lower))
    if not words.intersection(_DOMAIN_KEYWORDS):
        raise GuardrailError(
            "Esta consulta está fora do escopo. "
            "O assistente responde apenas sobre a tábua de risco agroclimático safra 2026-2027. "
            "Pergunte sobre culturas, municípios, estados, riscos ou períodos de plantio.",
            reason="out_of_scope",
        )

    return query


def validate_output(response: str, context_docs: list[str]) -> str:
    """
    Valida a resposta gerada pelo LLM.
    Retorna a resposta ou lança GuardrailError.
    """
    if not response or len(response.strip()) < 10:
        raise GuardrailError("Resposta inválida gerada.", reason="empty_response")

    response_lower = response.lower()

    # Respostas que admitem ausência de evidência são válidas
    no_evidence_phrases = [
        "não encontrei", "não há informações", "não tenho dados",
        "não foi possível", "sem evidência", "dados insuficientes",
        "não consta", "não disponível", "fora do escopo",
    ]
    if any(p in response_lower for p in no_evidence_phrases):
        return response

    # Verifica overlap lexical entre resposta e contexto recuperado
    if context_docs:
        context_words = set()
        for doc in context_docs:
            context_words.update(re.findall(r"[a-záéíóúãõâêîôûçà]{4,}", doc.lower()))

        response_words = set(re.findall(r"[a-záéíóúãõâêîôûçà]{4,}", response_lower))
        overlap = response_words.intersection(context_words)

        if len(overlap) < 3:
            raise GuardrailError(
                "Resposta não fundamentada nos documentos recuperados.",
                reason="ungrounded_response",
            )

    return response
