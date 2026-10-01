import ast
import re
import unicodedata
import warnings
from numbers import Integral
from string import ascii_uppercase
from typing import Any

_LATEX_START = re.compile(r"(\\+)([A-Za-z])")
_WHITESPACE = re.compile(r"\s+")
_VALID_ANSWERS = set("ABCDE")
_FENCED_CODE = re.compile(
    r"^ {0,3}(?P<ticks>`{3,})[^\n]*\n.*?(?:^ {0,3}(?P=ticks)`*[ \t]*(?:\n|$)|\Z)"
    r"|^ {0,3}(?P<tildes>~{3,})[^\n]*\n.*?(?:^ {0,3}(?P=tildes)~*[ \t]*(?:\n|$)|\Z)",
    re.MULTILINE | re.DOTALL,
)
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)", re.DOTALL)
_IMAGE_MARKUP = re.compile(
    r"!\[(?:\\.|[^\]\\])*\]\s*(?:\([^)]*\)|\[[^\]]*\])"
    r"|<img(?=[\s/>])|data:image/",
    re.IGNORECASE,
)

PUBLIC_FIELDS = [
    "exam",
    "exam_edition",
    "exam_url",
    # "num_questions",
    "num",
    "question",
    "choices",
    "answer",
    # "url",
    "academic_level",
]


def keep_question(statement: Any) -> Any:
    """Renomeia statement sem alterar seu conteúdo."""
    return statement


def normalize_question_for_dedup(question: Any) -> str:
    """Normaliza diferenças mecânicas antes da deduplicação exata."""
    if not isinstance(question, str):
        return ""
    normalized = unicodedata.normalize("NFKC", question).casefold()
    return _WHITESPACE.sub(" ", normalized).strip()


def parse_alternatives(raw: Any) -> dict[str, Any]:
    """Converte o literal Python legado em uma estrutura fail-closed."""
    try:
        if isinstance(raw, dict):
            parsed = raw
        elif isinstance(raw, str):
            fixed = raw.replace("\r\n", "\\n").replace("\n", "\\n")

            def make_even(match: re.Match[str]) -> str:
                slashes, letter = match.groups()
                if len(slashes) % 2 == 1:
                    slashes += "\\"
                return slashes + letter

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                parsed = ast.literal_eval(_LATEX_START.sub(make_even, fixed))
        else:
            return {"choices": [], "labels": [], "parse_error": True}
    except (MemoryError, SyntaxError, TypeError, ValueError):
        return {"choices": [], "labels": [], "parse_error": True}

    if not isinstance(parsed, dict):
        return {"choices": [], "labels": [], "parse_error": True}
    choices = parsed.get("text")
    labels = parsed.get("label")
    return {
        "choices": choices if isinstance(choices, list) else [],
        "labels": labels if isinstance(labels, list) else [],
        "parse_error": not isinstance(choices, list) or not isinstance(labels, list),
    }


def normalize_answer(answer: Any) -> int:
    if isinstance(answer, str) and answer in _VALID_ANSWERS:
        return ascii_uppercase.index(answer)
    return -1


def extract_choices(parsed: Any) -> list[str]:
    if isinstance(parsed, dict) and isinstance(parsed.get("choices"), list):
        choices = parsed["choices"]
        if all(isinstance(choice, str) for choice in choices):
            return choices
    return []


def serialize_choices(choices: list[str]) -> str:
    """Combina as alternativas em um texto para contagem de palavras."""
    return "\n".join(choices)


def serialize_question_and_choices(*, question: str, choices: list[str]) -> str:
    """Combina o enunciado e as alternativas para contagem de palavras."""
    return "\n".join([question, *choices])


def group_question_and_choices(*, question: Any, choices: list[str]) -> list[Any]:
    """Reúne os textos sem misturar delimitadores de código entre campos."""
    return [question, *choices]


def has_no_images(texts: list[Any]) -> bool:
    """Mantém registros sem imagens fora de trechos de código Markdown."""
    for text in texts:
        if not isinstance(text, str):
            continue
        text = _FENCED_CODE.sub("\n", text)
        text = _INLINE_CODE.sub(" ", text)
        if _IMAGE_MARKUP.search(text):
            return False
    return True


def has_matching_alternative_lengths(parsed: Any) -> bool:
    """Mantém questões com a mesma quantidade de alternativas e labels."""
    if not isinstance(parsed, dict):
        return False

    choices = parsed.get("choices")
    labels = parsed.get("labels")
    return (
        isinstance(choices, list)
        and isinstance(labels, list)
        and len(choices) == len(labels)
    )


def has_described_choices(choices: Any) -> bool:
    """Mantém apenas questões com alternativas não vazias."""
    if isinstance(choices, (str, bytes, dict)):
        return False
    try:
        values = list(choices)
    except TypeError:
        return False
    return bool(values) and all(isinstance(choice, str) and bool(choice.strip()) for choice in values)


def has_supported_choice_count(choices: Any) -> bool:
    """Mantém apenas questões com quatro ou cinco alternativas."""
    if isinstance(choices, (str, bytes, dict)):
        return False
    try:
        return 4 <= len(choices) <= 5
    except TypeError:
        return False


def has_answer(answer: Any) -> bool:
    """Mantém apenas questões com uma resposta normalizada válida."""
    return isinstance(answer, Integral) and not isinstance(answer, bool) and answer >= 0


def group_answer_and_choices(*, answer: Any, choices: Any) -> tuple[Any, Any]:
    """Agrupa os campos necessários para validar o índice da resposta."""
    return answer, choices


def has_answer_in_bounds(answer_and_choices: Any) -> bool:
    """Mantém questões cujo índice de resposta existe nas alternativas."""
    if not isinstance(answer_and_choices, (list, tuple)) or len(answer_and_choices) != 2:
        return False

    answer, choices = answer_and_choices
    return has_answer(answer) and isinstance(choices, list) and answer < len(choices)
