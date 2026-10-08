"""Rationale generation with the OpenAI Responses API and append-only caching."""

import functools
import json
import os
from pathlib import Path

from .config import LETTERS, PROMPT_VERSION, RunConfig, append_jsonl, digest, letter, timestamp

RATIONALE_FILE = "rationale.jsonl"

DEVELOPER_PROMPT = """Você é um professor brasileiro que escreve resoluções curtas de questões de múltipla escolha, usadas como exemplos resolvidos em avaliações de modelos de linguagem.
Você receberá uma questão, as alternativas rotuladas e o gabarito. Escreva uma explicação em português do Brasil, em prosa corrida, com 2 a 5 frases, que justifique por que a alternativa do gabarito está correta e, quando útil, por que as alternativas mais plausíveis estão erradas.
Regras:
- O gabarito é a referência; não o conteste.
- Não repita o enunciado nem as alternativas; não use listas, títulos ou Markdown.
- Não escreva "Resposta:" nem termine com a letra: a linha final será acrescentada automaticamente.
- Preencha "answer_letter" exatamente com a letra do gabarito informado."""

USER_TEMPLATE = """Área: {macro_area} — {subject}. Nível: {academic_level}.

Questão:
{question}

{labeled_choices}

Gabarito: {letter}"""

RATIONALE_SCHEMA = {
    "type": "json_schema",
    "name": "fewshot_rationale",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "explanation": {"type": "string"},
            "answer_letter": {"type": "string", "enum": list(LETTERS)},
        },
        "required": ["explanation", "answer_letter"],
        "additionalProperties": False,
    },
}


def prompt_hash(config: RunConfig) -> str:
    return digest([PROMPT_VERSION, DEVELOPER_PROMPT, USER_TEMPLATE, RATIONALE_SCHEMA,
                   config.model, config.reasoning_effort])


@functools.cache
def create_client():
    """Lazily created client; the SDK retries transient failures on its own."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY before generating rationales")
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=5, timeout=120)


def labeled_choices(choices: list[str]) -> str:
    return "\n".join(f"({letter(index)}) {choice}" for index, choice in enumerate(choices))


def response_body(row: dict, gold: str, config: RunConfig, prompt_digest: str) -> dict:
    user = USER_TEMPLATE.format(macro_area=row["macro_area"], subject=row["subject"],
                                academic_level=row["academic_level"], question=row["question"],
                                labeled_choices=labeled_choices(row["choices"]), letter=gold)
    return {"model": config.model,
            "input": [{"role": "developer", "content": DEVELOPER_PROMPT}, {"role": "user", "content": user}],
            "reasoning": {"effort": config.reasoning_effort},
            "text": {"format": RATIONALE_SCHEMA},
            "max_output_tokens": config.max_output_tokens,
            "store": False,
            "prompt_cache_key": f"mmlu-pt-fewshot-rationale-{prompt_digest[:16]}"}


def extract_text(body: dict) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    chunks = [content["text"] for item in body.get("output") or [] if item.get("type") == "message"
              for content in item.get("content") or [] if content.get("type") == "output_text"]
    text = "\n".join(chunks).strip()
    if not text:
        raise ValueError("response has no output_text")
    return text


def compose_rationale(explanation: str, answer_letter: str) -> str:
    return explanation.strip() + "\nResposta: " + answer_letter


def generate(row: dict, row_id: str, gold: str, attempt: int, config: RunConfig, prompt_digest: str) -> dict:
    """One API call; failures are recorded in the result instead of raised."""
    record = {"id": row_id, "attempt": attempt, "model": config.model, "prompt_hash": prompt_digest,
              "created_at": timestamp()}
    try:
        response = create_client().responses.create(**response_body(row, gold, config, prompt_digest))
        body = response.model_dump(mode="json")
        parsed = json.loads(extract_text(body))
        record |= {"status": "ok", "explanation": parsed["explanation"], "answer_letter": parsed["answer_letter"],
                   "rationale": compose_rationale(parsed["explanation"], parsed["answer_letter"]),
                   "usage": body.get("usage")}
    except Exception as error:  # API, parsing or schema failures: audited, then regenerated or replaced
        record |= {"status": "error", "error": f"{type(error).__name__}: {error}"}
        print(f"rationale error for {row_id[:12]} attempt {attempt}: {record['error']}")
    return record


def cached_record(records: list[dict], row_id: str, attempt: int, config: RunConfig, prompt_digest: str) -> dict | None:
    for record in records:
        if (record["id"], record["attempt"], record["model"], record["prompt_hash"]) == \
                (row_id, attempt, config.model, prompt_digest):
            return record
    return None


def obtain(row: dict, row_id: str, gold: str, attempt: int, records: list[dict], run_dir: Path,
           config: RunConfig, prompt_digest: str, allow_call: bool) -> tuple[dict | None, bool]:
    """Cached record for this attempt, or a fresh one when calls are allowed; returns (record, called)."""
    cached = cached_record(records, row_id, attempt, config, prompt_digest)
    if cached is not None:
        return cached, False
    if not allow_call:
        return None, False
    record = generate(row, row_id, gold, attempt, config, prompt_digest)
    append_jsonl(run_dir / RATIONALE_FILE, record)
    records.append(record)
    return record, True
