"""Lógica compartilhada pelos transportes síncrono e Batch."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

DEDUPLICATION_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = DEDUPLICATION_ROOT / "assets/prompts/question_pair_deduplication.md"

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 4_000

REQUESTS_SUFFIX = "judge-requests.jsonl"
STATE_SUFFIX = "batch-state.json"
SYNC_STATE_SUFFIX = "sync-state.json"
RAW_OUTPUT_SUFFIX = "responses.jsonl"
RAW_ERRORS_SUFFIX = "errors.jsonl"
JUDGED_SUFFIX = "judged.csv"

REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
VALID_LABELS = (
    "duplicate",
    "related_but_distinct",
    "distinct",
    "unsure",
)
REQUIRED_COLUMNS = (
    "left_exam",
    "right_exam",
    "left_question",
    "right_question",
    "left_choices",
    "right_choices",
    "left_answer",
    "right_answer",
)

OUTPUT_SCHEMA = {
    "type": "json_schema",
    "name": "question_pair_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": list(VALID_LABELS),
            },
            "notes": {
                "type": "string",
                "description": (
                    "Exactly one concise English sentence explaining the decisive "
                    "semantic reason."
                ),
            },
        },
        "required": ["label", "notes"],
        "additionalProperties": False,
    },
}


def artifact_path(csv_path: Path, suffix: str) -> Path:
    """Cria o caminho padrão de um artefato ao lado do CSV."""
    return csv_path.with_name(f"{csv_path.stem}.{suffix}")


def metadata_path(requests_path: Path) -> Path:
    """Retorna o sidecar associado ao JSONL de requisições."""
    return requests_path.with_suffix(".meta.json")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Grava um objeto JSON legível em UTF-8."""
    ensure_parent(path)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Acrescenta um registro JSON a um arquivo JSONL."""
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(value, ensure_ascii=False) + "\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Calcula o SHA-256 de um arquivo sem carregá-lo inteiro na memória."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_prompt() -> tuple[str, str]:
    """Carrega o prompt e retorna seu conteúdo e SHA-256."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"O prompt está vazio: {PROMPT_PATH}")
    return prompt, sha256_bytes(prompt.encode("utf-8"))


def create_openai_client() -> Any:
    """Cria o cliente usando exclusivamente OPENAI_API_KEY."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Defina OPENAI_API_KEY antes de executar comandos da API.")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "O SDK da OpenAI não está instalado. Execute: "
            "uv sync --group annotation"
        ) from error
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def json_safe(value: Any) -> Any:
    """Converte valores pandas e NumPy em valores serializáveis como JSON."""
    if is_missing(value):
        return None

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return int(value)

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
        return value

    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return value


def has_nonempty_value(value: Any) -> bool:
    return not is_missing(value) and bool(str(value).strip())


def validate_dataframe(data: pd.DataFrame) -> None:
    """Verifica o contrato mínimo do CSV de pares."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"O CSV não contém as colunas obrigatórias: {missing}")


def select_rows(
    data: pd.DataFrame,
    *,
    include_labeled: bool,
    limit: int | None,
) -> list[tuple[int, pd.Series]]:
    """Seleciona deterministicamente as primeiras linhas elegíveis."""
    rows: list[tuple[int, pd.Series]] = []
    for row_index, row in data.iterrows():
        existing_label = row.get("label") if "label" in data.columns else None
        if not include_labeled and has_nonempty_value(existing_label):
            continue
        rows.append((int(row_index), row))
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError(
            "Nenhuma linha foi selecionada; o CSV pode já estar completamente anotado."
        )
    return rows


def prepare_input(
    csv_path: Path,
    model: str,
    reasoning_effort: str,
    *,
    include_labeled: bool,
    limit: int | None,
) -> tuple[list[tuple[int, pd.Series]], str, str, dict[str, Any]]:
    """Carrega o CSV, o prompt e os metadados comuns da execução."""
    data = pd.read_csv(csv_path)
    validate_dataframe(data)
    rows = select_rows(
        data,
        include_labeled=include_labeled,
        limit=limit,
    )
    prompt, prompt_hash = read_prompt()
    metadata = {
        "csv": str(csv_path.resolve()),
        "csv_sha256": sha256_file(csv_path),
        "prompt": str(PROMPT_PATH.resolve()),
        "prompt_sha256": prompt_hash,
        "request_count": len(rows),
        "row_indices": [row_index for row_index, _row in rows],
        "model": model,
        "reasoning_effort": reasoning_effort,
    }
    return rows, prompt, prompt_hash, metadata


def make_user_payload(row: pd.Series) -> str:
    payload = {column: json_safe(row.get(column)) for column in REQUIRED_COLUMNS}
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        "Classify this ONE question pair according to the annotation guidelines."
        f"\n\n{serialized}"
    )


def make_response_body(
    row: pd.Series,
    prompt: str,
    prompt_hash: str,
    model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    """Monta o corpo comum da Responses API."""
    return {
        "model": model,
        "input": [
            {"role": "developer", "content": prompt},
            {"role": "user", "content": make_user_payload(row)},
        ],
        "reasoning": {"effort": reasoning_effort},
        "text": {"format": OUTPUT_SCHEMA},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "prompt_cache_key": f"question-pair-dedup-{prompt_hash[:16]}",
    }


def make_response_record(
    row_index: int,
    response_body: dict[str, Any],
) -> dict[str, Any]:
    """Normaliza uma resposta síncrona para o envelope usado pelo Batch."""
    return {
        "custom_id": f"row_{row_index:06d}",
        "response": {
            "status_code": 200,
            "body": response_body,
        },
        "error": None,
    }


def serialize_response(response: Any) -> dict[str, Any]:
    """Converte a resposta do SDK em um dicionário JSON-safe."""
    if isinstance(response, dict):
        return response
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    raise TypeError("O SDK retornou uma resposta em formato inesperado")


def extract_response_text(body: dict[str, Any]) -> str:
    """Extrai o texto de uma resposta bruta da Responses API."""
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    for item in body.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if content.get("type") == "output_text" and isinstance(text, str):
                chunks.append(text)

    response_text = "\n".join(chunks).strip()
    if not response_text:
        raise ValueError("A resposta não contém output_text")
    return response_text


def parse_judgment(body: dict[str, Any]) -> dict[str, str]:
    """Valida o julgamento estruturado contido em uma resposta."""
    judgment = json.loads(extract_response_text(body))
    if not isinstance(judgment, dict):
        raise ValueError("O julgamento não contém um objeto JSON")

    label = judgment.get("label")
    notes = judgment.get("notes")
    if label not in VALID_LABELS:
        raise ValueError(f"Label inválida: {label!r}")
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("O julgamento não contém notes")
    return {"label": label, "notes": notes.strip()}


def parse_response_records(
    output_path: Path,
) -> tuple[dict[int, dict[str, str]], list[dict[str, Any]]]:
    """Lê respostas normalizadas e preserva falhas para auditoria."""
    judgments: dict[int, dict[str, str]] = {}
    errors: list[dict[str, Any]] = []
    if not output_path.exists():
        return judgments, errors

    with output_path.open("r", encoding="utf-8") as output_file:
        for line_number, raw_line in enumerate(output_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            record: dict[str, Any] = {}
            custom_id = ""
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise ValueError("A linha JSONL não contém um objeto")
                record = parsed
                custom_id = record.get("custom_id", "")
                if not isinstance(custom_id, str) or not custom_id.startswith("row_"):
                    raise ValueError(f"custom_id inesperado: {custom_id!r}")

                row_index = int(custom_id.removeprefix("row_"))
                if row_index in judgments:
                    raise ValueError(f"custom_id repetido: {custom_id}")
                if record.get("error") is not None:
                    raise RuntimeError(f"Erro da requisição: {record['error']}")

                response = record.get("response") or {}
                if not isinstance(response, dict):
                    raise ValueError("A resposta não contém um objeto")
                status_code = response.get("status_code")
                if status_code != 200:
                    raise RuntimeError(f"Status HTTP {status_code}: {response}")

                body = response.get("body") or {}
                if not isinstance(body, dict):
                    raise ValueError("O body da resposta não contém um objeto")
                judgments[row_index] = parse_judgment(body)
            except (json.JSONDecodeError, TypeError, ValueError, RuntimeError) as error:
                errors.append(
                    {
                        "line_number": line_number,
                        "custom_id": custom_id,
                        "error": str(error),
                        "record": record or line,
                    }
                )

    return judgments, errors


def validate_csv_integrity(csv_path: Path, state: dict[str, Any]) -> None:
    """Impede a coleta sobre um CSV diferente do usado no preparo."""
    expected_hash = state.get("csv_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        return
    if sha256_file(csv_path) != expected_hash:
        raise RuntimeError(
            "O CSV foi alterado desde o início da execução. Crie uma nova "
            "execução antes de coletar os resultados."
        )


def load_json(path: Path) -> dict[str, Any]:
    """Carrega e valida superficialmente um objeto JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Objeto JSON inválido: {path}")
    return value


def audit_path(output_csv_path: Path) -> Path:
    return output_csv_path.with_suffix(".collect-errors.json")


def expected_indices(state: dict[str, Any], judgments: dict[int, Any]) -> set[int]:
    raw_indices = state.get("row_indices")
    if raw_indices is None:
        return set(judgments)
    if not isinstance(raw_indices, list) or not all(
        isinstance(index, int) for index in raw_indices
    ):
        raise ValueError("O estado contém row_indices inválidos")
    return set(raw_indices)


def merge_results(
    csv_path: Path,
    state: dict[str, Any],
    raw_output_path: Path,
    output_csv_path: Path,
    *,
    allow_partial: bool,
) -> None:
    """Valida e mescla respostas de qualquer transporte ao CSV."""
    validate_csv_integrity(csv_path, state)
    judgments, parse_errors = parse_response_records(raw_output_path)
    expected = expected_indices(state, judgments)

    data = pd.read_csv(csv_path)
    validate_dataframe(data)
    for column in ("label", "notes"):
        if column not in data.columns:
            data[column] = pd.Series(pd.NA, index=data.index, dtype="string")
        else:
            data[column] = data[column].astype("string")

    merged = 0
    for row_index, judgment in judgments.items():
        if row_index not in expected:
            parse_errors.append(
                {
                    "custom_id": f"row_{row_index:06d}",
                    "error": "O índice não pertence a esta execução",
                }
            )
            continue
        if row_index < 0 or row_index >= len(data):
            parse_errors.append(
                {
                    "custom_id": f"row_{row_index:06d}",
                    "error": "O índice está fora dos limites do CSV",
                }
            )
            continue
        data.at[row_index, "label"] = judgment["label"]
        data.at[row_index, "notes"] = judgment["notes"]
        merged += 1

    missing = sorted(expected - set(judgments))
    audit_file = audit_path(output_csv_path)
    if parse_errors or missing:
        write_json(
            audit_file,
            {
                "parse_errors": parse_errors,
                "missing_row_indices": missing,
            },
        )
        if not allow_partial:
            raise RuntimeError(
                f"A coleta encontrou {len(parse_errors)} erros e "
                f"{len(missing)} julgamentos ausentes. Consulte {audit_file} ou "
                "use --allow-partial para produzir um CSV parcial."
            )
    else:
        audit_file.unlink(missing_ok=True)

    ensure_parent(output_csv_path)
    data.to_csv(output_csv_path, index=False)
    print(f"Mesclados {merged} julgamentos -> {output_csv_path}")
    if parse_errors or missing:
        print(
            f"AVISO: saída parcial; erros={len(parse_errors)}, "
            f"ausentes={len(missing)}",
            file=sys.stderr,
        )
