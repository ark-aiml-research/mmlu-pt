"""Transporte síncrono baseado na Responses API."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .judge_common import (
    append_jsonl,
    create_openai_client,
    ensure_parent,
    load_json,
    make_response_body,
    make_response_record,
    merge_results,
    parse_judgment,
    parse_response_records,
    prepare_input,
    serialize_response,
    write_json,
)

RESUME_FIELDS = (
    "csv_sha256",
    "prompt_sha256",
    "row_indices",
    "model",
    "reasoning_effort",
)


def validate_resume_state(
    saved_state: dict[str, Any],
    current_state: dict[str, Any],
) -> None:
    """Confirma que o checkpoint pertence à mesma execução lógica."""
    changed = [
        field
        for field in RESUME_FIELDS
        if saved_state.get(field) != current_state.get(field)
    ]
    if changed:
        raise RuntimeError(
            "O checkpoint não corresponde à configuração atual. Campos "
            f"divergentes: {', '.join(changed)}"
        )


def completed_indices(raw_output_path: Path) -> set[int]:
    """Lê as respostas válidas que já foram persistidas."""
    judgments, errors = parse_response_records(raw_output_path)
    if errors:
        raise RuntimeError(
            f"O checkpoint contém {len(errors)} respostas inválidas; consulte "
            f"{raw_output_path} antes de retomá-lo."
        )
    return set(judgments)


def initialize_run_files(
    state_path: Path,
    raw_output_path: Path,
    raw_error_path: Path,
    metadata: dict[str, Any],
    *,
    resume: bool,
) -> set[int]:
    """Inicializa uma execução nova ou recupera seu checkpoint."""
    if resume:
        saved_state = load_json(state_path)
        validate_resume_state(saved_state, metadata)
        return completed_indices(raw_output_path)

    write_json(state_path, metadata)
    for path in (raw_output_path, raw_error_path):
        ensure_parent(path)
        path.write_text("", encoding="utf-8")
    return set()


def run_sync(
    csv_path: Path,
    state_path: Path,
    output_csv_path: Path,
    raw_output_path: Path,
    raw_error_path: Path,
    model: str,
    reasoning_effort: str,
    *,
    include_labeled: bool,
    allow_partial: bool,
    limit: int | None,
    resume: bool,
) -> None:
    """Executa e persiste uma requisição por vez."""
    rows, prompt, prompt_hash, metadata = prepare_input(
        csv_path,
        model,
        reasoning_effort,
        include_labeled=include_labeled,
        limit=limit,
    )
    metadata["mode"] = "sync"
    completed = initialize_run_files(
        state_path,
        raw_output_path,
        raw_error_path,
        metadata,
        resume=resume,
    )
    pending = [(index, row) for index, row in rows if index not in completed]
    if completed:
        print(f"Retomando checkpoint com {len(completed)} respostas concluídas.")

    client = create_openai_client() if pending else None
    for position, (row_index, row) in enumerate(pending, start=1):
        custom_id = f"row_{row_index:06d}"
        body = make_response_body(
            row,
            prompt,
            prompt_hash,
            model,
            reasoning_effort,
        )
        try:
            response = client.responses.create(**body)
            response_body = serialize_response(response)
            parse_judgment(response_body)
            record = make_response_record(row_index, response_body)
            append_jsonl(raw_output_path, record)
            print(f"[{position}/{len(pending)}] {custom_id} concluída")
        except Exception as error:  # erros do SDK variam conforme a falha externa
            append_jsonl(
                raw_error_path,
                {
                    "custom_id": custom_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            print(
                f"[{position}/{len(pending)}] {custom_id} falhou: {error}",
                file=sys.stderr,
            )

    merge_results(
        csv_path,
        metadata,
        raw_output_path,
        output_csv_path,
        allow_partial=allow_partial,
    )
