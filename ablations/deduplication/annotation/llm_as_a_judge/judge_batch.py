"""Transporte assíncrono baseado na Batch API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .judge_common import (
    create_openai_client,
    ensure_parent,
    load_json,
    make_response_body,
    merge_results,
    metadata_path,
    prepare_input,
    write_json,
)

TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def status_value(status: Any) -> str:
    return str(getattr(status, "value", status))


def default_state_path(
    requests_path: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Deriva o estado do CSV registrado ou do JSONL de requisições."""
    csv_value = (metadata or {}).get("csv")
    if isinstance(csv_value, str) and csv_value:
        csv_path = Path(csv_value)
        return csv_path.with_name(f"{csv_path.stem}.batch-state.json")

    marker = ".judge-requests.jsonl"
    if requests_path.name.endswith(marker):
        stem = requests_path.name[: -len(marker)]
        return requests_path.with_name(f"{stem}.batch-state.json")
    return requests_path.with_suffix(".batch-state.json")


def prepare_requests(
    csv_path: Path,
    requests_path: Path,
    model: str,
    reasoning_effort: str,
    *,
    include_labeled: bool,
    limit: int | None,
) -> dict[str, Any]:
    """Converte as linhas selecionadas em requisições Batch."""
    rows, prompt, prompt_hash, metadata = prepare_input(
        csv_path,
        model,
        reasoning_effort,
        include_labeled=include_labeled,
        limit=limit,
    )

    ensure_parent(requests_path)
    with requests_path.open("w", encoding="utf-8") as output_file:
        for row_index, row in rows:
            request = {
                "custom_id": f"row_{row_index:06d}",
                "method": "POST",
                "url": "/v1/responses",
                "body": make_response_body(
                    row,
                    prompt,
                    prompt_hash,
                    model,
                    reasoning_effort,
                ),
            }
            output_file.write(json.dumps(request, ensure_ascii=False) + "\n")

    metadata["mode"] = "batch"
    metadata["requests_file"] = str(requests_path.resolve())
    sidecar_path = metadata_path(requests_path)
    write_json(sidecar_path, metadata)
    print(f"Preparadas {len(rows)} requisições -> {requests_path}")
    print(f"Metadados -> {sidecar_path}")
    return metadata


def submit_batch(
    requests_path: Path,
    state_path: Path,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Envia o JSONL e cria um Batch para o endpoint Responses."""
    if not requests_path.is_file():
        raise FileNotFoundError(f"Arquivo de requisições não encontrado: {requests_path}")

    client = create_openai_client()
    with requests_path.open("rb") as requests_file:
        uploaded = client.files.create(file=requests_file, purpose="batch")

    batch_metadata = {
        "description": "Question-pair semantic deduplication judge",
    }
    prompt_hash = (metadata or {}).get("prompt_sha256")
    if isinstance(prompt_hash, str) and prompt_hash:
        batch_metadata["prompt_sha256"] = prompt_hash

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata=batch_metadata,
    )

    state = dict(metadata or {})
    state.update(
        {
            "batch_id": batch.id,
            "input_file_id": uploaded.id,
            "requests_file": str(requests_path.resolve()),
            "status": status_value(batch.status),
        }
    )
    write_json(state_path, state)
    print(f"Arquivo enviado: {uploaded.id}")
    print(f"Batch criado:    {batch.id}")
    print(f"Status inicial:  {status_value(batch.status)}")
    print(f"Estado salvo:    {state_path}")
    return state


def get_batch(state_path: Path) -> tuple[Any, dict[str, Any], Any]:
    client = create_openai_client()
    state = load_json(state_path)
    batch_id = state.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError(f"O estado não contém batch_id: {state_path}")
    return client, state, client.batches.retrieve(batch_id)


def print_status(state_path: Path) -> str:
    """Consulta e exibe o progresso de um Batch."""
    _client, _state, batch = get_batch(state_path)
    status = status_value(batch.status)
    counts = batch.request_counts
    if counts is None:
        print(f"batch={batch.id} status={status}")
        return status

    print(
        f"batch={batch.id} status={status} "
        f"completed={getattr(counts, 'completed', None)}/"
        f"{getattr(counts, 'total', None)} "
        f"failed={getattr(counts, 'failed', None)}"
    )
    return status


def download_batch_files(
    client: Any,
    batch: Any,
    raw_output_path: Path,
    raw_error_path: Path,
) -> None:
    """Baixa as respostas e os erros produzidos pela API."""
    if not batch.output_file_id and not batch.error_file_id:
        raise RuntimeError(
            f"O Batch terminou com status={status_value(batch.status)}, mas não "
            "possui arquivos de saída ou de erros."
        )

    ensure_parent(raw_output_path)
    if batch.output_file_id:
        content = client.files.content(batch.output_file_id)
        raw_output_path.write_text(content.text, encoding="utf-8")
        print(f"Respostas brutas -> {raw_output_path}")
    else:
        raw_output_path.write_text("", encoding="utf-8")
        print(f"O Batch não produziu respostas -> {raw_output_path}")

    if batch.error_file_id:
        content = client.files.content(batch.error_file_id)
        ensure_parent(raw_error_path)
        raw_error_path.write_text(content.text, encoding="utf-8")
        print(f"Erros da API -> {raw_error_path}")
    else:
        ensure_parent(raw_error_path)
        raw_error_path.write_text("", encoding="utf-8")


def collect_batch(
    csv_path: Path,
    state_path: Path,
    output_csv_path: Path,
    raw_output_path: Path,
    raw_error_path: Path,
    *,
    allow_partial: bool,
) -> None:
    """Baixa e mescla os resultados de um Batch concluído."""
    client, state, batch = get_batch(state_path)
    status = status_value(batch.status)
    if status not in TERMINAL_STATUSES:
        raise RuntimeError(
            f"O Batch ainda não terminou (status={status}). Consulte `status` mais "
            "tarde ou use `run --mode batch` para aguardar automaticamente."
        )

    download_batch_files(client, batch, raw_output_path, raw_error_path)
    merge_results(
        csv_path,
        state,
        raw_output_path,
        output_csv_path,
        allow_partial=allow_partial,
    )


def run_batch(
    csv_path: Path,
    requests_path: Path,
    state_path: Path,
    output_csv_path: Path,
    raw_output_path: Path,
    raw_error_path: Path,
    model: str,
    reasoning_effort: str,
    poll_seconds: int,
    *,
    include_labeled: bool,
    allow_partial: bool,
    limit: int | None,
) -> None:
    """Executa o ciclo completo da Batch API."""
    metadata = prepare_requests(
        csv_path,
        requests_path,
        model,
        reasoning_effort,
        include_labeled=include_labeled,
        limit=limit,
    )
    submit_batch(requests_path, state_path, metadata)

    while True:
        status = print_status(state_path)
        if status in TERMINAL_STATUSES:
            break
        time.sleep(poll_seconds)

    collect_batch(
        csv_path,
        state_path,
        output_csv_path,
        raw_output_path,
        raw_error_path,
        allow_partial=allow_partial,
    )
