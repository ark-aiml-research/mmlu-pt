"""Interface de linha de comando do LLM-as-a-Judge."""

from __future__ import annotations

import argparse
from pathlib import Path

from .judge_batch import (
    collect_batch,
    default_state_path,
    prepare_requests,
    print_status,
    run_batch,
    submit_batch,
)
from .judge_common import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    JUDGED_SUFFIX,
    RAW_ERRORS_SUFFIX,
    RAW_OUTPUT_SUFFIX,
    REASONING_EFFORTS,
    REQUESTS_SUFFIX,
    STATE_SUFFIX,
    SYNC_STATE_SUFFIX,
    artifact_path,
    load_json,
    metadata_path,
)
from .judge_sync import run_sync

DEFAULT_POLL_SECONDS = 60


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("o valor deve ser maior que zero")
    return parsed


def prepare_command(args: argparse.Namespace) -> None:
    requests_path = args.requests or artifact_path(args.csv, REQUESTS_SUFFIX)
    prepare_requests(
        args.csv,
        requests_path,
        args.model,
        args.reasoning_effort,
        include_labeled=args.include_labeled,
        limit=args.limit,
    )


def submit_command(args: argparse.Namespace) -> None:
    sidecar_path = metadata_path(args.requests)
    metadata = load_json(sidecar_path) if sidecar_path.exists() else None
    state_path = args.state or default_state_path(args.requests, metadata)
    submit_batch(args.requests, state_path, metadata)


def status_command(args: argparse.Namespace) -> None:
    print_status(args.state)


def collect_command(args: argparse.Namespace) -> None:
    csv_path = args.csv
    collect_batch(
        csv_path,
        args.state or artifact_path(csv_path, STATE_SUFFIX),
        args.output or artifact_path(csv_path, JUDGED_SUFFIX),
        args.raw_output or artifact_path(csv_path, RAW_OUTPUT_SUFFIX),
        args.raw_errors or artifact_path(csv_path, RAW_ERRORS_SUFFIX),
        allow_partial=args.allow_partial,
    )


def run_command(args: argparse.Namespace) -> None:
    csv_path = args.csv
    output_path = args.output or artifact_path(csv_path, JUDGED_SUFFIX)
    raw_output_path = args.raw_output or artifact_path(csv_path, RAW_OUTPUT_SUFFIX)
    raw_error_path = args.raw_errors or artifact_path(csv_path, RAW_ERRORS_SUFFIX)

    if args.mode == "sync":
        if args.requests is not None:
            raise ValueError("--requests está disponível apenas no modo batch")
        run_sync(
            csv_path,
            args.state or artifact_path(csv_path, SYNC_STATE_SUFFIX),
            output_path,
            raw_output_path,
            raw_error_path,
            args.model,
            args.reasoning_effort,
            include_labeled=args.include_labeled,
            allow_partial=args.allow_partial,
            limit=args.limit,
            resume=args.resume,
        )
        return

    if args.resume:
        raise ValueError("--resume está disponível apenas no modo sync")
    run_batch(
        csv_path,
        args.requests or artifact_path(csv_path, REQUESTS_SUFFIX),
        args.state or artifact_path(csv_path, STATE_SUFFIX),
        output_path,
        raw_output_path,
        raw_error_path,
        args.model,
        args.reasoning_effort,
        args.poll_seconds,
        include_labeled=args.include_labeled,
        allow_partial=args.allow_partial,
        limit=args.limit,
    )


def add_annotation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument(
        "--include-labeled",
        action="store_true",
        help="Reavalia também linhas que já possuem label.",
    )
    parser.add_argument(
        "--limit",
        type=positive_integer,
        help="Processa somente as primeiras N linhas elegíveis.",
    )


def add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--raw-errors", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Produz o CSV mesmo quando houver respostas inválidas ou ausentes.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser do anotador."""
    parser = argparse.ArgumentParser(
        description=(
            "Anota pares de questões pela Responses API síncrona ou pela Batch API."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Cria o JSONL de requisições Batch a partir do CSV.",
    )
    prepare_parser.add_argument("csv", type=Path)
    prepare_parser.add_argument("--requests", type=Path)
    add_annotation_arguments(prepare_parser)
    prepare_parser.set_defaults(func=prepare_command)

    submit_parser = subparsers.add_parser(
        "submit",
        help="Envia o JSONL e cria o Batch.",
    )
    submit_parser.add_argument("requests", type=Path)
    submit_parser.add_argument("--state", type=Path)
    submit_parser.set_defaults(func=submit_command)

    status_parser = subparsers.add_parser(
        "status",
        help="Consulta o estado atual do Batch.",
    )
    status_parser.add_argument("state", type=Path)
    status_parser.set_defaults(func=status_command)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Baixa os resultados do Batch e os mescla ao CSV.",
    )
    collect_parser.add_argument("csv", type=Path)
    collect_parser.add_argument("state", nargs="?", type=Path)
    add_collection_arguments(collect_parser)
    collect_parser.set_defaults(func=collect_command)

    run_parser = subparsers.add_parser(
        "run",
        help="Executa a anotação no modo sync ou batch.",
    )
    run_parser.add_argument("csv", type=Path)
    run_parser.add_argument(
        "--mode",
        choices=("sync", "batch"),
        default="batch",
        help="Transporte da API (padrão: batch).",
    )
    run_parser.add_argument("--requests", type=Path)
    run_parser.add_argument("--state", type=Path)
    run_parser.add_argument(
        "--poll-seconds",
        type=positive_integer,
        default=DEFAULT_POLL_SECONDS,
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma as respostas persistidas de uma execução sync.",
    )
    add_annotation_arguments(run_parser)
    add_collection_arguments(run_parser)
    run_parser.set_defaults(func=run_command)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0
