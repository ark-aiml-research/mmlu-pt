"""Command-line interface; all server access occurs only through annotate."""

import argparse
import asyncio
import json
import os
from dataclasses import fields
from pathlib import Path

from .checkpoint import run_lock
from .config import DEFAULT_ALIASES, DEFAULT_OUTPUT_DIR, DEFAULT_TAXONOMY, RunConfig, digest, write_json
from .dataset import inspect_dataset, load_source
from .pipeline import annotate, print_validation, read_run
from .reporting import export_review, generate_report
from .server import ServerConfig
from .taxonomy import load_taxonomy


def environment_default(name: str, default):
    raw = os.environ.get("MMLU_ANNOTATION_" + name.upper())
    if raw is None:
        return default
    if isinstance(default, bool):
        if raw.lower() not in ("true", "false", "1", "0"):
            raise ValueError(f"Invalid boolean environment value for {name}")
        return raw.lower() in ("true", "1")
    return type(default)(raw) if default is not None else raw


def add_config_options(parser: argparse.ArgumentParser) -> None:
    defaults = RunConfig()
    for field in fields(defaults):
        name, default = field.name, getattr(defaults, field.name)
        kwargs = {"default": environment_default(name, default), "help": f"default: {default}"}
        if isinstance(default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type(default) if default is not None else str
        if name == "num_independent_passes":
            kwargs["choices"] = (1, 2)
        if name == "structured_output":
            kwargs["choices"] = ("json_schema", "none")
        parser.add_argument("--" + name.replace("_", "-"), **kwargs)
    parser.add_argument("--taxonomy", type=Path, default=Path(os.environ.get("MMLU_ANNOTATION_TAXONOMY", DEFAULT_TAXONOMY)))
    parser.add_argument("--exam-aliases", type=Path, default=Path(os.environ.get("MMLU_ANNOTATION_EXAM_ALIASES", DEFAULT_ALIASES)))


def add_server_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server-mode", choices=("managed", "external"),
                        default=environment_default("server_mode", "managed"),
                        help="managed starts/stops vLLM within this Python job (default)")
    for field in fields(ServerConfig):
        name, default = field.name, field.default
        option = "server_" + name if name in ("startup_timeout", "shutdown_timeout", "poll_interval") else name
        kwargs = {"dest": "server_" + name, "default": environment_default("server_" + name, default),
                  "help": f"managed vLLM; default: {default}"}
        if isinstance(default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type(default)
        parser.add_argument("--" + option.replace("_", "-"), **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible answer-free academic subject annotation")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect schema and taxonomy mappings without inference")
    add_config_options(inspect)
    inspect.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "inspection")
    run = commands.add_parser("annotate", help="Start vLLM and annotate in one Python job")
    add_config_options(run)
    add_server_options(run)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--force", action="store_true")
    run.add_argument("--server-manifest", type=Path)
    run.add_argument("--push-to-hub")
    for name in ("report", "export-review"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path)
        if name == "export-review":
            command.add_argument("--confidence-below", type=float)
            command.add_argument("--include-disagreements", action="store_true")
            command.add_argument("--include-answer", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in ("report", "export-review"):
            with run_lock(args.run_dir):
                data, audits, manifest, generation = read_run(args.run_dir)
                if args.command == "report":
                    validation = json.loads((args.run_dir / "validation.json").read_text(encoding="utf-8"))
                    generate_report(data, audits, validation, args.output_dir or args.run_dir / "reports", manifest["run_id"], generation)
                else:
                    count = export_review(data, args.output_dir or args.run_dir / "review", args.confidence_below,
                                          args.include_disagreements, args.include_answer)
                    print(f"Exported {count} human-review cases")
            return 0
        config = RunConfig(**{field.name: getattr(args, field.name) for field in fields(RunConfig)})
        if args.command == "inspect":
            taxonomy = load_taxonomy(args.taxonomy, args.exam_aliases)
            data, source = load_source(config)
            validation = inspect_dataset(data, taxonomy)
            print(json.dumps(source, ensure_ascii=False, indent=2))
            print_validation(validation)
            write_json(args.output_dir / "inspection.json", {"source": source, "taxonomy_version": taxonomy.version,
                       "taxonomy_hash": taxonomy.checksum, "validation": validation})
            return 0 if validation["valid"] else 2
        run_dir = args.run_dir or DEFAULT_OUTPUT_DIR / (
            ("dry-run-" if config.dry_run else "run-") + digest(config.methodology())[:12])
        if args.server_mode not in ("managed", "external"):
            raise ValueError("server_mode must be managed or external")
        server_config = ServerConfig(**{field.name: getattr(args, "server_" + field.name)
                                        for field in fields(ServerConfig)}) if args.server_mode == "managed" else None
        report = asyncio.run(annotate(config, run_dir, args.taxonomy, args.exam_aliases,
                                      args.force, args.push_to_hub, args.server_manifest,
                                      server_config=server_config))
        return 2 if report["errors"] else 0
    except KeyboardInterrupt:
        print("Interrupted. Repeat the same annotate command to resume.")
        return 130
    except asyncio.CancelledError:
        print("Terminated. Repeat the same annotate command to resume.")
        return 143
    except Exception as exc:
        # Exception groups and third-party exceptions may contain model output; never print them.
        if isinstance(exc, (ValueError, FileNotFoundError)):
            print(f"Error: {exc}")
        else:
            print(f"Execution failed ({type(exc).__name__}); inspect the manifest and sanitized checkpoint errors.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
