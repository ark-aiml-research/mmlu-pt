"""Command-line interface for offline annotation and model-free inspection/reports."""

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

from .checkpoint import run_lock
from .config import DEFAULT_ALIASES, DEFAULT_OUTPUT_DIR, DEFAULT_TAXONOMY, RunConfig, digest, write_json
from .dataset import inspect_dataset, load_source
from .pipeline import AnnotationTerminated, annotate, print_validation, read_run
from .reporting import export_review, generate_report
from .inference import InferenceError, OfflineConfig
from .taxonomy import load_taxonomy
from .subsets import DEFAULT_REVIEW_NOTES, compare_subset, prepare_subset
from .post_annotation import DEFAULT_MANUAL, DEFAULT_REMOVALS, finalize


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


def add_offline_options(parser: argparse.ArgumentParser) -> None:
    for field in fields(OfflineConfig):
        name, default = field.name, field.default
        kwargs = {"dest": "offline_" + name, "default": environment_default(name, default),
                  "help": f"offline vLLM; default: {default}"}
        if isinstance(default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type(default)
        parser.add_argument("--" + name.replace("_", "-"), **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible answer-free academic subject annotation")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect schema and taxonomy mappings without inference")
    add_config_options(inspect)
    inspect.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "inspection")
    run = commands.add_parser("annotate", help="Annotate with local offline vLLM engines")
    add_config_options(run)
    add_offline_options(run)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--force", action="store_true")
    run.add_argument("--push-to-hub")
    subset = commands.add_parser("prepare-subset", help="Save final UNCERTAIN rows as a local dataset; no inference")
    subset.add_argument("--source-run", type=Path, required=True)
    subset.add_argument("--output-dir", type=Path, required=True)
    compare = commands.add_parser("compare-subset", help="Compare an abstention experiment with its baseline")
    compare.add_argument("--source-run", type=Path, required=True)
    compare.add_argument("--run-dir", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--review-notes", type=Path, default=DEFAULT_REVIEW_NOTES)
    final = commands.add_parser("finalize", help="Apply manual assignments and dataset removals to a published export")
    final.add_argument("--run-dir", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, help="default: <run-dir>/post-annotation")
    final.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    final.add_argument("--removals", type=Path, default=DEFAULT_REMOVALS)
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
    arguments = list(argv) if argv is not None else sys.argv[1:]
    removed = {"--endpoint", "--api-key", "--server-mode", "--server-manifest", "--concurrency", "--timeout",
               "--backoff-initial", "--backoff-max", "--server-startup-timeout", "--server-shutdown-timeout",
               "--server-poll-interval"}
    obsolete = sorted({arg.split("=", 1)[0] for arg in arguments} & removed)
    if obsolete:
        parser.error(f"Removed HTTP options: {', '.join(obsolete)}. Offline annotation uses --max-num-seqs "
                     "and --batch-size; lifecycle uses --startup-timeout, --shutdown-timeout and optional "
                     "--batch-timeout. Choose a new --run-dir for offline runs.")
    args = parser.parse_args(arguments)
    try:
        if args.command == "prepare-subset":
            result = prepare_subset(args.source_run, args.output_dir)
            print(f"Prepared {result['rows']} UNCERTAIN rows: {args.output_dir / 'dataset'}")
            return 0
        if args.command == "compare-subset":
            result = compare_subset(args.source_run, args.run_dir, args.output_dir, args.review_notes)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["valid"] and not result["counts"]["error"] else 2
        if args.command == "finalize":
            result = finalize(args.run_dir, args.output_dir or args.run_dir / "post-annotation", args.manual, args.removals)
            print(f"Final dataset: {result['output_rows']} rows; {len(result['manual_assignments'])} manual assignments; "
                  f"{len(result['removed'])} removed; {len(result['remaining_uncertain_row_indices'])} UNCERTAIN left")
            return 0
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
        settings = OfflineConfig(**{field.name: getattr(args, "offline_" + field.name)
                                    for field in fields(OfflineConfig)})
        run_dir = args.run_dir or DEFAULT_OUTPUT_DIR / (
            ("dry-run-offline-" if config.dry_run else "run-offline-")
            + digest({"methodology": config.methodology(), "offline": settings.identity()})[:12])
        report = annotate(config, run_dir, args.taxonomy, args.exam_aliases,
                          args.force, args.push_to_hub, offline_config=settings)
        return 2 if report["errors"] else 0
    except AnnotationTerminated:
        print("Terminated. Repeat the same annotate command to resume.")
        return 143
    except KeyboardInterrupt:
        print("Interrupted. Repeat the same annotate command to resume.")
        return 130
    except Exception as exc:
        # Exception groups and third-party exceptions may contain model output; never print them.
        if isinstance(exc, (ValueError, FileNotFoundError, InferenceError)):
            print(f"Error: {exc}")
        else:
            print(f"Execution failed ({type(exc).__name__}); inspect the manifest and sanitized checkpoint errors.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
