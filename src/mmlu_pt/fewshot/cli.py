"""Command-line interface: select demonstrations, generate rationales, build and publish the splits."""

import argparse
import importlib.metadata
import json
import os
import sys
from dataclasses import asdict, fields
from pathlib import Path

from . import audit, publish, rationale
from .config import (DEFAULT_OUTPUT_DIR, PROMPT_VERSION, REASONING_EFFORTS, RunConfig, letter, level_repo,
                     read_jsonl, run_id, slug, timestamp, write_json)
from .selection import build_strata, build_text_index, load_source, row_ids, validate_columns

OPTIONAL_TYPES = {"source_revision": str, "rationale_limit": int}


def environment_default(name: str, default):
    raw = os.environ.get("MMLU_FEWSHOT_" + name.upper())
    if raw is None:
        return default
    if isinstance(default, bool):
        if raw.lower() not in ("true", "false", "1", "0"):
            raise ValueError(f"Invalid boolean environment value for {name}")
        return raw.lower() in ("true", "1")
    return OPTIONAL_TYPES.get(name, type(default))(raw)


def add_config_options(parser: argparse.ArgumentParser) -> None:
    defaults = RunConfig()
    for field in fields(defaults):
        name, default = field.name, getattr(defaults, field.name)
        kwargs = {"default": environment_default(name, default), "help": f"default: {default}"}
        if isinstance(default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = OPTIONAL_TYPES.get(name, type(default))
        if name == "reasoning_effort":
            kwargs["choices"] = REASONING_EFFORTS
        parser.add_argument("--" + name.replace("_", "-"), **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Few-shot demonstrations with rationales and per-area splits")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Select, audit, generate rationales, build and publish")
    add_config_options(run)
    run.add_argument("--run-dir", type=Path)
    verify = commands.add_parser("verify", help="Reload the published datasets and check them against the manifest")
    verify.add_argument("--run-dir", type=Path, required=True)
    return parser


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def prepare_manifest(run_dir: Path, run: str, config: RunConfig, source: dict, prompt_digest: str) -> dict:
    path = run_dir / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["run_id"] != run:
            raise ValueError(f"{run_dir} belongs to run {existing['run_id']}; choose a new --run-dir")
        return existing
    manifest = {"run_id": run, "created_at": timestamp(), "status": "started", "config": asdict(config),
                "source": source, "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_digest,
                "developer_prompt": rationale.DEVELOPER_PROMPT,
                "dependencies": {name: package_version(name) for name in ("datasets", "huggingface-hub", "openai")}}
    write_json(path, manifest)
    return manifest


def candidate_entry(data, ids: list[str], index: int, rank_position: int) -> dict:
    row = data[index]
    answer = row["answer"]
    valid = isinstance(answer, int) and 0 <= answer < len(row["choices"] or [])
    return {"id": ids[index], "row_index": index, "rank_position": rank_position, "exam": row["exam"],
            "exam_edition": row["exam_edition"], "num": row["num"], "answer_letter": letter(answer) if valid else None}


def calls_allowed(budget: dict) -> bool:
    return budget["limit"] is None or budget["used"] < budget["limit"]


def generate_rationale(entry: dict, row: dict, records: list[dict], run_dir: Path, config: RunConfig,
                       prompt_digest: str, budget: dict) -> tuple[dict | None, list[str]]:
    """Up to rationale_attempts generations; returns (accepted record, reasons of the last rejection)."""
    reasons = []
    for attempt in range(1, config.rationale_attempts + 1):
        record, called = rationale.obtain(row, entry["id"], entry["answer_letter"], attempt, records, run_dir,
                                          config, prompt_digest, calls_allowed(budget))
        budget["used"] += called
        if record is None:
            if config.dry_run:
                return None, []
            raise ValueError("rationale call limit exhausted; raise --rationale-limit")
        reasons = audit.rationale_checks(record, entry["answer_letter"], len(row["choices"]), config)
        if not reasons:
            return record, []
    return None, reasons


def fill_stratum(level: str, area: str, ranking: list[int], data, ids: list[str], text_index: dict,
                 records: list[dict], run_dir: Path, config: RunConfig, prompt_digest: str, budget: dict) -> dict:
    """Walk the deterministic ranking until `shots` candidates pass every check."""
    accepted, excluded, cursor = [], [], 0
    while len(accepted) < config.shots:
        # Only rationale rejections cost API calls, so only they count against max_replacements.
        rationale_rejections = sum(entry["stage"] == "rationale" for entry in excluded)
        if cursor == len(ranking) or rationale_rejections > config.max_replacements:
            raise ValueError(f"{level}/{area}: could not fill {config.shots} demonstrations "
                             f"({cursor} candidates considered, {len(excluded)} excluded)")
        index = ranking[cursor]
        cursor += 1
        entry = candidate_entry(data, ids, index, cursor)
        reasons = audit.pre_rationale_checks(data, ids, index, text_index, config)
        if reasons:
            excluded.append(entry | {"stage": "pre_rationale", "reasons": reasons})
            continue
        record, reasons = generate_rationale(entry, data[index], records, run_dir, config, prompt_digest, budget)
        if reasons:
            excluded.append(entry | {"stage": "rationale", "reasons": reasons})
            continue
        accepted.append(entry | {"status": "ok" if record else "rationale_skipped",
                                 "rationale": record["rationale"] if record else None})
    print(f"{level}/{area}: {len(accepted)} accepted, {len(excluded)} excluded, {cursor} considered")
    return {"level": level, "macro_area": area, "config_name": slug(area), "stratum_rows": len(ranking),
            "ranking_considered": cursor, "dev": accepted, "excluded": excluded}


def run(config: RunConfig, run_dir_argument: Path | None) -> int:
    data, source = load_source(config)
    problems = validate_columns(data)
    if problems:
        print("Invalid source: " + "; ".join(problems))
        return 2
    ids = row_ids(data)
    prompt_digest = rationale.prompt_hash(config)
    run = run_id(config, source, prompt_digest)
    run_dir = run_dir_argument or DEFAULT_OUTPUT_DIR / (("dry-run-" if config.dry_run else "run-") + run)
    manifest = prepare_manifest(run_dir, run, config, source, prompt_digest)
    print(f"run {run} in {run_dir} ({source['identifier']}@{source['revision'][:12]}, {len(data)} rows)")

    strata = build_strata(data, ids, config.seed)
    manifest["strata"] = {f"{level}/{area}": len(ranking) for (level, area), ranking in strata.items()}
    text_index = build_text_index(data)
    records = read_jsonl(run_dir / rationale.RATIONALE_FILE)
    budget = {"limit": config.call_limit(), "used": 0}
    selection = [fill_stratum(level, area, ranking, data, ids, text_index, records, run_dir, config, prompt_digest, budget)
                 for (level, area), ranking in strata.items()]
    write_json(run_dir / "selection.json", selection)

    dev_ids = {entry["id"] for stratum in selection for entry in stratum["dev"]}
    built = {}
    for stratum in selection:
        ranking = strata[(stratum["level"], stratum["macro_area"])]
        built.setdefault(stratum["level"], {})[stratum["config_name"]] = publish.build_config(
            data, ids, ranking, stratum["dev"], dev_ids)
    final = audit.final_checks(built, len(data), config.shots)
    audit.write_reports(run_dir, selection, final)
    publish.save_configs(run_dir, built)
    manifest |= {"summary": publish.summarize(built), "api_calls": budget["used"], "status": "built"}
    write_json(run_dir / "manifest.json", manifest)
    print(f"dev rows: {final['dev_rows']}, test rows: {final['test_rows']}, API calls: {budget['used']}")
    if final["problems"]:
        print("Audit problems: " + "; ".join(final["problems"]))
        return 2
    if not config.push or config.dry_run:
        return 0

    for level, configs in built.items():
        repo = level_repo(config, level)
        body = publish.card_body(level, repo, manifest["summary"][level], manifest)
        (run_dir / f"README-{level}.md").write_text(body, encoding="utf-8")
        manifest.setdefault("published", {})[repo] = publish.push_level(repo, configs, body, config, run)
        write_json(run_dir / "manifest.json", manifest)
    manifest["status"] = "published"
    write_json(run_dir / "manifest.json", manifest)
    return 0


def verify(run_dir: Path) -> int:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = RunConfig(**manifest["config"])
    problems, dev_ids, test_ids = [], set(), set()
    for level, summary in manifest["summary"].items():
        result = publish.verify_published(level_repo(config, level), summary, config.shots)
        print(f"{result['repo']}: configs {result['configs']}, {len(result['problems'])} problems")
        problems += [f"{result['repo']}: {problem}" for problem in result["problems"]]
        dev_ids |= result["dev_ids"]
        test_ids |= result["test_ids"]
    if dev_ids & test_ids:
        problems.append(f"{len(dev_ids & test_ids)} ids appear in both dev and test across repos")
    if len(dev_ids) + len(test_ids) != manifest["source"]["rows"]:
        problems.append(f"dev + test = {len(dev_ids) + len(test_ids)}, source has {manifest['source']['rows']}")
    print(f"dev ids: {len(dev_ids)}, test ids: {len(test_ids)}, source rows: {manifest['source']['rows']}")
    for problem in problems:
        print("PROBLEM: " + problem)
    return 2 if problems else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    try:
        if args.command == "verify":
            return verify(args.run_dir)
        config = RunConfig(**{field.name: getattr(args, field.name) for field in fields(RunConfig)})
        return run(config, args.run_dir)
    except KeyboardInterrupt:
        print("Interrupted. Repeat the same command to resume.")
        return 130
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"Error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
