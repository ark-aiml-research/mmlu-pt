"""Preflight, audit manifests and crash-safe offline annotation orchestration."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
from dataclasses import asdict, fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from itertools import islice
from threading import current_thread, main_thread

from datasets import Dataset, load_from_disk
from tqdm.auto import tqdm

from .checkpoint import Checkpoint, run_lock
from .config import PACKAGE_DIR, RunConfig, canonical_json, digest, request_seed, timestamp, write_json
from .dataset import annotation_id, inspect_dataset, load_source, metadata, sample_indices
from .export import ANNOTATION_FEATURES, build_annotated, needs_adjudication, publish_exports, push_dataset
from .prompts import QuestionInput, adjudication_prompt, classification_prompt, prompt_identity, question_input
from .reporting import generate_report
from .inference import OfflineConfig, OfflineWorkers, prepare_inference
from .taxonomy import Taxonomy, load_taxonomy


def versions() -> dict:
    values = {}
    for name in ("datasets", "huggingface-hub", "pydantic", "vllm", "torch", "transformers", "pyarrow", "tqdm"):
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = None
    return values


def code_metadata() -> dict:
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(PACKAGE_DIR.glob("*.py"))}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PACKAGE_DIR,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PACKAGE_DIR,
                                            stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty, "file_hashes": hashes}


def source_digest(data: Dataset) -> str:
    """Audit source preservation, including the answer; never used in model inputs."""
    hash_value = hashlib.sha256()
    for row in data:
        hash_value.update((canonical_json(row) + "\n").encode("utf-8"))
    return hash_value.hexdigest()


def print_validation(report: dict) -> None:
    print(f"Dataset validation: {report['total_rows']} rows; valid={report['valid']}")
    print(f"{'Exam':<30} {'Rows':>7}  {'Taxonomy entry':<32} Candidates")
    for exam in report["exams"]:
        print(f"{exam['exam']:<30} {exam['rows']:>7}  {str(exam['taxonomy_exam']):<32} {exam['candidate_counts']}")
    if report.get("missing_fields"):
        print("Missing fields:", report["missing_fields"])
    if report["unmapped_exams"]:
        print("UNMAPPED EXAMS:", ", ".join(report["unmapped_exams"]))
    if report["invalid_rows"]:
        print(f"Invalid rows: {len(report['invalid_rows'])}; see validation.json for every row")


def initialize_run(run_dir: Path, data: Dataset, source: dict, taxonomy: Taxonomy,
                   config: RunConfig, validation: dict, indices: list[int], inference_metadata: dict | None) -> dict:
    code = code_metadata()
    identity = {"source": source, "methodology": config.methodology(), "taxonomy_hash": taxonomy.checksum,
                "aliases_hash": digest(taxonomy.aliases), "code_hashes": code["file_hashes"],
                "selected_indices_hash": digest(indices),
                "inference_metadata": {k: v for k, v in inference_metadata.items() if k != "created_at"} if inference_metadata else None}
    fingerprint = digest(identity)
    path = run_dir / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["identity_hash"] != fingerprint:
            raise ValueError("Run configuration/data/taxonomy/code changed; choose a new --run-dir")
        snapshot = load_from_disk(str(run_dir / "source"))
        if source_digest(snapshot) != manifest["selected_source_hash"]:
            raise ValueError("Saved source snapshot failed its integrity check")
        return manifest
    selected = data.select(indices)
    overlap = set(selected.column_names) & ANNOTATION_FEATURES.keys()
    if overlap:
        raise ValueError(f"Source has reserved annotation columns: {sorted(overlap)}")
    temporary = run_dir / ".source-staging"
    if temporary.exists():
        shutil.rmtree(temporary)
    selected.save_to_disk(str(temporary))
    snapshot_dir = run_dir / "source"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    os.replace(temporary, snapshot_dir)
    write_json(run_dir / "taxonomy.json", taxonomy.document)
    write_json(run_dir / "exam_aliases.json", taxonomy.aliases)
    write_json(run_dir / "validation.json", validation)
    manifest = {"format_version": 2, "run_id": fingerprint[:16], "identity_hash": fingerprint,
                "identity": identity, "created_at": timestamp(), "config": asdict(config),
                "source": source, "selected_rows": len(indices), "selected_indices": indices,
                "selected_source_hash": source_digest(selected),
                "taxonomy_version": taxonomy.version, "prompt": prompt_identity(), "code": code,
                "dependencies": versions(), "model_revision": (inference_metadata or {}).get("revision") or config.model_revision,
                "model_revision_status": "resolved_offline" if inference_metadata else "declared" if config.model_revision else "unknown",
                "inference_metadata": inference_metadata, "sessions": []}
    write_json(path, manifest)
    return manifest


def checkpoint_rows(source: Dataset, indices: list[int], config: RunConfig, taxonomy: Taxonomy) -> list[dict]:
    rows = []
    for position, (row_index, row) in enumerate(zip(indices, source.remove_columns("answer"), strict=True)):
        rows.append({"position": position, "row_index": row_index, "annotation_id": annotation_id(row),
                     "input": asdict(question_input(row, config.include_exam_edition)),
                     "candidates": taxonomy.resolve_candidates(row["exam"], metadata(row))})
    return rows


def task_messages(task: dict, checkpoint: Checkpoint, config: RunConfig, taxonomy: Taxonomy) -> list[dict]:
    question = QuestionInput(**json.loads(task["input_json"]))
    candidates = json.loads(task["candidates_json"])
    if task["kind"] == "adjudication":
        proposals = [t["result"] for t in checkpoint.tasks(task["annotation_id"]) if t["kind"] == "independent"]
        return adjudication_prompt(question, candidates, taxonomy, proposals)
    return classification_prompt(question, candidates, taxonomy)


def run_stage(checkpoint: Checkpoint, config: RunConfig, taxonomy: Taxonomy, workers,
              kind: str, pass_number: int, inference_metadata: dict, show_progress: bool = True) -> None:
    pending = checkpoint.pending(kind, pass_number)
    count = checkpoint.connection.execute(
        "SELECT COUNT(*), SUM(status='success') FROM tasks WHERE generation=? AND kind=? AND pass_number=?",
        (checkpoint.generation, kind, pass_number)).fetchone()
    progress = tqdm(total=count[0], initial=count[1] or 0, desc=f"{kind} {pass_number}", disable=not show_progress)
    examples = 0
    capacity = workers.settings.batch_size * workers.settings.data_parallel_size
    try:
        while batch := list(islice(pending, capacity)):
            tasks = {task["annotation_id"]: dict(task) for task in batch}
            local_attempts = {identity: 0 for identity in tasks}
            retry_tasks = list(tasks.values())
            while retry_tasks:
                items = []
                for task in retry_tasks:
                    identity = task["annotation_id"]
                    local_attempts[identity] += 1
                    attempt = checkpoint.mark_running(task) + 1
                    seed = request_seed(config.seed, identity, kind, pass_number, local_attempts[identity])
                    checkpoint.start_attempt(task, attempt, seed, {
                        "backend": "vllm_offline", "requested_model": config.model,
                        "model_revision": inference_metadata["revision"], "generation": config.generation(),
                        "thinking": config.thinking, "settings": workers.settings.identity()})
                    items.append({"annotation_id": identity, "attempt": attempt, "seed": seed,
                                  "messages": task_messages(task, checkpoint, config, taxonomy),
                                  "candidates": json.loads(task["candidates_json"])})
                retry_tasks = []
                for event in workers.generate(items):
                    for outcome in event["results"]:
                        identity = outcome["annotation_id"]
                        task = tasks[identity]
                        outcome["metadata"].update({"dp_rank": event["rank"], "round": event["round"],
                                                   "batch_id": f"{workers.state['started_at']}:{event['round']}:{event['rank']}",
                                                   "batch_duration_seconds": event["batch_duration_seconds"],
                                                   "batch_completed_at": event["batch_completed_at"],
                                                   "duration_scope": "batch_only"})
                        checkpoint.persist_attempt(task, outcome | {"duration_seconds": None})
                        retry = (outcome["error"] is not None and outcome["error"] != "context_overflow"
                                 and local_attempts[identity] < config.max_attempts
                                 and local_attempts[identity] <= config.format_retries)
                        if retry:
                            retry_tasks.append(task)
                            continue
                        if outcome["error"]:
                            checkpoint.finish_error(task, outcome["error"])
                        if config.dry_run and examples < 2:
                            print("EXAMPLE PROMPT:", canonical_json(task_messages(task, checkpoint, config, taxonomy)))
                            print("EXAMPLE RESULT:", canonical_json(outcome["result"] or {"error": outcome["error"]}))
                            examples += 1
                        progress.update(1)
    finally:
        pending.close()
        progress.close()


def prepare_adjudications(checkpoint: Checkpoint, config: RunConfig) -> None:
    if not config.adjudicate_disagreements or config.num_independent_passes == 1:
        return
    for identity in checkpoint.question_ids():
        independent = [t for t in checkpoint.tasks(identity) if t["kind"] == "independent"]
        if all(t["status"] == "success" for t in independent) and needs_adjudication([t["result"] for t in independent]):
            checkpoint.ensure_adjudication(identity)


def requires_inference(checkpoint: Checkpoint, config: RunConfig) -> bool:
    prepare_adjudications(checkpoint, config)
    return checkpoint.connection.execute(
        "SELECT 1 FROM tasks WHERE generation=? AND status!='success' LIMIT 1",
        (checkpoint.generation,)).fetchone() is not None


def execute_stages(checkpoint: Checkpoint, config: RunConfig, taxonomy: Taxonomy,
                   workers, inference_metadata: dict, show_progress: bool = True) -> None:
    for number in range(1, config.num_independent_passes + 1):
        run_stage(checkpoint, config, taxonomy, workers, "independent", number, inference_metadata, show_progress)
    prepare_adjudications(checkpoint, config)
    if config.adjudicate_disagreements and config.num_independent_passes > 1:
        run_stage(checkpoint, config, taxonomy, workers, "adjudication", 0, inference_metadata, show_progress)


class AnnotationTerminated(KeyboardInterrupt):
    """A SIGTERM interrupted the job after persisting completed batches."""


def terminated(signum, frame) -> None:
    raise AnnotationTerminated()


def annotate(config: RunConfig, run_dir: Path, taxonomy_path: Path, aliases_path: Path,
             force: bool = False, push_to_hub: str | None = None, show_progress: bool = True,
             offline_config: OfflineConfig | None = None) -> dict:
    settings = offline_config or OfflineConfig()
    if push_to_hub and (config.dry_run or push_to_hub.strip().rstrip("/") == config.dataset.strip().rstrip("/")):
        raise ValueError("Refusing dry-run publishing or overwriting the source dataset")
    with run_lock(run_dir):
        manifest_path = run_dir / "manifest.json"
        previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        if previous and (previous.get("inference_metadata") or {}).get("backend") != "vllm_offline":
            raise ValueError("Existing run used the HTTP backend; choose a new --run-dir for offline annotation")
        if previous and config.dataset_revision and config.dataset_revision != previous["source"]["revision"]:
            raise ValueError("Dataset revision differs from the existing run")
        taxonomy = load_taxonomy(taxonomy_path, aliases_path)
        if previous:
            data = load_from_disk(str(run_dir / "source"))
            source_info = previous["source"]
            validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
            indices = previous["selected_indices"]
        else:
            data, source_info = load_source(config)
            validation = inspect_dataset(data, taxonomy)
            indices = sample_indices(data, config) if validation["valid"] else []
        print_validation(validation)
        if not validation["valid"]:
            write_json(run_dir / "validation-failed.json", validation)
            raise ValueError(f"Preflight failed; unmapped exams: {validation['unmapped_exams']}; see validation-failed.json")
        inference_metadata = prepare_inference(config, settings, previous)
        manifest = initialize_run(run_dir, data, source_info, taxonomy, config, validation, indices, inference_metadata)
        source = load_from_disk(str(run_dir / "source"))
        checkpoint = Checkpoint(run_dir / "checkpoint.sqlite3")
        session = {"started_at": timestamp(), "config": asdict(config), "dependencies": versions(),
                   "inference_metadata": inference_metadata, "offline_settings": asdict(settings), "status": "running"}
        install_handler = current_thread() is main_thread()
        old_handler = signal.signal(signal.SIGTERM, terminated) if install_handler else None
        try:
            if force:
                checkpoint.new_generation()
            checkpoint.prepare(checkpoint_rows(source, indices, config, taxonomy), config.num_independent_passes)
            session["generation"] = checkpoint.generation
            manifest["sessions"].append(session)
            write_json(manifest_path, manifest)
            if requires_inference(checkpoint, config):
                with OfflineWorkers(config, settings, inference_metadata, run_dir) as workers:
                    try:
                        execute_stages(checkpoint, config, taxonomy, workers, inference_metadata, show_progress)
                    finally:
                        session["worker_lifecycle"] = workers.state
            annotated, audits = build_annotated(source, checkpoint, config, taxonomy, manifest["run_id"])
            destination = publish_exports(annotated, source, run_dir, checkpoint.generation)
            report = generate_report(annotated, audits, validation, run_dir / "reports",
                                     manifest["run_id"], checkpoint.generation)
            session["status"] = "completed_with_errors" if report["errors"] else "completed"
            session["export_directory"] = str(destination.relative_to(run_dir))
            if push_to_hub:
                push_dataset(annotated, push_to_hub, config)
                session["pushed_to_hub"] = push_to_hub
            print(f"Exported {len(annotated)} rows to {destination}; errors={report['errors']}")
            return report
        except KeyboardInterrupt:
            session["status"] = "interrupted"
            raise
        except BaseException:
            session["status"] = "failed"
            raise
        finally:
            session["finished_at"] = timestamp()
            if install_handler:
                signal.signal(signal.SIGTERM, old_handler)
            try:
                write_json(manifest_path, manifest)
            finally:
                checkpoint.close()


def stored_config(values: dict) -> RunConfig:
    """Read HTTP-era manifests without importing their transport or loading CUDA."""
    legacy = {"endpoint", "concurrency", "timeout", "backoff_initial", "backoff_max"}
    known = {field.name for field in fields(RunConfig)}
    if set(values) - known - legacy:
        raise ValueError("Manifest has unsupported annotation configuration fields")
    return RunConfig(**{k: v for k, v in values.items() if k in known})


def read_run(run_dir: Path) -> tuple[Dataset, list[dict], dict, int]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = stored_config(manifest["config"])
    taxonomy = load_taxonomy(run_dir / "taxonomy.json", run_dir / "exam_aliases.json")
    if taxonomy.checksum != manifest["identity"]["taxonomy_hash"] or digest(taxonomy.aliases) != manifest["identity"]["aliases_hash"]:
        raise ValueError("Taxonomy/aliases snapshot failed integrity check")
    source = load_from_disk(str(run_dir / "source"))
    if source_digest(source) != manifest["selected_source_hash"]:
        raise ValueError("Source snapshot failed integrity check")
    checkpoint = Checkpoint(run_dir / "checkpoint.sqlite3")
    try:
        annotated, audits = build_annotated(source, checkpoint, config, taxonomy, manifest["run_id"])
        return annotated, audits, manifest, checkpoint.generation
    finally:
        checkpoint.close()
