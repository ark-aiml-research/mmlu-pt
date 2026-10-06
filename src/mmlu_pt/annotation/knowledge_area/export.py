"""Final annotations and validated publication in source row order."""

import os
import shutil
import uuid
from pathlib import Path

from datasets import Dataset, Value, load_from_disk

from .checkpoint import Checkpoint
from .config import UNCERTAIN, RunConfig, canonical_json, digest, write_json
from .taxonomy import Taxonomy

ANNOTATION_FEATURES = {
    "annotation_id": "string", "annotation_row_index": "int64", "subject": "string",
    "macro_area": "string", "alternative_subject": "string", "subject_confidence": "float64",
    "subject_justification": "string", "annotation_status": "string", "annotation_agreement": "string",
    "annotation_pass_1_subject": "string", "annotation_pass_2_subject": "string",
    "annotation_model": "string", "annotation_prompt_version": "string", "taxonomy_version": "string",
    "annotation_run_id": "string"}


def needs_adjudication(results: list[dict]) -> bool:
    labels = [result["subject"] for result in results]
    return UNCERTAIN in labels or len(set(labels)) > 1


def aggregate(tasks: list[dict], config: RunConfig) -> dict:
    independent = sorted((t for t in tasks if t["kind"] == "independent"), key=lambda t: t["pass_number"])
    labels = [t["result"]["subject"] if t["status"] == "success" and t["result"] else None for t in independent]
    base = {"subject": None, "alternative_subject": None, "confidence": None, "justification": None,
            "status": "error", "agreement": "incomplete", "pass_labels": labels,
            "adjudication_requested": any(t["kind"] == "adjudication" for t in tasks)}
    if len(labels) != config.num_independent_passes or any(label is None for label in labels):
        return base
    results = [t["result"] for t in independent]
    if len(labels) == 1:
        agreement = "single_pass"
    elif len(set(labels)) > 1:
        agreement = "disagreement"
    elif UNCERTAIN in labels:
        agreement = "uncertain"
    else:
        agreement = "agreement"
    base["agreement"] = agreement
    if len(labels) > 1 and needs_adjudication(results):
        if not config.adjudicate_disagreements:
            return base | {"subject": UNCERTAIN, "status": "uncertain",
                           "justification": "Independent passes did not agree on an allowed subject."}
        adjudication = next((t for t in tasks if t["kind"] == "adjudication"), None)
        if not adjudication or adjudication["status"] != "success":
            return base
        result = adjudication["result"]
        return base | result | {"status": "uncertain" if result["subject"] == UNCERTAIN else "adjudicated"}
    result = dict(results[0])
    result["confidence"] = min(r["confidence"] for r in results)
    return base | result | {"status": "uncertain" if result["subject"] == UNCERTAIN else "accepted"}


def build_annotated(source: Dataset, checkpoint: Checkpoint, config: RunConfig,
                    taxonomy: Taxonomy, run_id: str) -> tuple[Dataset, list[dict]]:
    mapping = checkpoint.row_mapping()
    if len(mapping) != len(source) or [r["position"] for r in mapping] != list(range(len(source))):
        raise ValueError("Checkpoint/source row mapping is incomplete")
    overlap = set(source.column_names) & set(ANNOTATION_FEATURES)
    if overlap:
        raise ValueError(f"Source already contains annotation fields: {sorted(overlap)}")
    summaries = {identity: aggregate(checkpoint.tasks(identity), config) for identity in checkpoint.question_ids()}
    columns = {name: [] for name in ANNOTATION_FEATURES}
    audits = []
    for row in mapping:
        result = summaries[row["annotation_id"]]
        labels = result["pass_labels"]
        record = {
            "annotation_id": row["annotation_id"], "annotation_row_index": row["row_index"],
            "subject": result["subject"], "macro_area": taxonomy.macro_areas.get(result["subject"]),
            "alternative_subject": result["alternative_subject"], "subject_confidence": result["confidence"],
            "subject_justification": result["justification"], "annotation_status": result["status"],
            "annotation_agreement": result["agreement"], "annotation_pass_1_subject": labels[0] if labels else None,
            "annotation_pass_2_subject": labels[1] if len(labels) > 1 else None,
            "annotation_model": config.model, "annotation_prompt_version": config.prompt_version,
            "taxonomy_version": taxonomy.version, "annotation_run_id": run_id}
        for name, value in record.items():
            columns[name].append(value)
        audits.append({"adjudication_requested": result["adjudication_requested"]})
    annotated = source
    for name, values in columns.items():
        annotated = annotated.add_column(name, values, feature=Value(ANNOTATION_FEATURES[name]))
    return annotated, audits


def publish_exports(annotated: Dataset, source: Dataset, run_dir: Path, generation: int) -> Path:
    root = run_dir / "exports"
    root.mkdir(exist_ok=True)
    identifier = f"generation-{generation:04d}-{uuid.uuid4().hex[:8]}"
    stage = root / (".staging-" + identifier)
    stage.mkdir()
    try:
        annotated.save_to_disk(str(stage / "dataset"))
        annotated.to_parquet(str(stage / "data.parquet"))
        with (stage / "data.jsonl").open("w", encoding="utf-8") as handle:
            for row in annotated:
                handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        reloaded = load_from_disk(str(stage / "dataset"))
        if len(reloaded) != len(source) or reloaded.features != annotated.features:
            raise ValueError("Export count or schema mismatch")
        for original, saved in zip(source, reloaded, strict=True):
            if digest(original) != digest({name: saved[name] for name in source.column_names}):
                raise ValueError("Export changed source values or row order")
        import pyarrow.parquet as parquet
        if parquet.read_metadata(stage / "data.parquet").num_rows != len(source):
            raise ValueError("Parquet count mismatch")
        destination = root / identifier
        os.replace(stage, destination)
        write_json(root / "latest.json", {"directory": identifier, "generation": generation,
                                           "rows": len(annotated)})
        return destination
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def push_dataset(data: Dataset, repository: str, config: RunConfig) -> None:
    if repository.strip().rstrip("/") == config.dataset.strip().rstrip("/"):
        raise ValueError("Refusing to overwrite the source dataset")
    if config.dry_run:
        raise ValueError("Dry runs cannot be pushed to the Hub")
    data.push_to_hub(repository, split=config.split)
