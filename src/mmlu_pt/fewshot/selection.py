"""Source loading, stable row identities and deterministic per-stratum ranking."""

from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import HfApi

from mmlu_pt.annotation.knowledge_area.dataset import IDENTITY_FIELDS, annotation_id

from .config import ID_COLUMN, LEVELS, RATIONALE_COLUMN, REQUIRED_COLUMNS, RunConfig, digest


def load_source(config: RunConfig) -> tuple[Dataset, dict]:
    return load_input(config.source, config.source_revision, config.split, config.source_path)


def load_input(identifier: str, revision: str | None, split: str, path: str | None = None) -> tuple[Dataset, dict]:
    if path:
        data = load_from_disk(path)
        if not isinstance(data, Dataset):
            data = data[split]
    else:
        revision = HfApi().dataset_info(identifier, revision=revision).sha
        data = load_dataset(identifier, revision=revision, split=split)
    source = {"identifier": identifier, "revision": revision, "split": split,
              "rows": len(data), "fingerprint": data._fingerprint, "features": data.features.to_dict()}
    if path:
        source["local_path"] = str(Path(path).resolve())
    return data, source


def join_deduplicated(annotated: Dataset, deduplicated: Dataset) -> Dataset:
    """Keep annotated records selected by deduplication, checking every original field."""
    fields = deduplicated.column_names
    missing = set(fields) - set(annotated.column_names)
    if missing:
        raise ValueError(f"Missing original columns in annotations: {sorted(missing)}")
    originals = {}
    for row in deduplicated:
        identifier = annotation_id(row)
        if identifier in originals:
            raise ValueError(f"Duplicate deduplicated ID: {identifier}")
        originals[identifier] = row
    indices, seen = [], set()
    for index, row in enumerate(annotated.select_columns(fields)):
        identifier = annotation_id(row)
        if identifier in seen:
            raise ValueError(f"Duplicate annotated ID: {identifier}")
        seen.add(identifier)
        if identifier not in originals:
            continue
        if row != originals[identifier]:
            raise ValueError(f"Annotations differ from original question: {identifier}")
        indices.append(index)
    if originals.keys() - seen:
        raise ValueError(f"Missing annotations for {len(originals.keys() - seen)} deduplicated IDs")
    return annotated.select(indices)


def validate_columns(data: Dataset) -> list[str]:
    problems = [f"missing column: {name}" for name in REQUIRED_COLUMNS if name not in data.column_names]
    problems += [f"column already present: {name}" for name in (ID_COLUMN, RATIONALE_COLUMN)
                 if name in data.column_names]
    if "academic_level" in data.column_names:
        problems += [f"unknown academic_level: {value}"
                     for value in sorted(set(data["academic_level"]) - set(LEVELS))]
    return problems


def row_ids(data: Dataset) -> list[str]:
    return [annotation_id(row) for row in data.select_columns(list(IDENTITY_FIELDS))]


def build_strata(data: Dataset, ids: list[str], seed: int) -> dict[tuple[str, str], list[int]]:
    """Row indices per (academic_level, macro_area), ordered by a seeded hash of the row id."""
    ranked = defaultdict(list)
    levels, areas = data["academic_level"], data["macro_area"]
    for index, row_id in enumerate(ids):
        ranked[(levels[index], areas[index])].append((digest([seed, row_id]), row_id, index))
    return {key: [index for _, _, index in sorted(entries)] for key, entries in sorted(ranked.items())}


def normalize_text(text) -> str:
    return " ".join(text.split()) if isinstance(text, str) else ""


def build_text_index(data: Dataset) -> dict:
    """Normalized questions, their counts and row groups per exam edition, for leakage checks."""
    questions = [normalize_text(question) for question in data["question"]]
    groups = defaultdict(list)
    for index, (exam, edition) in enumerate(zip(data["exam"], data["exam_edition"])):
        groups[(exam, edition or "")].append(index)
    return {"questions": questions, "counts": Counter(questions), "groups": dict(groups)}
