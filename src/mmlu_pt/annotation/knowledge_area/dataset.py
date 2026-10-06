"""Source inspection, stable identities and deterministic stratified sampling."""

from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import HfApi

from .config import RunConfig, digest
from .taxonomy import CONTEXT_FIELDS, Taxonomy

IDENTITY_FIELDS = ("exam", "exam_edition", "num", "question", "choices")
REQUIRED_FIELDS = frozenset({"exam", "question", "choices", "answer"})


def annotation_id(row) -> str:
    return digest({field: row.get(field) for field in IDENTITY_FIELDS})


def metadata(row) -> dict:
    return {field: row.get(field) for field in CONTEXT_FIELDS}


def load_source(config: RunConfig, pinned_revision: str | None = None) -> tuple[Dataset, dict]:
    revision = config.dataset_revision or pinned_revision
    if config.dataset_path:
        loaded = load_from_disk(config.dataset_path)
        source = {"identifier": str(Path(config.dataset_path).resolve()), "revision": None, "kind": "local"}
    else:
        revision = HfApi().dataset_info(config.dataset, revision=revision).sha
        loaded = load_dataset(config.dataset, name=config.dataset_config, revision=revision)
        source = {"identifier": config.dataset, "revision": revision, "kind": "hub"}
    splits = list(loaded) if isinstance(loaded, DatasetDict) else [config.split]
    if isinstance(loaded, DatasetDict):
        if config.split not in loaded:
            raise ValueError(f"Unknown split {config.split!r}; available: {splits}")
        data = loaded[config.split]
    elif isinstance(loaded, Dataset):
        data = loaded
    else:
        raise ValueError("Expected a Hugging Face Dataset or DatasetDict")
    source.update({"config": data.info.config_name or config.dataset_config, "split": config.split, "available_splits": splits,
                   "fingerprint": data._fingerprint, "features": data.features.to_dict(), "rows": len(data)})
    return data, source


def inspect_dataset(data: Dataset, taxonomy: Taxonomy) -> dict:
    missing = sorted(REQUIRED_FIELDS - set(data.column_names))
    if missing:
        return {"valid": False, "total_rows": len(data), "missing_fields": missing,
                "exams": [], "unmapped_exams": [], "invalid_rows": []}
    counts, candidate_counts, ids = Counter(), {}, Counter()
    invalid, unknown = [], set()
    # All classification/preflight iteration explicitly excludes the answer column.
    for index, row in enumerate(data.remove_columns("answer")):
        exam = row["exam"]
        if not isinstance(exam, str) or not exam.strip():
            invalid.append({"row_index": index, "reason": "invalid_exam"})
            continue
        counts[exam] += 1
        if taxonomy.exam_key(exam) is None:
            unknown.add(exam)
        else:
            try:
                candidate_counts.setdefault(exam, set()).add(len(taxonomy.resolve_candidates(exam, metadata(row))))
            except ValueError:
                invalid.append({"row_index": index, "reason": "conflicting_context_rules"})
        if not isinstance(row["question"], str) or not row["question"].strip():
            invalid.append({"row_index": index, "reason": "invalid_question"})
        choices = row["choices"]
        if not isinstance(choices, list) or not choices or any(not isinstance(c, str) or not c.strip() for c in choices):
            invalid.append({"row_index": index, "reason": "invalid_choices"})
        ids[annotation_id(row)] += 1
    exams = [{"exam": exam, "rows": count, "taxonomy_exam": taxonomy.exam_key(exam),
              "candidate_counts": sorted(candidate_counts.get(exam, []))}
             for exam, count in sorted(counts.items())]
    return {"valid": not unknown and not invalid, "total_rows": len(data), "exams": exams,
            "unmapped_exams": sorted(unknown), "invalid_rows": invalid,
            "duplicate_id_occurrences": sum(n - 1 for n in ids.values()),
            "unused_taxonomy_exams": sorted(set(taxonomy.document["exams"]) -
                                            {taxonomy.exam_key(e) for e in counts})}


def sample_indices(data: Dataset, config: RunConfig) -> list[int]:
    if not config.dry_run:
        return list(range(len(data)))
    groups = {}
    for index, row in enumerate(data.remove_columns("answer")):
        rank = digest([config.seed, annotation_id(row)])
        groups.setdefault(row["exam"], []).append((rank, index))
    return sorted(index for group in groups.values()
                  for _, index in sorted(group)[:config.samples_per_exam])
