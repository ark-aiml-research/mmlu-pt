"""Build per-config DatasetDicts, publish them and verify the result on the Hub."""

from pathlib import Path

from datasets import Dataset, DatasetDict, Value, get_dataset_config_names, load_dataset, load_from_disk
from huggingface_hub import DatasetCard, HfApi

from mmlu_pt.annotation.knowledge_area.dataset import annotation_id

from .config import ID_COLUMN, LEVELS, OUTPUT_COLUMNS, RATIONALE_COLUMN, REQUIRED_COLUMNS, RunConfig, digest, letter, slug


def reuse_configs(annotated: Dataset, data: Dataset, snapshot: Path, shots: int) -> tuple[dict, dict]:
    """Retain the original demonstrations and filter test records by stable identity."""
    annotations = {annotation_id(row): row for row in annotated.select_columns(list(REQUIRED_COLUMNS))}
    selected_ids = {annotation_id(row) for row in data}
    built = {level: {} for level in LEVELS}
    dev_records = []
    for path in sorted(snapshot.glob("*/*/dataset_dict.json")):
        level, area = path.parent.parent.name, path.parent.name
        if level not in built:
            raise ValueError(f"Unknown snapshot academic level: {level}")
        original = load_from_disk(str(path.parent))
        if not isinstance(original, DatasetDict) or set(original) != {"test", "dev"}:
            raise ValueError(f"Expected test/dev DatasetDict: {path.parent}")
        if len(original["dev"]) != shots or original["test"].features != original["dev"].features:
            raise ValueError(f"Invalid snapshot dev/schema: {path.parent}")
        if not set(OUTPUT_COLUMNS).issubset(original["test"].column_names):
            raise ValueError(f"Snapshot is missing public columns: {path.parent}")
        for split in original:
            for row in original[split]:
                identifier = row[ID_COLUMN]
                fields = {field: row[field] for field in REQUIRED_COLUMNS}
                if (annotation_id(row) != identifier or annotations.get(identifier) != fields
                        or row["academic_level"] != level or slug(row["macro_area"]) != area):
                    raise ValueError(f"Snapshot content differs from annotations: {identifier}")
        dev_records.extend(list(original["dev"]))
        for row in original["dev"]:
            rationale = row[RATIONALE_COLUMN]
            if not isinstance(rationale, str) or not rationale.endswith("\nResposta: " + letter(row["answer"])):
                raise ValueError(f"Invalid dev rationale: {row[ID_COLUMN]}")
        indices = [i for i, identifier in enumerate(original["test"][ID_COLUMN]) if identifier in selected_ids]
        test = original["test"].select(indices)
        if any(value is not None for value in test[RATIONALE_COLUMN]):
            raise ValueError(f"Test rationale is not null: {path.parent}")
        built[level][area] = DatasetDict({"test": test, "dev": original["dev"]})
    if any(not configs for configs in built.values()):
        raise ValueError("Dev snapshot must contain both academic levels")
    return built, {"sha256_records": digest(dev_records), "rows": len(dev_records)}


def build_config(data: Dataset, ids: list[str], stratum_indices: list[int], dev_entries: list[dict],
                 dev_ids: set[str]) -> DatasetDict:
    """test = stratum rows minus every few-shot row of any stratum; dev = accepted rows in rank order."""
    test_indices = [index for index in sorted(stratum_indices) if ids[index] not in dev_ids]
    dev_indices = [entry["row_index"] for entry in dev_entries]
    test = with_columns(data.select(test_indices), [ids[i] for i in test_indices], [None] * len(test_indices))
    dev = with_columns(data.select(dev_indices), [ids[i] for i in dev_indices],
                       [entry["rationale"] for entry in dev_entries])
    return DatasetDict({"test": test, "dev": dev})


def with_columns(subset: Dataset, ids: list[str], rationales: list[str | None]) -> Dataset:
    source_columns = subset.column_names
    subset = subset.add_column(ID_COLUMN, ids, feature=Value("string"))
    subset = subset.add_column(RATIONALE_COLUMN, rationales, feature=Value("string"))
    return subset.select_columns([ID_COLUMN, *source_columns, RATIONALE_COLUMN])


def save_configs(run_dir: Path, built: dict[str, dict[str, DatasetDict]]) -> None:
    for level, configs in built.items():
        for name, dataset_dict in configs.items():
            path = run_dir / "datasets" / level / name
            dataset_dict.save_to_disk(str(path))
            exported = load_from_disk(str(path))
            for split in dataset_dict:
                if (dataset_dict[split].features != exported[split].features
                        or list(dataset_dict[split]) != list(exported[split])):
                    raise ValueError(f"Export differs from source: {level}/{name}/{split}")


def summarize(built: dict[str, dict[str, DatasetDict]]) -> dict[str, dict[str, dict[str, int]]]:
    return {level: {name: {"test": len(dd["test"]), "dev": len(dd["dev"])} for name, dd in configs.items()}
            for level, configs in built.items()}


def card_body(level: str, repo: str, summary: dict[str, dict[str, int]], manifest: dict) -> str:
    source = manifest["source"]
    test_source = source["test_source"]
    rows = [f"| `{name}` | {counts['test']} | {counts['dev']} |" for name, counts in summary.items()]
    first = next(iter(summary))
    return f"""# {repo}

Portuguese multiple-choice questions ({level.replace('_', ' ')}) organized by knowledge macro-area.
Each config is one macro-area with a `test` split (evaluation) and a `dev` split
({manifest['config']['shots']} few-shot demonstrations with a `{RATIONALE_COLUMN}` column that ends in `Resposta: X`).
Few-shot rows were removed from every `test` split. `{RATIONALE_COLUMN}` is null in `test`.

| config | test | dev |
| --- | --- | --- |
{chr(10).join(rows)}

## Provenance

- test selection: `{test_source['identifier']}` at revision `{test_source['revision']}` (split `{test_source['split']}`, {test_source['rows']} rows)
- annotations: `{source['identifier']}` at revision `{source['revision']}`
- few-shot selection: seed {manifest['config']['seed']}, ranked by a hash of the stable row `id`
- rationales: `{manifest['config']['model']}` (reasoning effort `{manifest['config']['reasoning_effort']}`),
  prompt `{manifest['prompt_version']}` hash `{manifest['prompt_hash'][:16]}`, gold answer provided to the generator
- automatic audit of every demonstration (valid answer, no duplicated or shared support text, rationale ends with the gold letter)
- run id `{manifest['run_id']}`

```python
from datasets import load_dataset
test = load_dataset("{repo}", "{first}", split="test")
dev = load_dataset("{repo}", "{first}", split="dev")
```
"""


def push_level(repo: str, configs: dict[str, DatasetDict], body: str, config: RunConfig, run: str) -> dict[str, str]:
    if repo.strip().rstrip("/") in {config.source.strip().rstrip("/"), config.deduplicated_source.strip().rstrip("/")}:
        raise ValueError("Refusing to overwrite the source dataset")
    if config.dry_run:
        raise ValueError("Dry runs cannot be pushed to the Hub")
    commits = {}
    for name, dataset_dict in configs.items():
        info = dataset_dict.push_to_hub(repo, config_name=name, commit_message=f"{name}: test/dev (run {run})")
        commits[name] = info.oid
        print(f"pushed {repo} config {name}: {commits[name]}")
    card = DatasetCard.load(repo)
    card.text = body
    card.push_to_hub(repo, commit_message=f"dataset card (run {run})")
    commits["revision"] = HfApi().dataset_info(repo).sha
    return commits


def verify_published(repo: str, expected: dict[str, dict[str, int]], shots: int,
                     local_dir: Path | None = None, revision: str | None = None) -> dict:
    """Reload every config from the Hub and compare with the counts recorded in the manifest."""
    problems, dev_ids, test_ids = [], set(), set()
    revision = revision or HfApi().dataset_info(repo).sha
    names = sorted(get_dataset_config_names(repo, revision=revision))
    if names != sorted(expected):
        problems.append(f"configs on the Hub {names} differ from manifest {sorted(expected)}")
    for name in names:
        dataset_dict = load_dataset(repo, name, revision=revision)
        if set(dataset_dict) != {"test", "dev"}:
            problems.append(f"{name}: splits {sorted(dataset_dict)}")
            continue
        test, dev = dataset_dict["test"], dataset_dict["dev"]
        counts = expected.get(name, {})
        if len(dev) != shots or len(dev) != counts.get("dev") or len(test) != counts.get("test"):
            problems.append(f"{name}: test={len(test)} dev={len(dev)}, expected {counts}")
        if test.features != dev.features:
            problems.append(f"{name}: test and dev features differ")
        if local_dir:
            local = load_from_disk(str(local_dir / name))
            for split in ("test", "dev"):
                if local[split].features != dataset_dict[split].features or list(local[split]) != list(dataset_dict[split]):
                    problems.append(f"{name}/{split}: published content differs from local snapshot")
        for split, subset, seen in (("dev", dev, dev_ids), ("test", test, test_ids)):
            identifiers = subset[ID_COLUMN]
            if len(set(identifiers)) != len(identifiers) or seen.intersection(identifiers):
                problems.append(f"{name}: duplicate {split} IDs")
        for row in dev:
            if not row[RATIONALE_COLUMN] or not row[RATIONALE_COLUMN].endswith("\nResposta: " + letter(row["answer"])):
                problems.append(f"{name}: dev row {row[ID_COLUMN][:12]} rationale does not end with the gold letter")
        dev_ids.update(dev[ID_COLUMN])
        test_ids.update(test[ID_COLUMN])
    return {"repo": repo, "configs": names, "dev_ids": dev_ids, "test_ids": test_ids, "problems": problems}
