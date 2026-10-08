"""Build per-config DatasetDicts, publish them and verify the result on the Hub."""

from pathlib import Path

from datasets import Dataset, DatasetDict, Value, get_dataset_config_names, load_dataset
from huggingface_hub import DatasetCard

from .config import ID_COLUMN, OUTPUT_COLUMNS, RATIONALE_COLUMN, RunConfig, letter


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
    subset = subset.add_column(ID_COLUMN, ids, feature=Value("string"))
    subset = subset.add_column(RATIONALE_COLUMN, rationales, feature=Value("string"))
    return subset.select_columns(list(OUTPUT_COLUMNS))


def save_configs(run_dir: Path, built: dict[str, dict[str, DatasetDict]]) -> None:
    for level, configs in built.items():
        for name, dataset_dict in configs.items():
            dataset_dict.save_to_disk(str(run_dir / "datasets" / level / name))


def summarize(built: dict[str, dict[str, DatasetDict]]) -> dict[str, dict[str, dict[str, int]]]:
    return {level: {name: {"test": len(dd["test"]), "dev": len(dd["dev"])} for name, dd in configs.items()}
            for level, configs in built.items()}


def card_body(level: str, repo: str, summary: dict[str, dict[str, int]], manifest: dict) -> str:
    source = manifest["source"]
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

- source: `{source['identifier']}` at revision `{source['revision']}` (split `{source['split']}`, {source['rows']} rows)
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
    if repo.strip().rstrip("/") == config.source.strip().rstrip("/"):
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
    return commits


def verify_published(repo: str, expected: dict[str, dict[str, int]], shots: int) -> dict:
    """Reload every config from the Hub and compare with the counts recorded in the manifest."""
    problems, dev_ids, test_ids = [], set(), set()
    names = sorted(get_dataset_config_names(repo))
    if names != sorted(expected):
        problems.append(f"configs on the Hub {names} differ from manifest {sorted(expected)}")
    for name in names:
        dataset_dict = load_dataset(repo, name)
        if set(dataset_dict) != {"test", "dev"}:
            problems.append(f"{name}: splits {sorted(dataset_dict)}")
            continue
        test, dev = dataset_dict["test"], dataset_dict["dev"]
        counts = expected.get(name, {})
        if len(dev) != shots or len(dev) != counts.get("dev") or len(test) != counts.get("test"):
            problems.append(f"{name}: test={len(test)} dev={len(dev)}, expected {counts}")
        if test.features != dev.features:
            problems.append(f"{name}: test and dev features differ")
        for row in dev:
            if not row[RATIONALE_COLUMN] or not row[RATIONALE_COLUMN].endswith("\nResposta: " + letter(row["answer"])):
                problems.append(f"{name}: dev row {row[ID_COLUMN][:12]} rationale does not end with the gold letter")
        dev_ids.update(dev[ID_COLUMN])
        test_ids.update(test[ID_COLUMN])
    return {"repo": repo, "configs": names, "dev_ids": dev_ids, "test_ids": test_ids, "problems": problems}
