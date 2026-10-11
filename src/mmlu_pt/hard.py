"""Remove questions unanimously answered correctly in recorded 0-shot and 5-shot results."""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from datasets import DatasetDict, disable_progress_bars, load_from_disk

from mmlu_pt.hard_reporting import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    build_distributions,
    build_mean_metrics,
    build_metrics,
    build_sensitivity,
    write_report,
)

PANEL = ("gemini-3.8-flash", "google/gemma-4-31B-it", "Qwen/Qwen3.5-27B")
LEVELS = ("high_school", "undergraduate")
SHOTS = (0, 5)
MISSING_SCORE = 255
DEFAULT_RESULTS = Path("/raid/aluno_paulosantana/repositories/light-benchmark/output/exp-mmlu-pt-direct")
DEFAULT_SOURCE = Path("output/fewshot-work/main/datasets")
DEFAULT_OUTPUT = Path("output/hard-direct-v2")
NAME = "MMLU-PT-Hard-Direct v2"
DETAIL_NAME = re.compile(r"details_mmlu-pt-(high-school|undergraduate)-direct:([^|]+)\|(0|5)_.*\.parquet$")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def input_hashes(root: Path, files: list[Path]) -> dict:
    return {str(path.relative_to(root)): file_hash(path) for path in sorted(files)}


def question_query(row: dict) -> str:
    alternatives = "\n".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(row["choices"]))
    return f"Questão:\n{row['question']}\n\nAlternativas:\n{alternatives}"


def load_sources(source_dir: Path) -> tuple[dict, list[dict], dict]:
    configs, rows, index = {}, [], {}
    dev_ids = set()
    for level in LEVELS:
        for path in sorted((source_dir / level).glob("*/dataset_dict.json")):
            area = path.parent.name
            dataset = load_from_disk(str(path.parent))
            if not isinstance(dataset, DatasetDict) or set(dataset) != {"test", "dev"}:
                raise ValueError(f"{path.parent}: expected test/dev DatasetDict")
            if dataset["test"].features != dataset["dev"].features:
                raise ValueError(f"{path.parent}: test/dev schemas differ")
            indices = []
            for row in dataset["test"]:
                key = (level, row["id"])
                if key in index:
                    raise ValueError(f"Duplicate test ID: {key}")
                if row["academic_level"] != level:
                    raise ValueError(f"Academic level differs from config: {key}")
                index[key] = len(rows)
                indices.append(len(rows))
                rows.append({**row, "config": area})
            for row in dataset["dev"]:
                if row["id"] in dev_ids:
                    raise ValueError(f"Duplicate dev ID: {row['id']}")
                dev_ids.add(row["id"])
            configs[level, area] = (dataset, indices)
    if not rows or any(not any(key[0] == level for key in configs) for level in LEVELS):
        raise ValueError("Source must contain test datasets for both academic levels")
    if dev_ids.intersection(row["id"] for row in rows):
        raise ValueError("Source dev and test IDs overlap")
    return configs, rows, index


def load_scores(results_dir: Path, rows: list[dict], index: dict) -> tuple[dict, list[Path]]:
    details = results_dir / "details"
    files = sorted(details.rglob("*.parquet"))
    if not files:
        raise ValueError(f"No prediction files in {details}")
    queries = [question_query(row) for row in rows]
    scores = {}
    for path in files:
        match = DETAIL_NAME.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unexpected detail filename: {path}")
        level, area, shot = match.groups()
        level, column = level.replace("-", "_"), SHOTS.index(int(shot))
        model = str(path.parent.parent.relative_to(details))
        if model not in scores:
            scores[model] = np.full((len(rows), len(SHOTS)), MISSING_SCORE, dtype=np.uint8)
            print(f"Reading {model}", flush=True)
        table = pq.read_table(path, columns=["doc.specific.id", "doc.query", "doc.gold_index",
                                             "doc.choices", "metric.extractive_match"])
        for record in table.to_pylist():
            key = (level, record["id"])
            if key not in index:
                raise ValueError(f"Unknown prediction ID in {path}: {key}")
            position = index[key]
            row = rows[position]
            labels = [chr(65 + i) for i in range(len(row["choices"]))]
            if (row["config"] != area or record["query"] != queries[position]
                    or record["gold_index"] != row["answer"] or record["choices"] != labels):
                raise ValueError(f"Prediction content differs from source in {path}: {key}")
            score = record["extractive_match"]
            if score not in (0, 1):
                raise ValueError(f"Non-binary extractive_match in {path}: {key}: {score}")
            if scores[model][position, column] != MISSING_SCORE:
                raise ValueError(f"Duplicate prediction for {model}, {key}, {shot}-shot")
            scores[model][position, column] = score
    for model in PANEL:
        if model not in scores:
            raise ValueError(f"Missing panel model: {model}")
    for model, values in scores.items():
        if np.any(values == MISSING_SCORE):
            raise ValueError(f"Missing predictions for {model}: {int((values == MISSING_SCORE).sum())}")
    return dict(sorted(scores.items())), files


def select_hard(scores: dict) -> tuple[np.ndarray, np.ndarray]:
    panel_scores = np.stack([scores[model] for model in PANEL], axis=1)
    return ~np.all(panel_scores.mean(axis=2) == 1, axis=1), panel_scores


def save_audit(path: Path, rows: list[dict], selected: np.ndarray, panel_scores: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row, keep, values in zip(rows, selected, panel_scores):
            record = {"id": row["id"], "academic_level": row["academic_level"],
                      "config": row["config"],
                      "scores_0shot": dict(zip(PANEL, map(int, values[:, SHOTS.index(0)]))),
                      "scores_5shot": dict(zip(PANEL, map(int, values[:, SHOTS.index(5)]))),
                      "mean_scores": dict(zip(PANEL, map(float, values.mean(axis=1)))),
                      "correct": int(values.sum()), "errors": int((values == 0).sum()),
                      "removed": bool(not keep), "selected": bool(keep)}
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def save_datasets(output_dir: Path, configs: dict, selected: np.ndarray) -> dict:
    summary = {level: {} for level in LEVELS}
    for (level, area), (source, indices) in configs.items():
        test = source["test"].select([i for i, position in enumerate(indices) if selected[position]])
        summary[level][area] = {"original_test": len(source["test"]), "test": len(test),
                                "dev": len(source["dev"]) if len(test) else 0}
        if not len(test):
            continue
        built = DatasetDict({"test": test, "dev": source["dev"]})
        target = output_dir / "datasets" / level / area
        target.parent.mkdir(parents=True, exist_ok=True)
        built.save_to_disk(str(target))
        json_dir = output_dir / "jsonl" / level / area
        json_dir.mkdir(parents=True, exist_ok=True)
        for split, dataset in built.items():
            json_path = json_dir / f"{split}.jsonl"
            dataset.to_json(str(json_path), force_ascii=False)
            reloaded = load_from_disk(str(target))[split]
            if dataset.features != reloaded.features or list(dataset) != list(reloaded):
                raise ValueError(f"Hugging Face export differs: {level}/{area}/{split}")
            with json_path.open(encoding="utf-8") as handle:
                json_rows = [json.loads(line) for line in handle]
            if json_rows != list(dataset):
                raise ValueError(f"JSONL export differs: {level}/{area}/{split}")
    return summary


def build(results_dir: Path, source_dir: Path, output_dir: Path) -> dict:
    results_dir, source_dir, output_dir = (path.resolve() for path in (results_dir, source_dir, output_dir))
    if output_dir.exists():
        raise ValueError(f"Output already exists; choose a new --output-dir: {output_dir}")
    for input_dir in (source_dir, results_dir):
        if output_dir.is_relative_to(input_dir) or input_dir.is_relative_to(output_dir):
            raise ValueError("Output and input directories must not contain each other")
    disable_progress_bars()
    configs, rows, index = load_sources(source_dir)
    scores, detail_files = load_scores(results_dir, rows, index)
    selected, panel_scores = select_hard(scores)
    if not np.array_equal(selected, np.any(panel_scores == 0, axis=(1, 2))):
        raise ValueError("Selection does not agree with removing six correct answers")
    reversed_protocols = {model: values[:, ::-1] for model, values in scores.items()}
    if not np.array_equal(selected, select_hard(reversed_protocols)[0]):
        raise ValueError("Selection depends on protocol order")
    if not np.array_equal(selected, ~np.all(panel_scores[:, ::-1].mean(axis=2) == 1, axis=1)):
        raise ValueError("Selection depends on panel order")
    changed_external = {model: values if model in PANEL else 1 - values for model, values in scores.items()}
    if not np.array_equal(selected, select_hard(changed_external)[0]):
        raise ValueError("Selection depends on models outside the panel")
    print(f"Selected {int(selected.sum())} / {len(rows)} questions", flush=True)
    print("Hashing prediction and source files", flush=True)
    source_files = [path for path in source_dir.rglob("*") if path.is_file()]
    result_files = detail_files + sorted((results_dir / "results").rglob("*.json"))
    if (results_dir / "metrics.csv").exists():
        result_files.append(results_dir / "metrics.csv")
    sources = {"path": str(source_dir), "sha256": input_hashes(source_dir, source_files)}
    results = {"path": str(results_dir), "sha256": input_hashes(results_dir, result_files)}
    selected_keys = [[row["academic_level"], row["id"]] for row, keep in zip(rows, selected) if keep]
    selection_hash = hashlib.sha256(json.dumps(selected_keys, separators=(",", ":")).encode()).hexdigest()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        summary = save_datasets(stage, configs, selected)
        save_audit(stage / "audit.jsonl", rows, selected, panel_scores)
        distributions = build_distributions(rows, selected)
        write_json(stage / "distributions.json", distributions)
        metrics = build_metrics(rows, selected, scores, SHOTS)
        metrics.to_csv(stage / "scores.csv", index=False)
        mean_metrics = build_mean_metrics(rows, selected, scores)
        mean_metrics.to_csv(stage / "mean_scores.csv", index=False)
        sensitivity = build_sensitivity(rows, scores, PANEL, SHOTS)
        sensitivity.to_csv(stage / "sensitivity.csv", index=False)
        manifest = {"name": NAME, "version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
                    "source": sources, "experiment": results, "panel": list(PANEL),
                    "selection": {"shots": list(SHOTS), "metric": "extractive_match",
                                  "model_score": "(score_0shot + score_5shot) / 2",
                                  "remove_when": "all three model means equal 1 (six correct answers)",
                                  "min_errors_to_keep": 1, "sha256_ordered_ids": selection_hash},
                    "models": list(scores), "summary": summary,
                    "original_test": len(rows), "selected_test": int(selected.sum()),
                    "statistics": {"confidence": 0.95, "micro_interval": "Wilson",
                                   "macro_interval": "stratified bootstrap", "bootstrap_samples": BOOTSTRAP_SAMPLES,
                                   "bootstrap_seed": BOOTSTRAP_SEED,
                                   "mean_interval": "paired item bootstrap of 0-shot/5-shot means"},
                    "checks": {"source_content_matches": True, "complete_predictions": True,
                               "dev_test_disjoint": True, "exports_equal": True,
                               "removes_exactly_six_correct": True, "protocol_order_invariant": True,
                               "panel_order_invariant": True, "selection_independent_of_external_models": True}}
        write_json(stage / "manifest.json", manifest)
        write_report(stage / "report.md", manifest, distributions, metrics, mean_metrics, sensitivity)
        stage.rename(output_dir)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(f"Artifacts saved to {output_dir}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        build(args.results_dir, args.source_dir, args.output_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
