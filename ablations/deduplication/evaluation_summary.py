"""Verify and summarize the zero-shot item-selection ablation."""

import argparse
import hashlib
import json
from pathlib import Path
from string import ascii_uppercase

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_from_disk

from mmlu_pt.annotation.knowledge_area.dataset import annotation_id

REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = Path(__file__).resolve().parent
OLD_OUTPUT = REPO_ROOT.parent / "light-benchmark" / "output" / "old"


def expected_query(row: dict) -> str:
    alternatives = "\n".join(f"({label}) {choice}" for label, choice in zip(ascii_uppercase, row["choices"]))
    return (f"Pergunta:\n{row['question']}\n\nAlternativas:\n{alternatives}\n\n"
            "Responda somente com a letra maiúscula da alternativa correta.\nResposta:")


def read_scores(path: Path, questions: list[dict], indices: set[int]) -> dict[int, float]:
    scores = {}
    table = pq.read_table(path, columns=["doc.id", "doc.query", "doc.gold_index", "doc.choices", "metric.extractive_match"])
    for identifier, query, gold, labels, score in zip(*(table[name].to_pylist() for name in
                                                     ("id", "query", "gold_index", "choices", "extractive_match")), strict=True):
        index = int(identifier)
        if index not in indices or index in scores:
            raise ValueError(f"Duplicate or unexpected document in {path}")
        source = questions[index]
        if query != source["query"] or gold != source["answer"] or labels != source["labels"]:
            raise ValueError(f"Question, choices or answer mismatch in {path}")
        if score not in (0.0, 1.0):
            raise ValueError(f"Non-binary score in {path}")
        scores[index] = score
    if set(scores) != indices:
        raise ValueError(f"Incomplete prediction coverage in {path}")
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ABLATION_DIR / "aggregates")
    args = parser.parse_args()
    filtered = list(load_from_disk(str(REPO_ROOT / "output" / "04 - filtered-huggingface")))
    exact_path = next((REPO_ROOT / "output" / "05 - deduplicated").glob("*.jsonl"))
    exact = [json.loads(line) for line in exact_path.read_text().splitlines() if line]
    # The Hub export stores edition and item metadata as strings.
    for row in exact:
        for key in ("num", "exam_edition"):
            row[key] = str(row[key])
    deduplicated = list(load_from_disk(str(REPO_ROOT / "output" / "06 - semantic-deduplicated" / "huggingface")))
    identifiers = [annotation_id(row) for row in filtered]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Filtered content IDs are not unique.")
    selections = {"filtered": set(range(len(filtered)))}
    for name, rows in (("exact", exact), ("deduplicated", deduplicated)):
        kept = {annotation_id(row) for row in rows}
        selections[name] = {index for index, identity in enumerate(identifiers) if identity in kept}
        if len(selections[name]) != len(rows):
            raise ValueError(f"Incomplete {name} selection.")
    if [len(selections[name]) for name in ("filtered", "exact", "deduplicated")] != [41653, 39042, 37887]:
        raise ValueError("Unexpected ablation populations.")
    questions = [{"query": expected_query(row), "answer": row["answer"],
                  "labels": list(ascii_uppercase[:len(row["choices"])])} for row in filtered]
    metadata = pd.DataFrame(filtered)[["exam", "academic_level"]]
    csvs = {name: pd.read_csv(OLD_OUTPUT / "experiments" / f"exp-mmlu-pt-{name}.csv").set_index("model_name")
            for name in ("filtered", "deduplicated")}
    if set(csvs["filtered"].index) != set(csvs["deduplicated"].index) or len(csvs["filtered"]) != 19:
        raise ValueError("The ablation requires the same 19 models in both populations.")
    rows, by_exam, sources = [], [], []
    for model in sorted(csvs["filtered"].index):
        files = {}
        for name in ("filtered", "deduplicated"):
            matches = sorted((OLD_OUTPUT / "details" / model).rglob(f"details_mmlu-pt-{name}|0_*.parquet"))
            if len(matches) != 1:
                raise ValueError(f"Expected one {name} prediction file for {model}.")
            files[name] = matches[0]
            sources.append({"path": str(matches[0]), "sha256": hashlib.sha256(matches[0].read_bytes()).hexdigest()})
        scores = read_scores(files["filtered"], questions, selections["filtered"])
        retained = read_scores(files["deduplicated"], questions, selections["deduplicated"])
        if any(value != scores[index] for index, value in retained.items()):
            raise ValueError(f"Selected scores differ from the original predictions for {model}.")
        frame = metadata.assign(score=pd.Series(scores))
        record = {"model": model}
        for name, indices in selections.items():
            part = frame.loc[sorted(indices)]
            record[name] = part["score"].mean()
            if name != "exact":
                column = f"mmlu-pt-{name}|0|extractive_match"
                if not np.isclose(record[name], csvs[name].at[model, column], atol=1e-12, rtol=0):
                    raise ValueError(f"CSV score mismatch for {model}, {name}.")
                record[f"{name}_exam_macro"] = part.groupby("exam")["score"].mean().mean()
                for level, group in part.groupby("academic_level"):
                    record[f"{name}_{level}"] = group["score"].mean()
                exams = part.groupby("exam")["score"].agg(["size", "mean"])
                for exam, values in exams.iterrows():
                    by_exam.append({"model": model, "population": name, "exam": exam,
                                    "questions": int(values['size']), "accuracy": values['mean']})
        rows.append(record)
        print(f"Verified {model}", flush=True)
    models = pd.DataFrame(rows).sort_values("deduplicated", ascending=False, ignore_index=True)
    for name in ("filtered", "exact", "deduplicated"):
        models[f"{name}_rank"] = models[name].rank(ascending=False, method="min").astype(int)
    models["delta_pp"] = 100 * (models["deduplicated"] - models["filtered"])
    exams = pd.DataFrame(by_exam)
    decompositions = []
    for model, group in exams.groupby("model"):
        first = group[group["population"] == "filtered"].set_index("exam")
        last = group[group["population"] == "deduplicated"].set_index("exam").loc[first.index]
        old_weights = first["questions"] / first["questions"].sum()
        new_weights = last["questions"] / last["questions"].sum()
        within = float((old_weights * (last["accuracy"] - first["accuracy"])).sum())
        composition = float(((new_weights - old_weights) * last["accuracy"]).sum())
        decompositions.append({"model": model, "within_exam_pp": 100 * within,
                               "composition_pp": 100 * composition, "total_pp": 100 * (within + composition)})
    decomposition = pd.DataFrame(decompositions)
    aggregates = []
    for label, suffix in (("Overall micro", ""), ("Equal weight per exam", "_exam_macro"),
                          ("High-school micro", "_high_school"), ("Undergraduate micro", "_undergraduate")):
        first, last = models[f"filtered{suffix}"], models[f"deduplicated{suffix}"]
        aggregates.append({"aggregation": label, "filtered_pct": 100 * first.mean(),
                           "deduplicated_pct": 100 * last.mean(), "delta_pp": 100 * (last - first).mean(),
                           "models_increasing": int((last > first).sum()), "models_decreasing": int((last < first).sum())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("evaluation_models", models), ("evaluation_aggregations", pd.DataFrame(aggregates)),
                        ("evaluation_exams", exams), ("evaluation_decomposition", decomposition)):
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    summary = {"models": len(models), "populations": {name: len(indices) for name, indices in selections.items()},
               "spearman_rho": models["filtered"].corr(models["deduplicated"], method="spearman"),
               "exact_delta_mean_pp": 100 * (models["exact"] - models["filtered"]).mean(),
               "semantic_delta_mean_pp": 100 * (models["deduplicated"] - models["exact"]).mean(),
               "within_exam_mean_pp": decomposition["within_exam_pp"].mean(),
               "composition_mean_pp": decomposition["composition_pp"].mean(),
               "sources": sources, "checks": {"complete_coverage": True, "content_equal": True,
                                                "retained_scores_equal": True, "csv_metrics_equal": True},
               "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(args.output_dir.glob("evaluation_*.csv"))}}
    (args.output_dir / "evaluation_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("sources", "files_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
