"""Compare corrected datasets and recorded metrics with the verified original backups."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from datasets import disable_progress_bars, load_from_disk
import pandas as pd
import pyarrow.parquet as pq

MMLU_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = MMLU_ROOT.parent / "light-benchmark"
EXPERIMENTS = ("direct", "cot", "direct-base", "reasoning-on", "hard-direct", "hard-cot")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def test_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/*/dataset_dict.json")):
        dataset = load_from_disk(str(path.parent))
        rows.extend({**row, "config": path.parent.name} for row in dataset["test"])
    return rows


def count_changes(before: list[dict], after: list[dict], dataset: str) -> list[dict]:
    changes = []
    for dimension in ("academic_level", "config", "exam", "subject"):
        old = Counter((row["academic_level"], str(row[dimension])) for row in before)
        new = Counter((row["academic_level"], str(row[dimension])) for row in after)
        for level, value in sorted(old.keys() | new.keys()):
            key = level, value
            changes.append({"dataset": dataset, "level": level, "dimension": dimension,
                            "value": value, "before": old[key], "after": new[key],
                            "removed": old[key] - new[key]})
    return changes


def scores(path: Path) -> list[int]:
    values = pq.read_table(path, columns=["metric.extractive_match"]).column(0).to_pylist()
    if not values or any(value not in (0, 1) for value in values):
        raise ValueError(f"Invalid recorded scores: {path}")
    return values


def metric_changes(experiment: Path, original: Path) -> list[dict]:
    manifest = json.loads((experiment / "manifest.json").read_text())
    groups = defaultdict(list)
    changes = []
    for record in manifest["tasks"]:
        relative = record["source_file"]
        old, new = scores(original / relative), scores(experiment / relative)
        task, shot = record["task"].rsplit("|", 1)
        prefix, area = task.split(":", 1)
        level = "high_school" if "high-school" in prefix else "undergraduate"
        identity = {"experiment": experiment.name, "model": record["model"],
                    "level": level, "shots": int(shot)}
        changes.append({**identity, "aggregation": "area", "config": area,
                        "before_n": len(old), "after_n": len(new),
                        "before_accuracy": statistics.mean(old), "after_accuracy": statistics.mean(new)})
        groups[record["model"], level, int(shot)].append((old, new))
    for (model, level, shot), areas in sorted(groups.items()):
        for aggregation in ("micro", "macro"):
            old_n = sum(len(old) for old, _ in areas)
            new_n = sum(len(new) for _, new in areas)
            if aggregation == "micro":
                old_mean = sum(sum(old) for old, _ in areas) / old_n
                new_mean = sum(sum(new) for _, new in areas) / new_n
            else:
                old_mean = statistics.mean(statistics.mean(old) for old, _ in areas)
                new_mean = statistics.mean(statistics.mean(new) for _, new in areas)
            changes.append({"experiment": experiment.name, "model": model, "level": level,
                            "shots": shot, "aggregation": aggregation, "config": "_all",
                            "before_n": old_n, "after_n": new_n,
                            "before_accuracy": old_mean, "after_accuracy": new_mean})
    wide = pd.read_csv(experiment / "metrics.csv").set_index("model_name")
    for record in manifest["tasks"]:
        values = scores(experiment / record["source_file"])
        actual = wide.loc[record["model"], record["task"] + "|extractive_match"]
        if abs(actual - statistics.mean(values)) > 1e-12:
            raise ValueError(f"Aggregate metric differs from predictions: {record['task']}")
    for (model, level, shot), areas in groups.items():
        exemplar = next(r["task"] for r in manifest["tasks"]
                        if r["model"] == model and int(r["task"].rsplit("|", 1)[1]) == shot
                        and ("high-school" in r["task"]) == (level == "high_school"))
        prefix = exemplar.split(":", 1)[0]
        column = f"{prefix}:_average|{shot}|extractive_match"
        expected = statistics.mean(statistics.mean(new) for _, new in areas)
        if abs(wide.loc[model, column] - expected) > 1e-12:
            raise ValueError(f"Macro metric differs from predictions: {model}/{column}")
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmlu-backup", type=Path, required=True)
    parser.add_argument("--benchmark-backup", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=MMLU_ROOT / "output/dataset-correction")
    args = parser.parse_args()
    disable_progress_bars()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset_counts = []
    checks = {}
    for name, relative in (("final", "output/fewshot-work/main/datasets"), ("hard_v2", "output/hard-direct-v2/datasets")):
        before = test_rows(args.mmlu_backup / "snapshot" / relative)
        after = test_rows(MMLU_ROOT / relative)
        if not {r["id"] for r in after} <= {r["id"] for r in before}:
            raise ValueError(f"Corrected {name} is not a subset of the original")
        dataset_counts.extend(count_changes(before, after, name))
        checks[name] = {"before": len(before), "after": len(after)}
    records = []
    for suffix in EXPERIMENTS:
        name = "exp-mmlu-pt-" + suffix
        records.extend(metric_changes(BENCHMARK_ROOT / "output" / name,
                                      args.benchmark_backup / "snapshot/output" / name))
        print("Compared", name, flush=True)
    metrics = pd.DataFrame(records)
    means = []
    paired_keys = ["experiment", "model", "level", "aggregation", "config"]
    for identity, protocols in metrics.groupby(paired_keys, sort=True):
        if (set(protocols.shots) != {0, 5} or protocols.before_n.nunique() != 1
                or protocols.after_n.nunique() != 1):
            raise ValueError(f"Unpaired protocols: {identity}")
        means.append({**dict(zip(paired_keys, identity)), "shots": "mean",
                      "before_n": int(protocols.before_n.iloc[0]), "after_n": int(protocols.after_n.iloc[0]),
                      "before_accuracy": protocols.before_accuracy.mean(),
                      "after_accuracy": protocols.after_accuracy.mean()})
    metrics["shots"] = metrics.shots.astype(str)
    metrics = pd.concat([metrics, pd.DataFrame(means)], ignore_index=True)
    metrics["delta_pp"] = 100 * (metrics.after_accuracy - metrics.before_accuracy)
    keys = ["experiment", "level", "shots", "aggregation", "config"]
    metrics["before_rank"] = metrics.groupby(keys).before_accuracy.rank(method="min", ascending=False).astype(int)
    metrics["after_rank"] = metrics.groupby(keys).after_accuracy.rank(method="min", ascending=False).astype(int)
    metrics["rank_change"] = metrics.before_rank - metrics.after_rank
    metrics.sort_values(keys + ["model"]).to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(dataset_counts).to_csv(args.output_dir / "counts.csv", index=False)
    aggregate = metrics[(metrics.aggregation.isin(["micro", "macro"])) & (metrics.config == "_all")]
    lines = ["# Impacto da correção da base deduplicada", "",
             "Os testes foram filtrados pela base deduplicada revisada e mantêm as anotações existentes.",
             "Resultados recalculados das predições registradas, sem novas inferências.", "",
             "| Dataset | Teste anterior | Teste corrigido | Removidas |", "| --- | ---: | ---: | ---: |"]
    for name, count in checks.items():
        lines.append(f"| {name} | {count['before']} | {count['after']} | {count['before'] - count['after']} |")
    lines += ["", "## Impacto por experimento", "",
              "Faixa da mudança por nível e protocolo, em pontos percentuais. Macro dá peso igual às áreas.", "",
              "| Experimento | Modelos | Mudança micro | Mudança macro | Posições alteradas na macro média | Maior mudança de posição |",
              "| --- | ---: | --- | --- | ---: | ---: |"]
    for experiment, subset in aggregate.groupby("experiment", sort=True):
        micro = subset[subset.aggregation == "micro"]
        macro = subset[subset.aggregation == "macro"]
        ranking = macro[macro.shots == "mean"]
        changed = int((ranking.rank_change != 0).sum())
        maximum = int(ranking.rank_change.abs().max())
        lines.append(f"| {experiment} | {subset.model.nunique()} | {micro.delta_pp.min():+.4f} a {micro.delta_pp.max():+.4f} | {macro.delta_pp.min():+.4f} a {macro.delta_pp.max():+.4f} | {changed} | {maximum} |")
    lines += ["", "As posições alteradas contam pares modelo/nível na macro média de 0/5-shot; os rankings são separados por nível."]
    lines += ["", "## Arquivos e reprodução", "",
              "- `counts.csv`: contagens por nível, configuração, exame e disciplina.",
              "- `metrics.csv`: scores antes/depois por área, micro e macro em 0-shot, 5-shot e sua média; diferenças em pontos percentuais e rankings por grupo. Empates recebem a mesma posição.",
              "- Manifestos dos experimentos: execuções selecionadas, hashes das entradas e saídas e verificações de conteúdo.",
              "- `manifest.json`: backups usados, inventário e hashes deste relatório e de suas tabelas.", "",
              "As versões corrigidas ocupam os caminhos atuais. Todos os artefatos anteriores estão nos backups verificados.",
              "A base completa anotada permanece como fonte de anotações; os testes finais usam a seleção deduplicada.", ""]
    (args.output_dir / "report.md").write_text("\n".join(lines))
    inventory = ["output/fewshot-work/main", "output/hard-direct-v2",
                 "output/easy-filter-analysis-v1", "output/easy-filter-mean-shots-v1", "ablations/hard"]
    manifest = {"mmlu_backup": str(args.mmlu_backup), "benchmark_backup": str(args.benchmark_backup),
                "dataset_counts": checks, "mmlu_artifacts": inventory,
                "benchmark_artifacts": ["output/exp-mmlu-pt-" + suffix for suffix in EXPERIMENTS],
                "checks": {"metrics_independently_recomputed": True, "subsets_verified": True},
                "sha256": {p.name: file_hash(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("Impact report written to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
