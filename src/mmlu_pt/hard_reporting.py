"""Statistics and reports for the hard subset built from recorded scores."""

from collections import Counter
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42


def wilson(correct: int, count: int) -> tuple[float, float]:
    if not count:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = correct / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    margin = z * sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def macro_bootstrap(groups: list[np.ndarray]) -> tuple[float, float]:
    # For binary scores, binomial counts exactly reproduce sampling rows with replacement.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = [rng.binomial(len(values), values.mean(), BOOTSTRAP_SAMPLES) / len(values)
               for values in groups]
    low, high = np.quantile(np.mean(samples, axis=0), [0.025, 0.975])
    return float(low), float(high)


def metric_row(model: str, shots: int, level: str, config: str, values: np.ndarray) -> dict:
    count, correct = len(values), int(values.sum())
    low, high = wilson(correct, count)
    return {"model": model, "shots": shots, "level": level, "aggregation": "micro", "config": config,
            "n": count, "correct": correct, "accuracy": correct / count if count else float("nan"),
            "ci_low": low, "ci_high": high, "n_areas": 1 if config != "_all" else None}


def build_metrics(rows: list[dict], selected: np.ndarray, scores: dict, shots: tuple) -> pd.DataFrame:
    records = []
    for model, matrix in scores.items():
        for level in sorted({row["academic_level"] for row in rows}):
            level_mask = np.array([row["academic_level"] == level for row in rows])
            areas = sorted({row["config"] for row in rows if row["academic_level"] == level})
            masks = {area: selected & level_mask & np.array([row["config"] == area for row in rows])
                     for area in areas}
            for column, shot in enumerate(shots):
                groups = []
                for area, mask in masks.items():
                    values = matrix[mask, column]
                    records.append(metric_row(model, shot, level, area, values))
                    if len(values):
                        groups.append(values)
                micro = metric_row(model, shot, level, "_all", matrix[selected & level_mask, column])
                micro["n_areas"] = len(groups)
                records.append(micro)
                low, high = macro_bootstrap(groups) if groups else (float("nan"), float("nan"))
                accuracy = float(np.mean([values.mean() for values in groups])) if groups else float("nan")
                records.append({"model": model, "shots": shot, "level": level, "aggregation": "macro",
                                "config": "_all", "n": micro["n"], "correct": None,
                                "accuracy": accuracy,
                                "ci_low": low, "ci_high": high, "n_areas": len(groups)})
    return pd.DataFrame(records)


def mean_metric_row(model: str, level: str, values: np.ndarray) -> dict:
    count = len(values)
    if not count:
        return {"model": model, "level": level, "n": 0, "accuracy_0shot": float("nan"),
                "accuracy_5shot": float("nan"), "accuracy_mean": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    # Each item is a pair; its mean is 0, 0.5 or 1. Never sample the protocols independently.
    counts = np.bincount(values.sum(axis=1).astype(int), minlength=3)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.multinomial(count, counts / count, size=BOOTSTRAP_SAMPLES)
    means = (samples[:, 1] + 2 * samples[:, 2]) / (2 * count)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"model": model, "level": level, "n": count,
            "accuracy_0shot": float(values[:, 0].mean()), "accuracy_5shot": float(values[:, 1].mean()),
            "accuracy_mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def build_mean_metrics(rows: list[dict], selected: np.ndarray, scores: dict) -> pd.DataFrame:
    records = []
    for model, matrix in scores.items():
        for level in sorted({row["academic_level"] for row in rows}):
            mask = selected & np.array([row["academic_level"] == level for row in rows])
            records.append(mean_metric_row(model, level, matrix[mask]))
    return pd.DataFrame(records)


def build_distributions(rows: list[dict], selected: np.ndarray) -> list[dict]:
    result = []
    for level in sorted({row["academic_level"] for row in rows}):
        for dimension in ("config", "macro_area", "subject", "exam"):
            before = Counter(row[dimension] for row in rows if row["academic_level"] == level)
            after = Counter(row[dimension] for row, keep in zip(rows, selected)
                            if keep and row["academic_level"] == level)
            for value, count in sorted(before.items(), key=lambda item: str(item[0])):
                result.append({"level": level, "dimension": dimension, "value": value,
                               "original_n": count, "hard_n": after[value],
                               "original_share": count / before.total(),
                               "hard_share": after[value] / after.total() if after.total() else None})
    return result


def build_sensitivity(rows: list[dict], scores: dict, panel: tuple, shots: tuple) -> pd.DataFrame:
    records = []
    for held_out in panel:
        remaining = [model for model in panel if model != held_out]
        selected = ~np.all(np.stack([scores[model] for model in remaining], axis=1) == 1, axis=(1, 2))
        for level in sorted({row["academic_level"] for row in rows}):
            mask = selected & np.array([row["academic_level"] == level for row in rows])
            for column, shot in enumerate(shots):
                record = metric_row(held_out, shot, level, "_all", scores[held_out][mask, column])
                record["selection_models"] = ";".join(remaining)
                records.append(record)
            mean = mean_metric_row(held_out, level, scores[held_out][mask])
            records.append({"model": held_out, "shots": "mean", "level": level,
                            "aggregation": "micro", "config": "_all", "n": mean["n"],
                            "correct": None, "accuracy": mean["accuracy_mean"],
                            "ci_low": mean["ci_low"], "ci_high": mean["ci_high"], "n_areas": None,
                            "selection_models": ";".join(remaining)})
    return pd.DataFrame(records)


def write_report(path: Path, manifest: dict, distributions: list[dict], metrics: pd.DataFrame,
                 mean_metrics: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    lines = [f"# {manifest['name']}", "",
             "Subconjunto obtido pela remoção de questões consensualmente fáceis para o painel.",
             "Uma questão é removida somente quando os três modelos acertam em 0-shot e 5-shot: seis acertos em seis resultados.",
             "A seleção usa diretamente o extractive_match registrado e preserva questões intermediárias.",
             "", f"Teste original: **{manifest['original_test']:,}**; hard: **{manifest['selected_test']:,}**.",
             "", "## Composição por configuração", "", "| Nível | Configuração | Original | Hard | Dev |",
             "| --- | --- | ---: | ---: | ---: |"]
    for level, configs in manifest["summary"].items():
        for config, counts in configs.items():
            lines.append(f"| {level} | {config} | {counts['original_test']} | {counts['test']} | {counts['dev']} |")
    lines += ["", "Configurações com zero questões hard são ausentes da exportação; sua nota é indefinida.",
              "", "## Scores dos modelos", "", "Percentuais micro por nível, com intervalos de Wilson de 95%.",
              "Os modelos marcados com **painel** determinaram a seleção.", "",
              "| Modelo | Papel | Nível | Shots | N | Acurácia | IC 95% |",
              "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    aggregate = metrics[(metrics["config"] == "_all") & (metrics["aggregation"] == "micro")]
    for row in aggregate.to_dict("records"):
        role = "**painel**" if row["model"] in manifest["panel"] else "avaliado"
        lines.append(f"| {row['model']} | {role} | {row['level']} | {row['shots']} | {row['n']} | "
                     f"{100 * row['accuracy']:.2f}% | [{100 * row['ci_low']:.2f}, {100 * row['ci_high']:.2f}]% |")
    lines += ["", "scores.csv inclui os scores por área e macroacurácia, com quantidade de áreas presentes.",
              "O IC macro usa 2.000 reamostragens dentro de cada área (seed 42), com pesos iguais entre áreas.",
              "", "## Média entre 0-shot e 5-shot", "",
              "Acurácia micro média por nível. IC de 95% por bootstrap de questões, preservando o par de protocolos (2.000 reamostragens, seed 42).",
              "mean_scores.csv mantém os valores sem arredondamento e os scores separados de cada protocolo.", "",
              "| Modelo | Papel | Nível | N questões | Média | IC 95% |",
              "| --- | --- | --- | ---: | ---: | --- |"]
    for row in mean_metrics.to_dict("records"):
        role = "**painel**" if row["model"] in manifest["panel"] else "avaliado"
        lines.append(f"| {row['model']} | {role} | {row['level']} | {row['n']} | "
                     f"{100 * row['accuracy_mean']:.2f}% | [{100 * row['ci_low']:.2f}, {100 * row['ci_high']:.2f}]% |")
    lines += [
              "", "## Distribuições", "", "| Nível | Dimensão | Valor | Original | Hard | % original | % hard |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in distributions:
        share = f"{100 * row['hard_share']:.2f}" if row["hard_share"] is not None else "ausente"
        value = str(row["value"]).replace("|", "\\|")
        lines.append(f"| {row['level']} | {row['dimension']} | {value} | {row['original_n']} | "
                     f"{row['hard_n']} | {100 * row['original_share']:.2f} | {share} |")
    lines += ["", "## Sensibilidade ao painel", "",
              "Retirar um selecionador e remover apenas os itens acertados em ambos os protocolos pelos dois restantes (quatro acertos).",
              "Medir o modelo retirado em 0-shot, 5-shot e na média (mean). Cada retirada produz um subconjunto distinto; estes resultados não formam um ranking comum.",
              "O IC da média usa bootstrap pareado por questão; os ICs por protocolo usam Wilson. Essa análise não altera o dataset principal.", "",
              "| Modelo retirado | Nível | Protocolo | N | Acurácia | IC 95% |",
              "| --- | --- | ---: | ---: | ---: | --- |"]
    for row in sensitivity.to_dict("records"):
        lines.append(f"| {row['model']} | {row['level']} | {row['shots']} | {row['n']} | "
                     f"{100 * row['accuracy']:.2f}% | [{100 * row['ci_low']:.2f}, {100 * row['ci_high']:.2f}]% |")
    lines += ["", "## Interpretação e referências", "",
              "Este subconjunto remove questões consensualmente fáceis segundo os resultados registrados no experimento. "
              "A seleção incorpora o comportamento da métrica existente. Os scores dos selecionadores "
              "são condicionados aos acertos usados na construção em ambos os protocolos. A média não elimina esse condicionamento; "
              "a filtragem pode alterar rankings e favorecer outros modelos no recorte. "
              "Os intervalos são condicionais ao subconjunto fixo e não corrigem o viés de seleção. "
              "Áreas com poucas questões devem ser interpretadas junto de seus tamanhos e intervalos; "
              "o bootstrap tem limitações em áreas pequenas ou com todos os acertos/erros. "
              "As tabelas das alternativas pela média 0/5-shot estão documentadas em ablations/hard/.", "",
              "- [ARC-Challenge](https://arxiv.org/abs/1803.05457): seleção por falhas de sistemas de referência.",
              "- [MMLU-Pro](https://arxiv.org/html/2406.01574v6): filtragem por consenso de modelos.",
              "- [Phang et al., 2022](https://aclanthology.org/2022.dadc-1.8/): "
              "dependência dos rankings em relação ao adversário.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
