"""Remove easy items using mean recorded accuracy across 0-shot and 5-shot."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mmlu_pt.hard import DEFAULT_RESULTS, DEFAULT_SOURCE, LEVELS, PANEL, SHOTS, file_hash, load_sources
from mmlu_pt.hard_reporting import build_distributions, build_metrics

OUT = Path("output/easy-filter-mean-shots-v1")
PREVIOUS = Path("output/easy-filter-analysis-v1")


def mean_metric(model: str, level: str, matrix: np.ndarray) -> dict:
    """Bootstrap paired item means, preserving dependence between shot protocols."""
    totals = matrix.sum(axis=1).astype(int)
    counts = np.bincount(totals, minlength=3)
    rng = np.random.default_rng(42)
    draws = rng.multinomial(len(totals), counts / len(totals), size=2000)
    means = (draws[:, 1] + 2 * draws[:, 2]) / (2 * len(totals))
    low, high = np.quantile(means, [0.025, 0.975])
    return {"model": model, "level": level, "n_items": len(totals),
            "accuracy_0shot": matrix[:, 0].mean(), "accuracy_5shot": matrix[:, 1].mean(),
            "accuracy_mean": matrix.mean(), "ci_low": low, "ci_high": high}


def main() -> None:
    global OUT, PREVIOUS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--previous-dir", type=Path, default=PREVIOUS)
    args = parser.parse_args()
    OUT, PREVIOUS = args.output_dir, args.previous_dir
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PREVIOUS / "manifest.json").read_text())
    for key, root in (("prediction_hashes", DEFAULT_RESULTS), ("source_hashes", DEFAULT_SOURCE)):
        for relative, expected in manifest[key].items():
            assert file_hash(root / relative) == expected, relative
    _, rows, _ = load_sources(DEFAULT_SOURCE)
    previous_selection = pd.read_csv(PREVIOUS / "selection.csv")
    identity = pd.DataFrame([{key: row[key] for key in ("id", "academic_level", "config")} for row in rows])
    pd.testing.assert_frame_equal(identity, previous_selection[identity.columns])
    with np.load(PREVIOUS / "recorded_scores.npz") as data:
        scores = {model: data[model] for model in manifest["models"]}
    assert all(values.shape == (len(rows), 2) and np.isin(values, [0, 1]).all() for values in scores.values())
    models = list(scores)
    compact_models = manifest["compact_panel"]
    # Twice the item mean is an integer in {0, 1, 2}; compare without rounding.
    paired = np.column_stack([scores[model].sum(axis=1) for model in models])
    total = paired.sum(axis=1)
    compact = sum(scores[model].sum(axis=1) for model in compact_models)
    reference = sum(scores[model].sum(axis=1) for model in PANEL)
    groups = manifest["groups"]
    common = int(np.lcm.reduce([len(group) for group in groups.values()]))
    weighted = sum(sum(scores[model].sum(axis=1) for model in group) * (common // len(group))
                   for group in groups.values())
    variants = {
        "original": (np.ones(len(rows), dtype=bool), "Sem remoção", []),
        "all21": (total < 42, "Remover soma das médias =21: 42/42 acertos", models),
        "atleast19": (total < 38, "Remover soma das médias >=19: >=38/42 acertos (90,48%)", models),
        "atleast17": (total < 34, "Remover soma das médias >=17: >=34/42 acertos (80,95%)", models),
        "families80": (weighted * 5 < 8 * common * len(groups),
                       "Remover média >=80% com peso igual por grupo", models),
        "compact4": (compact < 8, "Remover soma das médias dos compactos >=4: >=8/10 acertos", compact_models),
        "compact3": (compact < 6, "Remover soma das médias dos compactos >=3: >=6/10 acertos", compact_models),
        "reference3": (reference < 6, "Remover soma das médias do painel =3: 6/6 acertos", list(PANEL)),
    }
    selections = identity.copy()
    selections["sum_model_means"] = total / 2
    selections["sum_compact_means"] = compact / 2
    selections["sum_reference_means"] = reference / 2
    records, means, summaries, distributions = [], [], [], []
    for name, (keep, rule, selectors) in variants.items():
        print(f"Computing {name}: {keep.sum()} items", flush=True)
        selections[name] = keep
        metrics = build_metrics(rows, keep, scores, SHOTS)
        metrics.insert(0, "variant", name)
        metrics["selector"] = metrics.model.isin(selectors)
        records.append(metrics)
        distributions.extend({"variant": name, **r} for r in build_distributions(rows, keep))
        old_keep = previous_selection[name].to_numpy()
        for level in LEVELS:
            level_mask = np.array([r["academic_level"] == level for r in rows])
            mask = keep & level_mask
            summaries.append({"variant": name, "rule": rule, "level": level,
                              "original_n": int(level_mask.sum()), "retained_n": int(mask.sum()),
                              "removed_fraction": float(1 - mask.sum() / level_mask.sum()),
                              "previous_5shot_retained_n": int((old_keep & level_mask).sum()),
                              "added_vs_5shot": int((mask & ~old_keep).sum()),
                              "removed_vs_5shot": int((~keep & old_keep & level_mask).sum()),
                              "uniform_random_accuracy": float(np.mean([1 / len(r["choices"]) for r, yes in zip(rows, mask) if yes]))})
            for model in models:
                record = {"variant": name, "selector": model in selectors,
                          **mean_metric(model, level, scores[model][mask])}
                old = scores[model][old_keep & level_mask]
                record["previous_5shot_selection_mean"] = float(old.mean())
                record["delta_selection_pp"] = 100 * (record["accuracy_mean"] - old.mean())
                means.append(record)
    metrics = pd.concat(records, ignore_index=True)
    mean_scores = pd.DataFrame(means)
    summary = pd.DataFrame(summaries)
    old_scores = pd.read_csv(PREVIOUS / "scores.csv")
    keys = ["model", "shots", "level", "aggregation", "config"]
    for new, old in (("original", "original"),):
        pd.testing.assert_frame_equal(
            metrics[metrics.variant == new].drop(columns="variant").sort_values(keys).reset_index(drop=True),
            old_scores[old_scores.variant == old].drop(columns="variant").sort_values(keys).reset_index(drop=True),
            check_dtype=False, atol=1e-12, rtol=1e-12)
    for lower, upper in (("atleast17", "atleast19"), ("atleast19", "all21"),
                         ("compact3", "compact4")):
        assert np.all(variants[lower][0] <= variants[upper][0])
    assert np.allclose(mean_scores.accuracy_mean, (mean_scores.accuracy_0shot + mean_scores.accuracy_5shot) / 2)
    # Any failure in either protocol precludes unanimous success in both.
    for name in ("all21", "reference3"):
        assert np.all(previous_selection[name].to_numpy() <= variants[name][0])
    metrics.to_csv(OUT / "scores.csv", index=False)
    mean_scores.to_csv(OUT / "mean_scores.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    selections.to_csv(OUT / "selection.csv", index=False)
    pd.DataFrame(distributions).to_csv(OUT / "distributions.csv", index=False)
    metadata = {**manifest, "selection_shots": [0, 5], "selection_score": "(score_0shot + score_5shot) / 2 per item and model",
                "variants": {name: {"rule": rule, "selectors": selectors} for name, (_, rule, selectors) in variants.items()},
                "script_sha256": file_hash(Path(__file__)),
                "cache_sha256": file_hash(PREVIOUS / "recorded_scores.npz"),
                "cache_identity_sha256": file_hash(PREVIOUS / "selection.csv"),
                "mean_intervals": "Bootstrap por questão mantendo par 0/5-shot; 2000 reamostragens, seed 42; multinomial sobre médias 0, 0.5, 1. Não usar Wilson com 2N respostas independentes.",
                "checks": "Input hashes and IDs checked; original scores reproduced; nested masks and exact averaged accuracies verified."}
    (OUT / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Remoção de fáceis pela média de 0-shot e 5-shot", "",
             "Para cada questão e modelo: média=(extractive_match_0shot+extractive_match_5shot)/2, com valores 0, 0.5 ou 1.",
             "Limiar aplicado à soma dessas médias; não se exige que cada modelo acerte ambos os protocolos, exceto nas regras de unanimidade.",
             "Exemplo compact4: 8 acertos entre as 10 combinações modelo/protocolo; não necessariamente quatro modelos acertando os dois protocolos.",
             "Os limiares 19/21 e 17/21 são preservados exatamente, não arredondados para 90% e 80%.",
             "Nenhuma nova inferência ou exportação de dataset. Análise anterior preservada.", "", "## Regras", ""]
    lines += [f"- **{name}**: {rule}." for name, (_, rule, _) in variants.items()]
    lines += ["", "Compactos: " + "; ".join(compact_models), "Painel: " + "; ".join(PANEL),
              "Grupos com pesos iguais: " + "; ".join(groups), manifest["group_caveat"],
              "", "## Tamanhos e mudança em relação à seleção só por 5-shot", "",
              "| Regra | Nível | N anterior | N agora | % removida do original | Entraram | Saíram |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in summaries:
        lines.append(f"| {r['variant']} | {r['level']} | {r['previous_5shot_retained_n']} | {r['retained_n']} | {100*r['removed_fraction']:.2f} | {r['added_vs_5shot']} | {r['removed_vs_5shot']} |")
    for level in LEVELS:
        for field, label in (("accuracy_mean", "média 0/5-shot"), ("accuracy_0shot", "0-shot"), ("accuracy_5shot", "5-shot")):
            lines += ["", f"## {level} — {label}", "", "Acurácia micro (%).", "",
                      "| Modelo | " + " | ".join(variants) + " |", "| --- | " + " | ".join("---:" for _ in variants) + " |"]
            pivot = mean_scores[mean_scores.level == level].pivot(index="model", columns="variant", values=field)
            for model in models:
                lines.append("| " + model + " | " + " | ".join(f"{100*pivot.loc[model, name]:.2f}" for name in variants) + " |")
    lines += ["", "## Interpretação e estatística", "",
              "Ambos os protocolos participam da seleção. Isso reduz a dependência exclusiva de 5-shot, mas não elimina viés em relação aos modelos selecionadores; 0-shot também deixa de ser um protocolo não utilizado na seleção.",
              "Remover itens fáceis pela média preserva itens intermediários. A média entre protocolos não equivale a replicações independentes nem a várias seeds.",
              "Comparar a coluna accuracy_mean entre seleções para isolar a mudança do filtro. Comparar média atual a 5-shot anterior mistura alteração do filtro com alteração da métrica.",
              "mean_scores.csv contém resultados por nível, média dos protocolos e IC por bootstrap pareado por questão (2000, seed 42). A reamostragem multinomial das contagens de médias 0/0.5/1 equivale à reamostragem dos pares por questão para essa estatística.",
              "scores.csv mantém resultados por protocolo e área, Wilson micro e bootstrap macro (2000, seed 42). Intervalos são condicionais aos subconjuntos e não corrigem seleção ou dependência entre questões da mesma prova.",
              "Tamanhos estão em summary.csv; decisões e somas por questão em selection.csv; composição por área/disciplina/exame em distributions.csv.", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print(summary.to_string(index=False), flush=True)
    print("Verified: original scores, hashes, IDs, thresholds and paired means.", flush=True)


if __name__ == "__main__":
    main()
