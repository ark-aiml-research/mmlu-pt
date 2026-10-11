"""Compare removal of easy items using recorded 5-shot scores only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mmlu_pt.hard import (
    DEFAULT_RESULTS, DEFAULT_SOURCE, LEVELS, PANEL, SHOTS,
    file_hash, load_scores, load_sources,
)
from mmlu_pt.hard_reporting import build_distributions, build_metrics

OUT = Path("output/easy-filter-analysis-v1")
COMPACT = (
    "Polygl0t/Tucano2-qwen-0.5B-Instruct",
    "Qwen/Qwen3.5-0.8B",
    "google/gemma-4-E2B-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
)


def family(model: str) -> str:
    if model.startswith("gemini-"):
        return "gemini"
    return model.split("/")[0]


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    OUT = parser.parse_args().output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    _, rows, index = load_sources(DEFAULT_SOURCE)
    scores, files = load_scores(DEFAULT_RESULTS, rows, index)
    models = list(scores)
    assert len(models) == 21
    five = np.column_stack([scores[model][:, 1] for model in models])
    total = five.sum(axis=1)
    compact = np.column_stack([scores[model][:, 1] for model in COMPACT]).sum(axis=1)
    reference = np.column_stack([scores[model][:, 1] for model in PANEL]).sum(axis=1)
    families = sorted({family(model) for model in models})
    # Integer weights avoid floating-point ambiguity at the 80% boundary.
    groups = [[i for i, model in enumerate(models) if family(model) == name] for name in families]
    common = int(np.lcm.reduce([len(group) for group in groups]))
    weighted = sum(five[:, group].sum(axis=1) * (common // len(group)) for group in groups)
    variants = {
        "original": (np.ones(len(rows), dtype=bool), "Sem remoção", []),
        "all21": (total < 21, "Remover acerto de 21/21 modelos em 5-shot", models),
        "atleast19": (total < 19, "Remover acerto de pelo menos 19/21 modelos em 5-shot", models),
        "atleast17": (total < 17, "Remover acerto de pelo menos 17/21 modelos em 5-shot", models),
        "families80": (weighted * 5 < 4 * common * len(groups),
                       "Remover média de acerto >=80%, com peso igual por grupo de modelos", models),
        "compact4": (compact < 4, "Remover acerto de pelo menos 4/5 modelos compactos em 5-shot", list(COMPACT)),
        "compact3": (compact < 3, "Remover acerto de pelo menos 3/5 modelos compactos em 5-shot", list(COMPACT)),
        "reference3": (reference < 3, "Remover acerto dos 3/3 modelos de referência em 5-shot", list(PANEL)),
    }
    metadata = {
        "experiment": str(DEFAULT_RESULTS), "source": str(DEFAULT_SOURCE),
        "selection_shots": 5, "evaluation_shots": list(SHOTS), "models": models,
        "compact_panel": list(COMPACT), "reference_panel": list(PANEL),
        "groups": {name: [models[i] for i in group] for name, group in zip(families, groups)},
        "group_caveat": "Grupos operacionais por série/fornecedor, não famílias independentes: Tucano2-qwen compartilha ancestralidade com Qwen.",
        "variants": {name: {"rule": rule, "selectors": selectors}
                     for name, (_, rule, selectors) in variants.items()},
        "prediction_hashes": {str(p.relative_to(DEFAULT_RESULTS)): file_hash(p) for p in files},
        "source_hashes": {str(p.relative_to(DEFAULT_SOURCE)): file_hash(p)
                          for p in sorted(DEFAULT_SOURCE.rglob("*")) if p.is_file()},
        "script_sha256": file_hash(Path(__file__)),
        "statistics": "Micro Wilson 95%; macro bootstrap estratificado 2000, seed 42; IC condicional ao subconjunto.",
    }
    (OUT / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    selections = pd.DataFrame([{key: row[key] for key in ("id", "academic_level", "config")} for row in rows])
    summaries, metrics, distributions = [], [], []
    for name, (keep, rule, selectors) in variants.items():
        print(f"Computing {name}: {keep.sum()} items", flush=True)
        selections[name] = keep
        result = build_metrics(rows, keep, scores, SHOTS)
        result.insert(0, "variant", name)
        result["selector"] = result.model.isin(selectors)
        metrics.append(result)
        distributions.extend({"variant": name, **record} for record in build_distributions(rows, keep))
        for level in LEVELS:
            mask = np.array([row["academic_level"] == level for row in rows])
            retained = keep & mask
            summaries.append({"variant": name, "rule": rule, "level": level,
                              "original_n": int(mask.sum()), "retained_n": int(retained.sum()),
                              "removed_n": int((mask & ~keep).sum()), "removed_fraction": float((mask & ~keep).sum() / mask.sum()),
                              "uniform_random_accuracy": float(np.mean([1 / len(row["choices"]) for row, yes in zip(rows, retained) if yes]))})
    all_metrics = pd.concat(metrics, ignore_index=True)
    summary = pd.DataFrame(summaries)
    all_metrics.to_csv(OUT / "scores.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(distributions).to_csv(OUT / "distributions.csv", index=False)
    selections.to_csv(OUT / "selection.csv", index=False)
    np.savez_compressed(OUT / "recorded_scores.npz", **scores)

    keys = ["model", "shots", "level", "aggregation", "config"]
    assert np.all(variants["atleast17"][0] <= variants["atleast19"][0])
    assert np.all(variants["atleast19"][0] <= variants["all21"][0])
    assert np.all(variants["compact3"][0] <= variants["compact4"][0])
    aggregate = all_metrics[(all_metrics.config == "_all") & (all_metrics.aggregation == "micro")]
    changes = aggregate.merge(aggregate[aggregate.variant == "original"][keys[:3] + ["accuracy"]],
                              on=keys[:3], suffixes=("", "_original"))
    changes["drop_pp"] = 100 * (changes.accuracy_original - changes.accuracy)
    changes.to_csv(OUT / "comparison.csv", index=False)

    lines = ["# Remoção de questões fáceis — simulação", "",
             "Seleção exclusivamente por extractive_match registrado em 5-shot; avaliação em 0-shot e 5-shot.",
             "Nenhuma inferência nova. Não há exportação de um novo dataset.", "",
             "As regras abaixo são alternativas exploratórias; consultar seus scores para escolher uma regra torna a decisão adaptativa.",
             "Remover itens acertados ainda condiciona os scores dos selecionadores. Os intervalos não corrigem esse viés.", "",
             "## Regras", ""]
    lines += [f"- **{name}**: {rule}." for name, (_, rule, _) in variants.items()]
    lines += ["", "Painel compacto (fixado por identificadores, sem ordenação por scores): " + "; ".join(COMPACT) + ".",
              "Painel de referência: " + "; ".join(PANEL) + ".",
              "Grupos para families80: " + "; ".join(f"{f} ({len(g)})" for f, g in zip(families, groups)) + ".",
              metadata["group_caveat"], "", "## Composição", "",
              "| Regra | Nível | Mantidas | Removidas | % removida | Acerto ao acaso (%) |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in summaries:
        lines.append(f"| {r['variant']} | {r['level']} | {r['retained_n']} | {r['removed_n']} | {100*r['removed_fraction']:.2f} | {100*r['uniform_random_accuracy']:.2f} |")
    for level in LEVELS:
        for shot in SHOTS:
            lines += ["", f"## {level} — {shot}-shot", "", "Acurácia micro (%).", "",
                      "| Modelo | " + " | ".join(variants) + " |",
                      "| --- | " + " | ".join("---:" for _ in variants) + " |"]
            part = aggregate[(aggregate.level == level) & (aggregate.shots == shot)]
            pivot = part.pivot(index="model", columns="variant", values="accuracy")
            for model in models:
                lines.append("| " + model + " | " + " | ".join(f"{100*pivot.loc[model, name]:.2f}" for name in variants) + " |")
    lines += ["", "## Interpretação", "",
              "all21 é a remoção mais conservadora por consenso total. atleast19 e atleast17 relaxam o consenso, aumentando a remoção.",
              "families80 reduz o peso de grupos com muitos checkpoints, mas não elimina parentesco ou dependência entre modelos.",
              "compact4 e compact3 usam capacidade de modelos compactos como sinal de facilidade; os três selecionadores do painel de referência não participam diretamente desse filtro, mas há parentesco de famílias.",
              "reference3 preserva também itens intermediários: basta um erro do painel para manter uma questão.",
              "Baixa acurácia dos selecionadores não é evidência independente de dificuldade. Não escolher limiares para produzir um ranking ou score desejado.",
              "Micro e macro por área e respectivos intervalos estão em scores.csv; comparações com o original em comparison.csv; distribuições em distributions.csv; IDs e decisões em selection.csv.",
              "O bootstrap macro tem limitações em áreas pequenas ou com todos os acertos/erros; intervalos individuais não são testes pareados entre modelos.", "",
              "[MMLU-Pro](https://arxiv.org/html/2406.01574v6): sua filtragem inicial remove questões acertadas por mais de quatro de oito modelos. As regras aqui são adaptações, sem expansão de alternativas ou outras etapas do MMLU-Pro.", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print(summary.to_string(index=False), flush=True)
    print("Verified: nested filtering masks consistent.", flush=True)


if __name__ == "__main__":
    main()
