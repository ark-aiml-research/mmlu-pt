"""Publish the recorded ablation analyses as reproducible CSV and Markdown tables."""

import json
from pathlib import Path

import pandas as pd

from mmlu_pt.hard import file_hash

SOURCE = Path("output/easy-filter-mean-shots-v1")
TARGET = Path("ablations/hard")


def main() -> None:
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    counts = pd.read_csv(SOURCE / "summary.csv")
    counts["removed_n"] = counts.original_n - counts.retained_n
    counts = counts[["variant", "level", "original_n", "retained_n", "removed_n", "removed_fraction", "uniform_random_accuracy"]]
    scores = pd.read_csv(SOURCE / "mean_scores.csv")
    scores = scores[["variant", "selector", "model", "level", "n_items", "accuracy_0shot", "accuracy_5shot", "accuracy_mean", "ci_low", "ci_high"]]
    counts.to_csv(TARGET / "counts.csv", index=False)
    scores.to_csv(TARGET / "scores.csv", index=False)
    provenance = {**manifest, "source_analysis": str(SOURCE.resolve()),
                  "analysis_files_sha256": {name: file_hash(SOURCE / name) for name in ("manifest.json", "summary.csv", "mean_scores.csv")},
                  "tables_sha256": {name: file_hash(TARGET / name) for name in ("counts.csv", "scores.csv")},
                  "table_script_sha256": file_hash(Path(__file__))}
    (TARGET / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    variants = list(manifest["variants"])
    lines = ["# Alternativas de remoção de questões fáceis", "",
             "Resultados das regras exploratórias sobre a base deduplicada. A versão principal é MMLU-PT-Hard-Direct v2 (`reference3`).", "",
             "Para cada questão e modelo: média = (extractive_match_0shot + extractive_match_5shot)/2. Ambos os protocolos participam da seleção.", "",
             "Painel de referência: " + "; ".join(manifest["reference_panel"]) + ".",
             "Painel compacto: " + "; ".join(manifest["compact_panel"]) + ".", "",
             "Grupos com pesos iguais: " + "; ".join(f"{name}: {', '.join(models)}" for name, models in manifest["groups"].items()) + ".",
             manifest["group_caveat"], "", "## Regras", "",
             "| Identificador | Remover quando |", "| --- | --- |"]
    for name, specification in manifest["variants"].items():
        lines.append(f"| {name} | {specification['rule']} |")
    lines += ["", "## Contagens", "", "| Regra | Nível | Original | Mantidas | Removidas | % removida |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in counts.to_dict("records"):
        lines.append(f"| {row['variant']} | {row['level']} | {row['original_n']} | {row['retained_n']} | {row['removed_n']} | {100 * row['removed_fraction']:.2f} |")
    for level in ("high_school", "undergraduate"):
        for field, label in (("accuracy_mean", "média 0/5-shot"), ("accuracy_0shot", "0-shot"), ("accuracy_5shot", "5-shot")):
            lines += ["", f"## {level} — {label}", "", "Acurácia micro (%).", "",
                      "| Modelo | " + " | ".join(variants) + " |", "| --- | " + " | ".join("---:" for _ in variants) + " |"]
            pivot = scores[scores.level == level].pivot(index="model", columns="variant", values=field)
            for model in manifest["models"]:
                lines.append("| " + model + " | " + " | ".join(f"{100 * pivot.loc[model, name]:.2f}" for name in variants) + " |")
    lines += ["", "## Reprodução e interpretação", "",
              "Executar na raiz do projeto:", "", "```bash",
              "python ablations/hard/analyze_5shot.py", "python ablations/hard/analyze_mean.py",
              "python ablations/hard/update_tables.py", "```", "",
              "`counts.csv` registra as contagens por nível; `scores.csv` contém N, scores por protocolo, média, papel na seleção e IC de 95% da média.",
              "`provenance.json` registra painéis, limiares e hashes das análises, predições, snapshot e tabelas.",
              "ICs da média usam bootstrap pareado por questão (2.000 reamostragens, seed 42). São condicionais ao subconjunto fixo e não corrigem viés de seleção ou dependência entre questões da mesma prova.",
              "As regras são exploratórias; escolher um limiar após observar scores torna a decisão adaptativa. Os scores dos selecionadores não são uma medida independente de dificuldade.",
              "A metodologia principal está em [docs/hard_direct_methodology.md](../../docs/hard_direct_methodology.md).", ""]
    (TARGET / "README.md").write_text("\n".join(lines))
    print("Updated ablation tables and provenance", flush=True)


if __name__ == "__main__":
    main()
