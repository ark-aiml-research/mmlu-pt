"""Generate the figures and verification summary for the Hard-subset appendix.

Reads the build artifacts in output/hard-direct-v2, the rule comparison in ablations/hard and the
recorded predictions of the selection panel. Any inconsistency between them raises ValueError.
No inference is run. Run with .venv/bin/python from the repo root.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from mmlu_pt.hard import DEFAULT_RESULTS, LEVELS, PANEL, SHOTS, file_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = Path(__file__).resolve().parent
HARD_DIR = REPO_ROOT / "output" / "hard-direct-v2"
# The filtered experiment keeps only details; the run configurations are in the pre-correction snapshot.
RUN_CONFIG_DIR = Path("/raid/aluno_paulosantana/repositories/light-benchmark/backups/"
                      "dataset-correction-20261010-170232/snapshot/output/exp-mmlu-pt-direct/results")

ORIGINAL_TEST = 37798
GENERATION_SIZE = 64
RULES = ["all21", "atleast19", "atleast17", "families80", "compact4", "compact3", "reference3"]
RULE_LABELS = {"all21": "21/21 models", "atleast19": "$\\geq$19/21 models", "atleast17": "$\\geq$17/21 models",
               "families80": "$\\geq$80%, by series", "compact4": "$\\geq$4/5 compact",
               "compact3": "$\\geq$3/5 compact", "reference3": "3/3 panel (adopted)"}
# A compliant response contains the requested answer line or is a bare letter.
ANSWER_LINE = re.compile(r"resposta\W{0,6}\(?[A-E]\b", re.IGNORECASE)
BARE_LETTER = re.compile(r"\W*\(?[A-E]\)?\W*")
LEVEL_LABELS = {"high_school": "High school", "undergraduate": "Undergraduate"}
MACRO_AREA_LABELS = {
    "Computing, Engineering, Architecture, and Design": "Computing, Eng., Arch., Design",
    "Economics, Business, and Accounting": "Economics, Business, Accounting",
    "Health Sciences": "Health Sciences",
    "Humanities": "Humanities",
    "Languages, Literature, Arts, and Communication": "Languages, Lit., Arts, Comm.",
    "Law": "Law",
    "Mathematics, Statistics, and Logic": "Mathematics, Statistics, Logic",
    "Natural, Earth, Agricultural and Veterinary Sciences": "Natural, Earth, Agric., Vet. Sci.",
    "Social and Applied Social Sciences": "Social and Applied Social Sci.",
}

# ACL page geometry: 7.7 cm columns.
COLUMN_WIDTH = 7.7 / 2.54
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
LIGHT = "#b5b4ad"
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def short_name(model: str) -> str:
    name = model.split("/")[-1].removesuffix("-Instruct").removesuffix("-BF16").removesuffix("-it")
    return name.removeprefix("NVIDIA-")


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def print_frame(frame: pd.DataFrame) -> None:
    print(frame.to_string(index=False), end="\n\n")


# -----------------------------------------------------------------------------
# Loading and verification


def load_audit() -> pd.DataFrame:
    """One row per original test question with the six panel scores and the build decision."""
    records = []
    with (HARD_DIR / "audit.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record = {key: row[key] for key in ("id", "academic_level", "config", "correct", "removed", "selected")}
            for model in PANEL:
                record[f"{model}|0"] = row["scores_0shot"][model]
                record[f"{model}|5"] = row["scores_5shot"][model]
            records.append(record)
    return pd.DataFrame(records)


def check_artifacts(audit: pd.DataFrame, manifest: dict, distributions: pd.DataFrame, counts: pd.DataFrame,
                    rule_scores: pd.DataFrame, mean_scores: pd.DataFrame, provenance: dict,
                    publication: dict) -> None:
    if len(audit) != ORIGINAL_TEST or audit.id.duplicated().any():
        raise ValueError("The audit must list every original test question once.")
    if not audit.removed.eq(audit.correct == 2 * len(PANEL)).all() or audit.removed.eq(audit.selected).any():
        raise ValueError("The audit decisions differ from the six-correct rule.")
    if manifest["panel"] != list(PANEL) or manifest["selection"]["shots"] != list(SHOTS):
        raise ValueError("The manifest panel or protocols differ from mmlu_pt.hard.")
    if manifest["original_test"] != len(audit) or manifest["selected_test"] != int(audit.selected.sum()):
        raise ValueError("The manifest totals differ from the audit.")
    by_config = audit.groupby(["academic_level", "config"]).selected.agg(["size", "sum"])
    for level, areas in manifest["summary"].items():
        for area, summary in areas.items():
            if (summary["original_test"], summary["test"]) != tuple(by_config.loc[(level, area)]):
                raise ValueError(f"Manifest counts differ from the audit: {level}/{area}")
    configs = distributions[distributions.dimension == "config"].set_index(["level", "value"])
    if not (configs.original_n.equals(by_config["size"].rename_axis(["level", "value"]).rename("original_n"))
            and configs.hard_n.equals(by_config["sum"].rename_axis(["level", "value"]).rename("hard_n"))):
        raise ValueError("distributions.json differs from the audit.")
    adopted = counts[counts.variant == "reference3"].set_index("level")
    for level in LEVELS:
        level_rows = audit[audit.academic_level == level]
        if (adopted.loc[level, "original_n"], adopted.loc[level, "retained_n"]) != (len(level_rows),
                                                                                   level_rows.selected.sum()):
            raise ValueError(f"counts.csv reference3 differs from the build: {level}")
    joined = rule_scores[rule_scores.variant == "reference3"].merge(mean_scores, on=["model", "level"])
    if len(joined) != len(mean_scores) or not np.allclose(joined.accuracy_mean_x, joined.accuracy_mean_y,
                                                          rtol=0, atol=1e-12):
        raise ValueError("ablations/hard/scores.csv reference3 differs from the build mean scores.")
    for name, expected in provenance["tables_sha256"].items():
        if file_hash(ABLATION_DIR / name) != expected:
            raise ValueError(f"{name} differs from provenance.json")
    if publication["checks"]["test_rows"] != manifest["selected_test"] or publication["status"] != "published":
        raise ValueError("publication.json differs from the build.")


def load_panel_errors(audit: pd.DataFrame) -> pd.DataFrame:
    """Every panel error with whether the response follows the answer format and has an extracted choice."""
    known = set(zip(audit.academic_level, audit.id))
    records = []
    for model in PANEL:
        files = sorted((DEFAULT_RESULTS / "details" / model).rglob("*.parquet"))
        if len(files) != 2 * (6 + 9):
            raise ValueError(f"Unexpected number of detail files for {model}: {len(files)}")
        for path in files:
            level = "high_school" if "-high-school-" in path.name else "undergraduate"
            shot = int(path.name.split("|")[1].split("_")[0])
            columns = ["doc.specific.id", "doc.specific.extracted_predictions", "doc.generation_size",
                       "metric.extractive_match", "model_response.text"]
            for row in pq.read_table(path, columns=columns).to_pylist():
                if row["generation_size"] != GENERATION_SIZE:
                    raise ValueError(f"Unexpected generation size in {path}")
                if (level, row["id"]) not in known:
                    raise ValueError(f"Unknown prediction ID in {path}")
                if row["extractive_match"] == 1:
                    continue
                text = row["text"][0] if row["text"] else ""
                compliant = bool(ANSWER_LINE.search(text) or BARE_LETTER.fullmatch(text))
                records.append({"academic_level": level, "id": row["id"], "model": model, "shot": shot,
                                "compliant": compliant, "extracted": bool(row["extracted_predictions"])})
    errors = pd.DataFrame(records)
    expected = sum(int((audit[f"{model}|{shot}"] == 0).sum()) for model in PANEL for shot in SHOTS)
    if len(errors) != expected:
        raise ValueError("Panel errors in the details differ from the audit.")
    return errors


def check_run_configs(models: list[str]) -> dict:
    """Generation settings recorded for every evaluated model."""
    settings = {}
    for model in models:
        files = sorted((RUN_CONFIG_DIR / model).glob("results_*.json"))
        if len(files) != 1:
            raise ValueError(f"Expected one results file for {model}, found {len(files)}")
        config = read_json(files[0])["config_general"]["model_config"]
        if config["generation_parameters"]["temperature"] != 0:
            raise ValueError(f"{model} was not decoded greedily")
        settings[model] = config.get("extra_body")
    return settings


# -----------------------------------------------------------------------------
# Printed summary


def print_rules(counts: pd.DataFrame, provenance: dict) -> None:
    print("== Alternative removal rules (selection on the mean of 0-shot and 5-shot)")
    for rule in RULES:
        print(f"{rule}: {provenance['variants'][rule]['rule']} | selectors: "
              f"{len(provenance['variants'][rule]['selectors'])}")
    table = counts[counts.variant.isin(RULES)].copy()
    table["removed_pct"] = (100 * table.removed_fraction).round(2)
    table["retained_pct"] = (100 * table.retained_n / table.original_n).round(2)
    print_frame(table[["variant", "level", "original_n", "retained_n", "removed_n", "removed_pct", "retained_pct"]])
    totals = table.groupby("variant", sort=False)[["original_n", "retained_n"]].sum()
    totals["removed_pct"] = (100 * (1 - totals.retained_n / totals.original_n)).round(2)
    print(totals.to_string(), end="\n\n")


def print_protocols(audit: pd.DataFrame) -> None:
    print("== Panel unanimity per protocol (all three panel models correct)")
    for level in LEVELS:
        rows = audit[audit.academic_level == level]
        zero = np.all([rows[f"{model}|0"] == 1 for model in PANEL], axis=0)
        five = np.all([rows[f"{model}|5"] == 1 for model in PANEL], axis=0)
        print(f"{level}: n={len(rows)} unanimous 0-shot={zero.sum()} 5-shot={five.sum()} both={np.sum(zero & five)} "
              f"5-shot only={np.sum(five & ~zero)} 0-shot only={np.sum(zero & ~five)} "
              f"either={np.sum(zero | five)}")
    print()


def print_drivers(audit: pd.DataFrame) -> None:
    print("== What keeps a question (retained questions only)")
    for level in LEVELS:
        kept = audit[(audit.academic_level == level) & audit.selected]
        print(f"{level}: retained={len(kept)}")
        print("  correct answers out of 6:", kept.correct.value_counts().sort_index().to_dict())
        bins = pd.cut(kept.correct, [-1, 0, 2, 4, 5], labels=["0", "1-2", "3-4", "5"])
        print("  binned:", bins.value_counts().sort_index().to_dict())
        failing = pd.DataFrame({model: (kept[f"{model}|0"] == 0) | (kept[f"{model}|5"] == 0) for model in PANEL})
        print("  panel members with at least one error:", failing.sum(axis=1).value_counts().sort_index().to_dict())
        for model in PANEL:
            alone = int((failing[model] & (failing.sum(axis=1) == 1)).sum())
            solved = int((~failing[model]).sum())
            print(f"  {model}: correct in both protocols={solved} ({percent(solved / len(kept))}), "
                  f"only failing member={alone} ({percent(alone / len(kept))})")
    print("retained with 0/6 correct, both levels:", int((audit.selected & (audit.correct == 0)).sum()), end="\n\n")


def print_non_answers(audit: pd.DataFrame, errors: pd.DataFrame) -> None:
    print("== Panel errors without the requested answer format")
    summary = errors.groupby(["model", "shot", "academic_level"]).agg(
        errors=("compliant", "size"), no_answer_line=("compliant", lambda s: int((~s).sum())),
        no_extracted_choice=("extracted", lambda s: int((~s).sum()))).reset_index()
    print_frame(summary)
    flags = errors.groupby(["academic_level", "id"]).agg(all_non_compliant=("compliant", lambda s: not s.any()),
                                                         all_unextracted=("extracted", lambda s: not s.any()))
    kept = audit[audit.selected].set_index(["academic_level", "id"]).join(flags)
    for level in LEVELS:
        rows = kept.loc[level]
        non_compliant, unextracted = int(rows.all_non_compliant.sum()), int(rows.all_unextracted.sum())
        print(f"{level}: retained={len(rows)}; every panel error lacks the answer line={non_compliant} "
              f"({percent(non_compliant / len(rows))}); every panel error has no extracted choice="
              f"{unextracted} ({percent(unextracted / len(rows))})")
    total = len(kept)
    print(f"total: lacks answer line={int(kept.all_non_compliant.sum())} "
          f"({percent(kept.all_non_compliant.sum() / total)}); no extracted choice="
          f"{int(kept.all_unextracted.sum())} ({percent(kept.all_unextracted.sum() / total)})\n")


def print_composition(distributions: pd.DataFrame) -> None:
    print("== Composition, original vs Hard (shares within level, %)")
    for dimension in ("macro_area", "exam"):
        table = distributions[distributions.dimension == dimension].copy()
        table["kept_pct"] = (100 * table.hard_n / table.original_n).round(1)
        table["original_share"] = (100 * table.original_share).round(2)
        table["hard_share"] = (100 * table.hard_share).round(2)
        print_frame(table[["level", "value", "original_n", "hard_n", "kept_pct", "original_share", "hard_share"]])
        change = (table.hard_share - table.original_share).abs().groupby(table.level).max()
        print(f"largest absolute {dimension} share change (pp):", change.round(2).to_dict(), end="\n\n")
    subjects = distributions[distributions.dimension == "subject"].copy()
    subjects["kept_pct"] = (100 * subjects.hard_n / subjects.original_n).round(1)
    print("Subjects with fewer than 10 Hard questions:")
    print_frame(subjects[subjects.hard_n < 10][["level", "value", "original_n", "hard_n"]])
    print("Subjects with the highest and lowest kept share (original_n >= 100):")
    large = subjects[subjects.original_n >= 100].sort_values("kept_pct")
    print_frame(pd.concat([large.head(5), large.tail(5)])[["level", "value", "original_n", "hard_n", "kept_pct"]])


def print_scores(rule_scores: pd.DataFrame, counts: pd.DataFrame) -> None:
    print("== Mean 0/5-shot accuracy, original vs Hard (%)")
    original = rule_scores[rule_scores.variant == "original"].set_index(["level", "model"])
    hard = rule_scores[rule_scores.variant == "reference3"].set_index(["level", "model"])
    table = pd.DataFrame({"original": 100 * original.accuracy_mean, "hard": 100 * hard.accuracy_mean,
                          "hard_0shot": 100 * hard.accuracy_0shot, "hard_5shot": 100 * hard.accuracy_5shot,
                          "ci_low": 100 * hard.ci_low, "ci_high": 100 * hard.ci_high}).round(2)
    table["drop_pp"] = (table.original - table.hard).round(2)
    table["panel"] = [model in PANEL for _, model in table.index]
    print(table.reset_index().sort_values(["level", "hard"], ascending=[True, False]).to_string(index=False))
    half_width = np.maximum(table.hard - table.ci_low, table.ci_high - table.hard).groupby(level="level").max()
    print("largest 95% CI half-width of the Hard means (pp):", half_width.round(2).to_dict(), end="\n\n")
    strongest = rule_scores[rule_scores.model == "gemini-2.5-flash"].pivot(index="variant", columns="level",
                                                                         values="accuracy_mean")
    print("gemini-2.5-flash mean accuracy per rule (%):")
    print((100 * strongest).round(2).to_string(), end="\n\n")
    chance = counts[counts.variant.isin(["original", "reference3"])].set_index(["variant", "level"])
    for level in LEVELS:
        rows = table.loc[level]
        base = 100 * chance.loc[("reference3", level), "uniform_random_accuracy"]
        outside = rows[~rows.panel]
        near = outside[outside.hard - base < 5]
        print(f"{level}: chance original={100 * chance.loc[('original', level), 'uniform_random_accuracy']:.2f} "
              f"hard={base:.2f}; non-panel models within 5 pp of chance on Hard: {len(near)}/{len(outside)}; "
              f"non-panel Hard range {outside.hard.min():.2f}-{outside.hard.max():.2f}; "
              f"original range {outside.original.min():.2f}-{outside.original.max():.2f}")
        spearman = outside.original.rank().corr(outside.hard.rank())
        print(f"  Spearman rank correlation original vs Hard, {len(outside)} non-panel models: {spearman:.3f}")
        rank_original = outside.original.rank(ascending=False, method="min")
        rank_hard = outside.hard.rank(ascending=False, method="min")
        moved = (rank_original - rank_hard).sort_values()
        print("  largest rank changes (positive = moves up on Hard):",
              {short_name(model): int(delta) for model, delta in pd.concat([moved.head(3), moved.tail(3)]).items()})
    print()


def print_sensitivity(sensitivity: pd.DataFrame, mean_scores: pd.DataFrame) -> None:
    print("== Leave-one-selector-out (subset built by the two other panel models; held-out model scored)")
    table = sensitivity[(sensitivity.aggregation == "micro") & (sensitivity.config == "_all")].copy()
    table["accuracy"] = (100 * table.accuracy).round(2)
    table["ci"] = [f"[{100 * low:.2f}, {100 * high:.2f}]" for low, high in zip(table.ci_low, table.ci_high)]
    print_frame(table[["model", "shots", "level", "n", "accuracy", "ci"]])
    means = table[table.shots == "mean"]
    half_width = 100 * np.maximum(means.accuracy / 100 - means.ci_low, means.ci_high - means.accuracy / 100)
    print("largest 95% CI half-width of the held-out means (pp):", round(float(half_width.max()), 2))
    panel = mean_scores.set_index(["model", "level"]).accuracy_mean
    for _, row in means.iterrows():
        delta = row.accuracy - 100 * panel.loc[(row.model, row.level)]
        print(f"  {row.model} {row.level}: held-out minus in-panel Hard mean = {delta:+.2f} pp")
    print()


def print_build(manifest: dict, publication: dict, settings: dict) -> None:
    print("== Build, run settings and publication")
    print("panel:", manifest["panel"], "| selection:", manifest["selection"]["remove_when"])
    print(f"greedy decoding verified for {len(settings)} models; extra request settings:",
          {model: extra for model, extra in settings.items() if extra})
    print("checks:", ", ".join(name for name, ok in manifest["checks"].items() if ok))
    empty = [f"{level}/{area}" for level, areas in manifest["summary"].items()
             for area, summary in areas.items() if summary["test"] == 0]
    print("configurations without Hard questions:", empty or "none")
    dev = {level: sum(summary["dev"] for summary in areas.values()) for level, areas in manifest["summary"].items()}
    print("dev examples:", dev, "total:", sum(dev.values()), f"| generation size: {GENERATION_SIZE} tokens")
    print("statistics:", manifest["statistics"])
    for repo, record in publication["verified"].items():
        print(f"{repo}: test={record['test']} dev={record['dev']} revision={record['revision']}")
    print()


# -----------------------------------------------------------------------------
# Figures


def plot_rule_sizes(counts: pd.DataFrame) -> Figure:
    """Share of each level removed by every alternative rule."""
    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.0), layout="constrained")
    positions = np.arange(len(RULES))[::-1]
    height = 0.36
    for offset, level, color in ((height / 2, "high_school", BLUE), (-height / 2, "undergraduate", ORANGE)):
        values = [100 * counts[(counts.variant == rule) & (counts.level == level)].removed_fraction.item()
                  for rule in RULES]
        ax.barh(positions + offset, values, height=height - 0.04, color=color, label=LEVEL_LABELS[level])
        ax.text(values[-1] + 1, positions[-1] + offset, f"{values[-1]:.1f}%", va="center", fontsize=5.5)
    ax.set_yticks(positions, [RULE_LABELS[rule] for rule in RULES])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel("Questions removed (%)")
    ax.set_xlim(0, 75)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper right", handlelength=1)
    return figure


def style_row_axis(ax: plt.Axes) -> None:
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def plot_composition(distributions: pd.DataFrame, level: str) -> Figure:
    """Macro-area shares of one level before and after the selection."""
    rows = distributions[(distributions.dimension == "macro_area") & (distributions.level == level)]
    rows = rows.sort_values("original_share")
    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.65 + 0.17 * len(rows)), layout="constrained")
    positions = np.arange(len(rows))
    ax.hlines(positions, 100 * rows.original_share, 100 * rows.hard_share, color=LIGHT, linewidth=1)
    ax.scatter(100 * rows.original_share, positions, s=14, facecolor="white", edgecolor=MUTED, linewidth=0.8,
               zorder=3)
    ax.scatter(100 * rows.hard_share, positions, s=14, color=BLUE, zorder=3)
    ax.set_yticks(positions, [f"{MACRO_AREA_LABELS[value]} ({n:,})" for value, n in zip(rows.value, rows.hard_n)])
    ax.set_title(LEVEL_LABELS[level], fontsize=7)
    ax.set_xlabel("Share of the level's test questions (%)")
    ax.set_xlim(0, 60)
    ax.set_xticks(range(0, 61, 10))
    style_row_axis(ax)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=4, markerfacecolor="white",
                      markeredgecolor=MUTED, label="Original test"),
               Line2D([], [], marker="o", linestyle="", markersize=4, color=BLUE, label="Hard")]
    ax.legend(handles=handles, loc="lower right")
    return figure


def plot_accuracy_shift(rule_scores: pd.DataFrame, counts: pd.DataFrame, level: str) -> Figure:
    """Mean 0/5-shot accuracy of every model on one level's original test and Hard subset."""
    rows = rule_scores[rule_scores.level == level].pivot(index="model", columns="variant", values="accuracy_mean")
    rows = rows.sort_values("original")
    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.1), layout="constrained")
    positions = np.arange(len(rows))
    panel = np.array([model in PANEL for model in rows.index])
    ax.hlines(positions, 100 * rows.reference3, 100 * rows.original, color=LIGHT, linewidth=1)
    ax.scatter(100 * rows.original, positions, s=12, facecolor="white", edgecolor=MUTED, linewidth=0.8, zorder=3)
    ax.scatter(100 * rows.reference3[~panel], positions[~panel], s=12, color=BLUE, zorder=3)
    ax.scatter(100 * rows.reference3[panel], positions[panel], s=16, marker="D", color=ORANGE, zorder=3)
    chance = 100 * counts[(counts.variant == "reference3") & (counts.level == level)].uniform_random_accuracy.item()
    ax.axvline(chance, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.text(chance + 0.8, len(rows) - 0.4, "chance", fontsize=5.5, color=MUTED, va="top")
    ax.set_yticks(positions, [short_name(model) for model in rows.index])
    for label, is_panel in zip(ax.get_yticklabels(), panel):
        label.set_fontweight("bold" if is_panel else "normal")
    ax.set_title(LEVEL_LABELS[level], fontsize=7)
    ax.set_xlabel("Mean of 0-shot and 5-shot accuracy (%)")
    ax.set_xlim(15, 100)
    style_row_axis(ax)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=4, markerfacecolor="white",
                      markeredgecolor=MUTED, label="Original test"),
               Line2D([], [], marker="o", linestyle="", markersize=4, color=BLUE, label="Hard"),
               Line2D([], [], marker="D", linestyle="", markersize=4, color=ORANGE, label="Hard, panel model")]
    ax.legend(handles=handles, loc="lower right")
    return figure


def save_figure(figure: Figure, path: Path) -> None:
    figure.savefig(path, metadata={"CreationDate": None})
    plt.close(figure)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ABLATION_DIR / "figures")
    args = parser.parse_args()
    manifest = read_json(HARD_DIR / "manifest.json")
    publication = read_json(HARD_DIR / "publication.json")
    provenance = read_json(ABLATION_DIR / "provenance.json")
    distributions = pd.DataFrame(read_json(HARD_DIR / "distributions.json"))
    mean_scores = pd.read_csv(HARD_DIR / "mean_scores.csv")
    sensitivity = pd.read_csv(HARD_DIR / "sensitivity.csv")
    counts = pd.read_csv(ABLATION_DIR / "counts.csv")
    rule_scores = pd.read_csv(ABLATION_DIR / "scores.csv")
    audit = load_audit()
    check_artifacts(audit, manifest, distributions, counts, rule_scores, mean_scores, provenance, publication)
    errors = load_panel_errors(audit)
    settings = check_run_configs(manifest["models"])

    print_rules(counts, provenance)
    print_protocols(audit)
    print_drivers(audit)
    print_non_answers(audit, errors)
    print_composition(distributions)
    print_scores(rule_scores, counts)
    print_sensitivity(sensitivity, mean_scores)
    print_build(manifest, publication, settings)

    plt.rcParams.update(STYLE)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(plot_rule_sizes(counts), args.output_dir / "rule_sizes.pdf")
    for level in LEVELS:
        save_figure(plot_composition(distributions, level), args.output_dir / f"composition_shift_{level}.pdf")
        save_figure(plot_accuracy_shift(rule_scores, counts, level), args.output_dir / f"accuracy_shift_{level}.pdf")
    print("Verified: audit, manifest, distributions, rule tables, provenance hashes, panel details and publication.")


if __name__ == "__main__":
    main()
