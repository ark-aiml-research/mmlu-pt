"""Generate the figures and verification summary for the deduplication appendix."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.metrics import average_precision_score, cohen_kappa_score

from mmlu_pt.mcqa_minimal import PUBLIC_FIELDS
from mmlu_pt.pipelines import fuzzy, semantic

REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = Path(__file__).resolve().parent
FIGURE_DIR = ABLATION_DIR / "figures"

ANNOTATION_DIR = ABLATION_DIR / "assets" / "annotations"
LABELS_PATH = ANNOTATION_DIR / "manual_labels_llm_annotated.csv"
HUMAN_SUBSET_PATH = ANNOTATION_DIR / "manual_labels_blind_10_human_annotated.csv"
LLM_SUBSET_PATH = ANNOTATION_DIR / "manual_labels_blind_10_llm_annotated.csv"
SNAPSHOT_PATH = ABLATION_DIR / "assets" / "sources" / "fuzzy_dedup" / "05 - deduplicated" / "0814daf3a224.jsonl"

FUZZY_OUTPUT_DIR = ABLATION_DIR / "fuzzy" / "output"
SWEEP_DIR = FUZZY_OUTPUT_DIR / "ablation" / "studies" / "9603b934c72c"
PAIRWISE_DIR = FUZZY_OUTPUT_DIR / "pairwise_benchmark" / "79482abe9e4b"
SEARCH_DIR = FUZZY_OUTPUT_DIR / "parameter_search" / "79482abe9e4b-a9b58c25bc03"
SEMANTIC_DIR = ABLATION_DIR / "semantic" / "output" / "ablation" / "7e16f8554e7fb8462646"

PIPELINE_OUTPUT_DIR = REPO_ROOT / "output"
FILTERED_DIR = PIPELINE_OUTPUT_DIR / "04 - filtered"
EXACT_DIR = PIPELINE_OUTPUT_DIR / "05 - deduplicated"
SEMANTIC_OUTPUT_PATH = PIPELINE_OUTPUT_DIR / "06 - semantic-deduplicated" / "data.jsonl"
SEMANTIC_WORK_DIR = PIPELINE_OUTPUT_DIR / "semantic-deduplication-work"

# Fingerprints recorded by the ablation notebooks.
ANNOTATION_SHA256 = "a51cbf1099b969b0573abd104184bf91874df64f1dc09bbb5a418ac2110290a6"
SNAPSHOT_FINGERPRINT = "9603b934c72ceb641880da0aa02de34c1a4e55383f18a0593eb269c7f10fb456"

SELECTED_TREATMENT = "Octen-Embedding-8B__question_choices"
SEED_RUNS = [
    "question_choices-b20-r13-n24-s17", "question_choices-b20-r13-n24-s42", "question_choices-b20-r13-n24-s101",
]
SAMPLED_RUNS = [
    "question_choices-b10-r26-n24-s42", "question_choices-b13-r20-n24-s42", "question_choices-b20-r13-n24-s42",
    "question_choices-b26-r10-n24-s42", "question_choices-b20-r13-n20-s42",
]
BANDINGS = [(10, 26), (13, 20), (20, 13), (26, 10)]
MODELS = [
    "bge-m3", "gte-multilingual-base", "Octen-Embedding-0.6B", "Octen-Embedding-4B", "Octen-Embedding-8B",
    "Nemotron-3-Embed-1B-BF16", "Nemotron-3-Embed-8B-BF16", "Qwen3-Embedding-4B", "Qwen3-Embedding-8B",
    "llama-embed-nemotron-8b",
]
REPRESENTATIONS = ["question", "question_choices"]
LABELS = ["duplicate", "related_but_distinct", "distinct"]
SIMILARITY_BANDS = [">=0.90", "0.80-0.90", "0.70-0.80", "<0.70"]
STAGES = ["exact", "semantic"]
LEVEL_LABELS = {"high_school": "High school", "undergraduate": "Undergraduate"}
EXAM_LABELS = {"RESIDENCIA_USP_UNICAMP": "Res. USP/Unicamp"}
BAND_LABELS = {">=0.90": r"$\geq$0.90", "0.80-0.90": "0.80–0.90", "0.70-0.80": "0.70–0.80", "<0.70": "<0.70"}
LABEL_NAMES = {"duplicate": "Duplicate", "related_but_distinct": "Related but distinct", "distinct": "Distinct"}
REPRESENTATION_NAMES = {"question": "Question", "question_choices": "Question + choices"}
STAGE_NAMES = {"exact": "Exact", "semantic": "Semantic"}
REMOVAL_COSINE_BINS = [semantic.COSINE_THRESHOLD, 0.98, 0.99, 0.995, 0.999, 1.0]

# ACL page geometry: 7.7 cm columns inside a 16 cm text block.
COLUMN_WIDTH = 7.7 / 2.54
TEXT_WIDTH = 16.0 / 2.54

# First three slots of the reference categorical palette (validated for all pairs).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
LABEL_COLORS = {"duplicate": BLUE, "related_but_distinct": ORANGE, "distinct": AQUA}
LABEL_MARKERS = {"duplicate": "o", "related_but_distinct": "s", "distinct": "^"}
REPRESENTATION_COLORS = {"question": ORANGE, "question_choices": BLUE}
STAGE_COLORS = {"exact": BLUE, "semantic": ORANGE}

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
SHADE = "#f0efec"
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL = LinearSegmentedColormap.from_list("sequential", ["#f7f9fc", *BLUE_RAMP])

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
    "xtick.minor.width": 0.4,
    "xtick.minor.size": 1.2,
    "ytick.minor.width": 0.4,
    "ytick.minor.size": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_parquet_tree(path: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(file) for file in sorted(path.rglob("*.parquet"))], ignore_index=True)


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    share = successes / total
    center = (share + z**2 / (2 * total)) / (1 + z**2 / total)
    margin = z * np.sqrt(share * (1 - share) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    return center - margin, center + margin


def confusion(labels: pd.Series, predictions: pd.Series) -> dict[str, float]:
    """Binary confusion counts and metrics with `duplicate` as the positive class."""
    positive = labels.eq("duplicate")
    tp, fp = int((positive & predictions).sum()), int((~positive & predictions).sum())
    fn, tn = int((positive & ~predictions).sum()), int((~positive & ~predictions).sum())
    mcc = (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / (tp + fp), "recall": tp / (tp + fn), "specificity": tn / (tn + fp),
        "f1": 2 * tp / (2 * tp + fp + fn), "mcc": mcc,
    }


def load_labels() -> pd.DataFrame:
    """Read the LLM-labelled reference panel after checking its fingerprint."""
    if file_sha256(LABELS_PATH) != ANNOTATION_SHA256:
        raise ValueError(f"Unexpected annotation file: {LABELS_PATH}")
    return pd.read_csv(LABELS_PATH)


def load_snapshot() -> pd.DataFrame:
    """Order the ablation snapshot as NeMo did, so `_curator_dedup_id` is the row position."""
    rows = []
    for record in read_jsonl(SNAPSHOT_PATH):
        payload = json.dumps({field: record.get(field) for field in PUBLIC_FIELDS},
                             ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        rows.append({
            "study_id": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "exam": record["exam"],
            "exam_edition": record["exam_edition"],
            "academic_level": record["academic_level"],
        })
    snapshot = pd.DataFrame(rows).sort_values("study_id", ignore_index=True)

    digest = hashlib.sha256()
    for study_id in snapshot["study_id"]:
        digest.update(study_id.encode("ascii") + b"\n")
    if digest.hexdigest() != SNAPSHOT_FINGERPRINT:
        raise ValueError(f"Unexpected ablation snapshot: {SNAPSHOT_PATH}")
    return snapshot


def sweep_exam_removal(snapshot: pd.DataFrame, slugs: list[str]) -> pd.DataFrame:
    """Share of each exam flagged for removal by every corpus-level LSH run (%)."""
    totals = snapshot["exam"].value_counts()
    rates = {}
    for slug in slugs:
        removed = read_parquet_tree(SWEEP_DIR / "runs" / slug / "results" / "FuzzyDuplicateIds")
        exams = snapshot.loc[removed["_curator_dedup_id"].astype(int), "exam"]
        rates[slug] = 100 * exams.value_counts().reindex(totals.index, fill_value=0) / totals
    return pd.DataFrame(rates)


def clustered_ids(slug: str) -> set[int]:
    components = read_parquet_tree(SWEEP_DIR / "runs" / slug / "cache" / "ConnectedComponentsStage")
    return set(components["_curator_dedup_id"].astype(int))


def load_semantic_pairs() -> pd.DataFrame:
    """Join labels, cosine scores, the fuzzy reference and exam names for each pair."""
    pairs = pd.read_parquet(SEMANTIC_DIR / "pairs.parquet")[["pair_key", "label"]]
    scores = pd.read_parquet(SEMANTIC_DIR / "pair_scores.parquet")
    reference = pd.read_parquet(SEMANTIC_DIR / "fuzzy_reference.parquet").rename(
        columns={"prediction": "fuzzy_prediction"}
    )
    audit = pd.read_parquet(SEMANTIC_DIR / "out_of_fold_pair_audit.parquet")[
        ["pair_key", "left_exam", "left_question", "oof_error_rate"]
    ]
    return pairs.merge(scores).merge(reference).merge(audit)


def treatment_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the fixed-treatment outer-CV metrics and add full-panel AP and selection counts."""
    metrics = pd.read_parquet(SEMANTIC_DIR / "treatment_metrics.parquet")
    summary = metrics.groupby("configuration")[["mcc", "precision", "recall", "f1"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    positive = pairs["label"].eq("duplicate")
    summary["average_precision"] = [average_precision_score(positive, pairs[name]) for name in summary.index]
    selected = pd.read_parquet(SEMANTIC_DIR / "outer_folds.parquet")["configuration"].value_counts()
    summary["times_selected"] = selected.reindex(summary.index, fill_value=0)
    summary[["model", "representation"]] = summary.index.to_series().str.split("__", expand=True)
    return summary.sort_values("mcc_mean", ascending=False)


def stage_counts() -> pd.DataFrame:
    """Records per exam before and after each deduplication stage of the pipeline."""
    stages = {
        "filtered": sorted(FILTERED_DIR.glob("*.jsonl")),
        "after_exact": sorted(EXACT_DIR.glob("*.jsonl")),
        "after_semantic": [SEMANTIC_OUTPUT_PATH],
    }
    counts = {}
    levels = {}
    for stage, paths in stages.items():
        records = [record for path in paths for record in read_jsonl(path)]
        exams = pd.Series([record["exam"] for record in records])
        counts[stage] = exams.value_counts()
        levels.update({record["exam"]: record["academic_level"] for record in records})
    table = pd.DataFrame(counts).fillna(0).astype(int)
    table["academic_level"] = table.index.map(levels)
    table["exact"] = table["filtered"] - table["after_exact"]
    table["semantic"] = table["after_exact"] - table["after_semantic"]
    return table


def production_run() -> tuple[dict, pd.DataFrame]:
    """Find the semantic run that produced the published output and label its removals by exam."""
    output_sha256 = file_sha256(SEMANTIC_OUTPUT_PATH)
    manifests = [read_jsonl(path)[0] | {"path": path} for path in SEMANTIC_WORK_DIR.glob("*/run-*/manifest.json")]
    manifest = next(manifest for manifest in manifests if manifest["output_sha256"] == output_sha256)

    run_dir = manifest["path"].parent
    exams = {
        path.name: [record["exam"] for record in read_jsonl(path)] for path in sorted(EXACT_DIR.glob("*.jsonl"))
    }
    records = pd.DataFrame(read_jsonl(run_dir / "records.jsonl"))
    record_exams = [exams[source][line - 1] for source, line in zip(records["source"], records["line"], strict=True)]
    exam_by_id = dict(zip(records["id"], record_exams, strict=True))
    removals = pd.DataFrame(read_jsonl(run_dir / "removals.jsonl"))
    removals["removed_exam"] = removals["removed_id"].map(exam_by_id)
    removals["representative_exam"] = removals["representative_id"].map(exam_by_id)
    return manifest, removals


def format_thousands(value: float, _position: int = 0) -> str:
    return f"{value:,.0f}"


def print_summary(
    labels: pd.DataFrame,
    snapshot: pd.DataFrame,
    exam_removal: pd.DataFrame,
    pairs: pd.DataFrame,
    treatments: pd.DataFrame,
    stages: pd.DataFrame,
    manifest: dict,
    removals: pd.DataFrame,
) -> None:
    """Print every number quoted in the appendix for comparison with the notebooks."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("== Reference panel")
    items = pd.concat([labels["left_id"], labels["right_id"]]).nunique()
    print(f"Pairs: {len(labels)}, unique items: {items}, same exam: {int(labels['same_exam'].sum())}")
    print("Labels:", labels["label"].value_counts().to_dict())
    draws = labels["sampled_for_config"].str.split("|").explode()
    print("Draws per sampled configuration:", draws.value_counts().to_dict())
    print("Jaccard:", labels["jaccard"].describe().round(3).to_dict())
    for column in ["similarity_band", "component_band", "left_exam"]:
        table = pd.crosstab(labels[column], labels["label"], margins=True)
        table["duplicate_pct"] = (100 * table["duplicate"] / table["All"]).round(1)
        print(table.to_string())

    human, llm = pd.read_csv(HUMAN_SUBSET_PATH), pd.read_csv(LLM_SUBSET_PATH)
    if not human["pair_key"].equals(llm["pair_key"]):
        raise ValueError("Human and LLM subsets are not aligned")
    blind = labels.set_index("pair_key").loc[human["pair_key"]]
    print("Human subset equals LLM full-panel labels:", bool((blind["label"].to_numpy() == llm["label"]).all()))
    print("Human subset labels:", human["label"].value_counts().to_dict(),
          "LLM:", llm["label"].value_counts().to_dict())
    print(pd.crosstab(human["label"], llm["label"]).to_string())
    for name, left, right in [
        ("4-way", human["label"], llm["label"]),
        ("binary", human["label"].eq("duplicate"), llm["label"].eq("duplicate")),
    ]:
        agree = int((left == right).sum())
        low, high = wilson_interval(agree, len(left))
        print(f"{name}: {agree}/{len(left)} ({100 * agree / len(left):.2f}%, Wilson {100 * low:.2f}-{100 * high:.2f}), "
              f"kappa={cohen_kappa_score(left, right):.4f}")

    editions = snapshot.set_index("study_id")["exam_edition"]
    traced = labels.assign(
        left_edition=labels["left_study_id"].map(editions), right_edition=labels["right_study_id"].map(editions)
    ).dropna(subset=["left_edition", "right_edition"])
    for name, pattern in [("year", r"((?:19|20)\d\d)"), ("phase", r"(\d{4} Fase \d)")]:
        traced[f"same_{name}"] = (traced["left_edition"].str.extract(pattern)[0]
                                  == traced["right_edition"].str.extract(pattern)[0])
    traced["same_edition"] = traced["left_edition"] == traced["right_edition"]
    duplicates = traced[traced["label"] == "duplicate"]
    print("Duplicate pairs with traceable editions:")
    print(duplicates.groupby("left_exam")[["same_edition", "same_year", "same_phase"]].agg(["size", "sum"]).to_string())

    print("\n== Corpus-level LSH sweep")
    print(f"Snapshot: {len(snapshot):,} records, {snapshot['exam'].nunique()} exams, "
          f"{snapshot['exam_edition'].nunique():,} editions, "
          f"levels {snapshot['academic_level'].value_counts().to_dict()}")
    sweep = pd.read_csv(SWEEP_DIR / "metrics.csv")
    columns = ["slug", "p50_similarity", "duplicates", "retention_pct", "components", "largest_component",
               "cross_exam_components", "max_exam_removal_pct", "max_exam_share_shift_pp", "jaccard_median",
               "edges_ge_0_8_pct", "total_time_s"]
    print(sweep[columns].round(3).to_string(index=False))
    print("Worst exam per run:", exam_removal.idxmax().to_dict())
    print(exam_removal.round(2).sort_values(exam_removal.columns[0], ascending=False).to_string())
    for slug in exam_removal.columns:
        flagged = exam_removal[slug] / 100 * snapshot["exam"].value_counts()
        by_level = flagged.groupby(snapshot.groupby("exam")["academic_level"].first()).sum()
        level_totals = snapshot["academic_level"].value_counts()
        print(f"{slug} by level (%):", (100 * by_level / level_totals).round(2).to_dict())
    clustered = {slug: clustered_ids(slug) for slug in SEED_RUNS}
    for slug in SEED_RUNS:
        expected = sweep.set_index("slug").loc[slug, "clustered_documents"]
        print(f"Clustered documents {slug}: {len(clustered[slug])} (metrics.csv {expected})")
    for index, left in enumerate(SEED_RUNS):
        for right in SEED_RUNS[index + 1:]:
            overlap = len(clustered[left] & clustered[right]) / len(clustered[left] | clustered[right])
            print(f"Seed overlap {left[-3:]} vs {right[-3:]}: {overlap:.3f}")

    print("\n== Pairwise LSH benchmark")
    pairwise = pd.read_csv(PAIRWISE_DIR / "metrics.csv")
    columns = ["slug", "tp", "fp", "tn", "fn", "precision", "precision_95_lower", "precision_95_upper", "recall",
               "recall_95_lower", "recall_95_upper", "specificity", "f1", "mcc"]
    print(pairwise[columns].round(4).to_string(index=False))
    mcnemar = pd.read_csv(PAIRWISE_DIR / "mcnemar.csv")
    print(mcnemar.to_string(index=False))
    print(f"McNemar comparisons: {len(mcnemar)}, significant after Holm: {int((mcnemar['p_value_holm'] < 0.05).sum())}")

    print("\n== Parameter search")
    outer = pd.read_csv(SEARCH_DIR / "outer_fold_results.csv")
    print("Outer-fold selections:", outer["selected_slug"].value_counts().to_dict())
    print("Outer-fold MCC:", outer["mcc"].agg(["mean", "std", "min", "max"]).round(4).to_dict())
    repeats = pd.read_csv(SEARCH_DIR / "nested_repeat_metrics.csv")
    print(repeats[["repeat", "tp", "fp", "tn", "fn", "precision", "recall", "mcc"]].round(4).to_string(index=False))
    means = repeats[["mcc", "precision", "recall", "specificity", "f1"]].agg(["mean", "std"])
    print("Repeat means:", means.round(4).to_dict())
    selected = json.loads((SEARCH_DIR / "selected_configuration.json").read_text(encoding="utf-8"))
    parameters = selected["parameters"]
    print("Selected:", parameters)
    print("Selected rule matches mmlu_pt.pipelines.fuzzy:", all(
        parameters[key] == fuzzy.PARAMETERS[key]
        for key in ["char_ngrams", "num_bands", "minhashes_per_band", "min_matching_bands", "jaccard_threshold", "seed"]
    ))
    full = selected["descriptive_full_panel_metrics"]
    print("Full panel:", {key: round(value, 4) for key, value in full.items()})
    for name, successes, total in [("precision", full["tp"], full["tp"] + full["fp"]),
                                   ("recall", full["tp"], full["tp"] + full["fn"]),
                                   ("specificity", full["tn"], full["tn"] + full["fp"])]:
        print(f"Wilson {name}:", tuple(round(bound, 4) for bound in wilson_interval(successes, total)))
    for interval in selected["component_bootstrap_intervals"]:
        low, high = interval["bootstrap_95_lower"], interval["bootstrap_95_upper"]
        print(f"Bootstrap {interval['metric']}: [{low:.4f}, {high:.4f}]")
    cv = pd.read_parquet(SEARCH_DIR / "cv_ranking.parquet").sort_values("cv_mean_mcc", ascending=False, kind="stable")
    ties = cv[np.isclose(cv["cv_mean_mcc"], cv["cv_mean_mcc"].max(), rtol=0, atol=1e-12)]
    print(f"Configurations: {len(cv)}, tied at the best CV MCC: {len(ties)}, "
          f"n-grams {sorted(ties['char_ngrams'].unique())}, thresholds {sorted(ties['jaccard_threshold'].unique())}")
    columns = ["slug", "cv_mean_mcc", "cv_mean_precision", "cv_mean_recall", "cv_mean_f1"]
    print("Best per n-gram (any rule):")
    print(cv.groupby("char_ngrams").head(1).sort_values("char_ngrams")[columns].round(4).to_string(index=False))
    print("Best per n-gram (pure LSH):")
    pure = cv[cv["jaccard_threshold"].isna()]
    print(pure.groupby("char_ngrams").head(1).sort_values("char_ngrams")[columns].round(4).to_string(index=False))
    precise = cv[cv["cv_mean_precision"] >= 0.95]
    best_recall = precise.loc[precise["cv_mean_recall"].idxmax()]
    print(f"Best CV recall with CV precision >= 0.95: {best_recall['cv_mean_recall']:.4f} ({best_recall['slug']})")
    family = selected_family(cv)
    print("Threshold curve of the selected family:")
    columns = ["jaccard_threshold", "cv_mean_mcc", "cv_mean_precision", "cv_mean_recall"]
    print(family[columns].round(4).to_string(index=False))

    print("\n== Semantic study")
    print("Items:", pd.read_parquet(SEMANTIC_DIR / "items.parquet").shape[0],
          "groups:", pd.read_parquet(SEMANTIC_DIR / "pairs.parquet")["group_id"].nunique())
    columns = ["times_selected", "mcc_mean", "mcc_std", "precision_mean", "precision_std", "recall_mean",
               "recall_std", "f1_mean", "f1_std", "average_precision"]
    print(treatments[columns].round(4).to_string())
    gain = treatments.pivot(index="model", columns="representation", values="mcc_mean")
    print("MCC gain from adding choices:", (gain["question_choices"] - gain["question"]).round(3).to_dict())
    folds = pd.read_parquet(SEMANTIC_DIR / "outer_folds.parquet")
    print("Outer-fold thresholds:", folds["threshold"].round(6).value_counts().to_dict())
    print("Outer-fold selections:", folds["configuration"].value_counts().to_dict())
    repeats = pd.read_parquet(SEMANTIC_DIR / "repeat_metrics.parquet")
    print(repeats.round(4).to_string(index=False))
    means = repeats[["mcc", "precision", "recall", "f1"]].agg(["mean", "std", "min", "max"])
    print("Repeat means:", means.round(4).to_dict())
    print("Repeat totals:", repeats[["tp", "fp", "tn", "fn"]].sum().to_dict())

    chosen = json.loads((SEMANTIC_DIR / "selected_configuration.json").read_text(encoding="utf-8"))
    print("Selected:", chosen)
    print("Selected treatment matches mmlu_pt.pipelines.semantic:",
          chosen["configuration"] == f"{semantic.MODEL_ID.split('/')[1]}__{semantic.PARAMETERS['representation']}"
          and chosen["threshold"] == semantic.COSINE_THRESHOLD)
    predictions = pairs[SELECTED_TREATMENT] >= semantic.COSINE_THRESHOLD
    full = confusion(pairs["label"], predictions)
    print("Full panel (semantic):", {key: round(value, 4) for key, value in full.items()})
    print("Wilson precision (semantic):",
          tuple(round(bound, 4) for bound in wilson_interval(full["tp"], full["tp"] + full["fp"])))
    lexical = confusion(pairs["label"], pairs["fuzzy_prediction"])
    print("Full panel (fuzzy):", {key: round(value, 4) for key, value in lexical.items()})
    print("Fuzzy rule equals 8-gram Jaccard >= 0.94:",
          bool((pairs["fuzzy_prediction"] == (pairs["jaccard"] >= fuzzy.JACCARD_THRESHOLD)).all()))
    rules = [predictions.rename("semantic"), pairs["fuzzy_prediction"].rename("fuzzy")]
    print(pd.crosstab(pairs["label"], rules).to_string())
    print("Scores by label:")
    print(pairs.groupby("label")[[SELECTED_TREATMENT, "jaccard"]].describe().T.round(4).to_string())
    missed_by_fuzzy = pairs[pairs["label"].eq("duplicate") & ~pairs["fuzzy_prediction"]]
    print("J8 of duplicates missed by the fuzzy rule:", missed_by_fuzzy["jaccard"].describe().round(4).to_dict())
    fuzzy_errors = pairs[pairs["fuzzy_prediction"] & ~pairs["label"].eq("duplicate")]
    print("Fuzzy false positives:")
    print(fuzzy_errors[["label", "left_exam", SELECTED_TREATMENT, "jaccard"]]
          .assign(question=fuzzy_errors["left_question"].str[:70]).round(4).to_string(index=False))
    errors = pairs[predictions != pairs["label"].eq("duplicate")]
    print("Full-panel errors:")
    print(errors[["label", "left_exam", SELECTED_TREATMENT, "jaccard", "oof_error_rate"]]
          .assign(question=errors["left_question"].str[:70]).round(4).to_string(index=False))
    missed = pairs[pairs["label"].eq("duplicate") & pairs["oof_error_rate"].gt(0)]
    print(f"Duplicates missed in at least one repeat: {len(missed)}")
    print(missed[["left_exam", SELECTED_TREATMENT, "oof_error_rate"]]
          .assign(question=missed["left_question"].str[:60]).round(6).to_string(index=False))
    wrong_negatives = pairs[~pairs["label"].eq("duplicate") & pairs["oof_error_rate"].gt(0)]
    print(f"Negatives misclassified in at least one repeat: {len(wrong_negatives)}",
          wrong_negatives["left_exam"].value_counts().to_dict(),
          wrong_negatives["oof_error_rate"].round(1).value_counts().to_dict())
    scores = pairs[SELECTED_TREATMENT]
    print(f"Scores bracketing the threshold: {scores[~predictions].max():.6f} "
          f"< {semantic.COSINE_THRESHOLD:.6f} <= {scores[predictions].min():.6f}")

    tokens = pd.read_parquet(SEMANTIC_DIR / "token_audit.parquet")
    print("Truncated items (all configurations):", int(tokens["truncated"].sum()),
          "max tokens:", int(tokens["tokens"].max()))
    octen = tokens[tokens["configuration"] == SELECTED_TREATMENT]["tokens"]
    print("Selected tokens:", octen.describe(percentiles=[0.5, 0.95, 0.99]).round(1).to_dict())
    print(pd.read_csv(SEMANTIC_DIR / "embedding_runtime.csv").round(2).to_string(index=False))
    print(pd.read_csv(SEMANTIC_DIR / "nemo_cluster_diagnostics.csv").round(4).to_string(index=False))
    marked = [set(pd.read_parquet(path)["id"]) for path in sorted(SEMANTIC_DIR.glob("nemo/*/k*/removal_audit.parquet"))]
    print(f"NeMo items marked by every k: {len(set.intersection(*marked))}, by any k: {len(set.union(*marked))}")

    print("\n== Final pipeline")
    total = stages[["filtered", "exact", "after_exact", "semantic", "after_semantic"]].sum()
    removed = total["exact"] + total["semantic"]
    print(f"Removed {removed:,} ({100 * removed / total['filtered']:.3f}%): "
          f"exact {100 * total['exact'] / total['filtered']:.3f}% of D_F, "
          f"semantic {100 * total['semantic'] / total['after_exact']:.3f}% of D_E "
          f"({100 * total['semantic'] / total['filtered']:.3f}% of D_F); "
          f"retention {100 * total['after_semantic'] / total['filtered']:.3f}%")
    sizes = ["filtered", "after_exact", "after_semantic"]
    levels = stages.groupby("academic_level")[sizes].sum()
    print(pd.concat([levels, stages[sizes].sum().rename("all").to_frame().T]).to_string())
    table = stages.assign(
        exact_pct=100 * stages["exact"] / stages["filtered"],
        semantic_pct=100 * stages["semantic"] / stages["after_exact"],
        retention_pct=100 * stages["after_semantic"] / stages["filtered"],
    ).sort_values(["academic_level", "filtered"], ascending=[True, False])
    print(table.round(3).to_string())
    embeddings = manifest["embeddings"]
    print(f"Semantic run: {manifest['input_records']} -> {manifest['output_records']} "
          f"(removed {manifest['removed_records']}), {manifest['elapsed_seconds']:.1f} s total, "
          f"embeddings {embeddings['seconds']:.1f} s on {embeddings['gpu']}, "
          f"peak {embeddings['peak_gpu_gib']:.1f} GiB, "
          f"truncated {embeddings['truncated_items']}")
    token_counts = pd.Series(embeddings["tokens"])
    print("Production tokens:", token_counts.describe(percentiles=[0.5, 0.99]).round(1).to_dict())
    print("Parameters match the pipeline:", manifest["parameters"] == semantic.PARAMETERS)
    print("Removal cosines:", removals["cosine"].describe().round(5).to_dict())
    bins = pd.cut(removals["cosine"], REMOVAL_COSINE_BINS, right=False, include_lowest=True)
    print("Removal cosine bins:", bins.value_counts(sort=False).to_dict())
    false_positives = pairs[predictions & ~pairs["label"].eq("duplicate")][SELECTED_TREATMENT]
    risky = removals[removals["cosine"] <= false_positives.max()]
    print(f"Removals with cosine <= the largest panel false positive ({false_positives.max():.4f}): {len(risky)} "
          f"({100 * len(risky) / len(removals):.1f}%)", risky["removed_exam"].value_counts().to_dict())
    per_representative = removals["representative_id"].value_counts()
    print(f"Representatives: {len(per_representative)}, largest group removes {per_representative.max()}, "
          f"cross-exam removals: {int((removals['removed_exam'] != removals['representative_exam']).sum())}")


def selected_family(ranking: pd.DataFrame) -> pd.DataFrame:
    """Configurations that differ from the selected fuzzy rule only in the Jaccard threshold."""
    family = ranking[
        (ranking["char_ngrams"] == fuzzy.CHAR_NGRAMS)
        & (ranking["num_bands"] == fuzzy.NUM_BANDS)
        & (ranking["minhashes_per_band"] == fuzzy.HASHES_PER_BAND)
        & (ranking["min_matching_bands"] == fuzzy.MIN_MATCHING_BANDS)
    ]
    return family.sort_values("jaccard_threshold", na_position="first")


def banding_label(bands: int, rows: int) -> str:
    return f"{bands}×{rows}"


def exam_rows(stages: pd.DataFrame, order: pd.Series) -> list[tuple[str, str | None]]:
    """Group exams by level, sorted by `order` in descending order, with one header row per level."""
    rows = []
    for level, level_label in LEVEL_LABELS.items():
        rows.append((level_label, None))
        exams = stages.index[stages["academic_level"] == level]
        for exam in order.reindex(exams).sort_values(ascending=False, kind="stable").index:
            rows.append((EXAM_LABELS.get(exam, exam), exam))
    return rows


def style_row_axis(ax: plt.Axes, rows: list[tuple[str, str | None]]) -> None:
    """Label exam rows and set the level header rows in bold."""
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    for tick_label, (_, exam) in zip(ax.get_yticklabels(), rows, strict=True):
        if exam is None:
            tick_label.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)


def draw_share_bars(ax: plt.Axes, counts: pd.DataFrame, names: list[str]) -> None:
    """Stack label shares per group and print the group size at the end of each bar."""
    shares = 100 * counts.div(counts.sum(axis=1), axis=0)
    left = np.zeros(len(counts))
    for label in LABELS:
        ax.barh(range(len(counts)), shares[label], left=left, height=0.7, color=LABEL_COLORS[label],
                edgecolor="white", linewidth=0.4, label=LABEL_NAMES[label])
        left += shares[label].to_numpy()
    for row, total in enumerate(counts.sum(axis=1)):
        ax.text(101.5, row, f"{total}", ha="left", va="center", fontsize=5.5, color=MUTED)
    ax.set_yticks(range(len(counts)), names)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.spines["left"].set_visible(False)


def plot_panel_composition(labels: pd.DataFrame) -> Figure:
    """Figure: label composition of the reference panel by similarity band and by exam."""
    by_band = pd.crosstab(labels["similarity_band"], labels["label"]).reindex(SIMILARITY_BANDS)
    by_exam = pd.crosstab(labels["left_exam"], labels["label"])
    by_exam = by_exam.loc[by_exam.sum(axis=1).sort_values(ascending=False, kind="stable").index]
    by_band, by_exam = by_band.reindex(columns=LABELS, fill_value=0), by_exam.reindex(columns=LABELS, fill_value=0)

    figure, axes = plt.subplots(
        2, 1, sharex=True, figsize=(COLUMN_WIDTH, 2.9), layout="constrained",
        gridspec_kw={"height_ratios": [len(by_band), len(by_exam)]},
    )
    draw_share_bars(axes[0], by_band, [BAND_LABELS[band] for band in by_band.index])
    draw_share_bars(axes[1], by_exam, [EXAM_LABELS.get(exam, exam) for exam in by_exam.index])
    axes[0].set_ylabel("(a) Jaccard band")
    axes[1].set_ylabel("(b) Exam")
    axes[1].set_xlabel("Share of pairs (%); right: number of pairs")
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(25))
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    return figure


def plot_lsh_sweep(sweep: pd.DataFrame) -> Figure:
    """Figure: theoretical LSH collision curves and the corpus-level coverage/purity trade-off."""
    figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.2), layout="constrained",
                                gridspec_kw={"width_ratios": [1, 1.35]})
    similarity = np.linspace(0, 1, 501)
    ax = axes[0]
    for color, (bands, rows) in zip(BLUE_RAMP[3::3], BANDINGS, strict=True):
        p50 = (1 - 0.5 ** (1 / bands)) ** (1 / rows)
        ax.plot(similarity, 1 - (1 - similarity**rows) ** bands, color=color, linewidth=1,
                label=f"$b$={bands}, $r$={rows} ($s_{{50}}$={p50:.2f})")
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=0.5)
    ax.set_xlim(0.4, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"(a) Jaccard similarity $s$ of a pair")
    ax.set_ylabel("Probability of becoming a candidate")
    ax.legend(loc="upper left", handlelength=1.2)
    ax.grid(color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)

    ax = axes[1]
    banding = sweep[sweep["family"] == "banding"]
    for representation in REPRESENTATIONS:
        runs = banding[banding["representation"] == representation].sort_values("num_bands")
        ax.plot(runs["duplicates"], runs["edges_ge_0_8_pct"], color=REPRESENTATION_COLORS[representation],
                linewidth=0.8, marker="o", markersize=3.5, label=REPRESENTATION_NAMES[representation])
        for _, run in runs.iterrows():
            ax.annotate(banding_label(run["num_bands"], run["minhashes_per_band"]),
                        (run["duplicates"], run["edges_ge_0_8_pct"]), xytext=(3, 2), textcoords="offset points",
                        fontsize=5, color=MUTED)
    variants = sweep[sweep["family"] != "banding"]
    for _, run in variants.iterrows():
        marker = "s" if run["family"] == "seed" else "^" if run["char_ngrams"] < 24 else "v"
        ax.scatter(run["duplicates"], run["edges_ge_0_8_pct"], marker=marker, s=12,
                   color=REPRESENTATION_COLORS["question_choices"], edgecolors="white", linewidths=0.3, zorder=3)
    sampled = sweep[sweep["slug"].isin(SAMPLED_RUNS)]
    ax.scatter(sampled["duplicates"], sampled["edges_ge_0_8_pct"], s=48, facecolors="none", edgecolors=INK,
               linewidths=0.6, zorder=4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
    ax.set_xlabel("(b) Records flagged for removal (of 39,120)")
    ax.set_ylabel(r"Audited edges with $J \geq 0.8$ (%)")
    ax.set_ylim(45, 101)
    ax.grid(color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    handles, names = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], linestyle="", marker="^", markersize=3.5, color=BLUE, label="$n$=20"),
        Line2D([], [], linestyle="", marker="v", markersize=3.5, color=BLUE, label="$n$=30"),
        Line2D([], [], linestyle="", marker="s", markersize=3.5, color=BLUE, label="Seeds 17, 101"),
        Line2D([], [], linestyle="", marker="o", markersize=6, markerfacecolor="none", markeredgecolor=INK,
               markeredgewidth=0.6, label="Sampled for the panel"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=3, columnspacing=1)
    return figure


def plot_lsh_exam_removal(exam_removal: pd.DataFrame, snapshot: pd.DataFrame) -> Figure:
    """Figure: per-exam removal rate of the eight banding runs of the corpus-level sweep."""
    slugs = [f"{representation}-b{bands}-r{rows}-n24-s42"
             for representation in REPRESENTATIONS for bands, rows in BANDINGS]
    levels = snapshot.groupby("exam")["academic_level"].first().to_frame()
    rows = exam_rows(levels, exam_removal[slugs[-1]])
    values = exam_removal[slugs].reindex([exam for _, exam in rows]).to_numpy(dtype=float)
    norm = Normalize(vmin=0, vmax=40, clip=True)

    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.3), layout="constrained")
    colormap = SEQUENTIAL.copy()
    colormap.set_bad("none")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=colormap, norm=norm, aspect="auto")
    for (row, col), value in np.ndenumerate(values):
        if not np.isnan(value):
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=4.8,
                    color="white" if value >= 22 else INK)
    ax.axvline(len(BANDINGS) - 0.5, color="white", linewidth=2)
    ax.set_xticks(range(len(slugs)), [banding_label(bands, rows) for bands, rows in BANDINGS] * 2)
    ax.tick_params(length=0)
    for index, representation in enumerate(REPRESENTATIONS):
        ax.text(index * len(BANDINGS) + (len(BANDINGS) - 1) / 2, -1.3, REPRESENTATION_NAMES[representation],
                ha="center", va="bottom", fontsize=6)
    ax.set_xlabel(r"Banding $b \times r$ (260 hashes, $n$=24, seed 42)")
    style_row_axis(ax, rows)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=ax, extend="max", shrink=0.6, aspect=25, pad=0.02, ticks=range(0, 41, 10))
    colorbar.set_label("Records flagged (% of exam)")
    colorbar.outline.set_visible(False)
    return figure


def plot_fuzzy_search(ranking: pd.DataFrame) -> Figure:
    """Figure: best cross-validated MCC per n-gram and banding, and the Jaccard threshold curve."""
    labels = [banding_label(bands, rows)
              for bands, rows in zip(ranking["num_bands"], ranking["minhashes_per_band"], strict=True)]
    ranking = ranking.assign(banding=labels)
    bandings = (ranking.drop_duplicates("banding").sort_values("num_bands")["banding"]).tolist()
    best = ranking.pivot_table(index="char_ngrams", columns="banding", values="cv_mean_mcc", aggfunc="max")[bandings]

    figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.1), layout="constrained",
                                gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    image = ax.imshow(best.to_numpy(), cmap=SEQUENTIAL, vmin=0.45, vmax=0.75, aspect="auto")
    for (row, col), value in np.ndenumerate(best.to_numpy()):
        ax.text(col, row, f"{value:.2f}"[1:], ha="center", va="center", fontsize=4.8,
                color="white" if value >= 0.64 else INK)
    row = best.index.get_loc(fuzzy.CHAR_NGRAMS)
    col = bandings.index(banding_label(fuzzy.NUM_BANDS, fuzzy.HASHES_PER_BAND))
    ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=0.9))
    ax.set_xticks(range(len(bandings)), bandings, rotation=90)
    ax.set_yticks(range(len(best)), [f"$n$={ngram}" for ngram in best.index])
    ax.tick_params(length=0)
    ax.set_xlabel(r"(a) Banding $b \times r$")
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=ax, extend="min", shrink=0.8, aspect=20, pad=0.01)
    colorbar.set_label("Best mean CV MCC")
    colorbar.outline.set_visible(False)

    ax = axes[1]
    family = selected_family(ranking)
    unfiltered = family[family["jaccard_threshold"].isna()].iloc[0]
    curve = family.dropna(subset=["jaccard_threshold"])
    for column, color, name in [("cv_mean_mcc", INK, "MCC"), ("cv_mean_precision", BLUE, "Precision"),
                                ("cv_mean_recall", ORANGE, "Recall")]:
        ax.plot(curve["jaccard_threshold"], curve[column], color=color, linewidth=1, label=name)
        ax.axhline(unfiltered[column], color=color, linestyle=":", linewidth=0.6)
    ax.axvline(fuzzy.JACCARD_THRESHOLD, color=MUTED, linestyle="--", linewidth=0.6)
    ax.annotate(f"θ = {fuzzy.JACCARD_THRESHOLD}", (fuzzy.JACCARD_THRESHOLD, 0.12), xytext=(-2, 0),
                textcoords="offset points", ha="right", fontsize=5.5)
    ax.set_xlim(0.6, 0.985)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("(b) Jaccard threshold θ")
    ax.set_ylabel("Mean CV score")
    ax.legend(loc="lower left", handlelength=1.2)
    ax.grid(color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    return figure


def plot_precision_recall(ranking: pd.DataFrame, treatments: pd.DataFrame) -> Figure:
    """Figure: cross-validated precision and recall of every lexical rule and embedding treatment."""
    figure, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(TEXT_WIDTH, 2.3), layout="constrained")
    selected_fuzzy = ranking[ranking["selected"]].iloc[0]

    ax = axes[0]
    verified = ranking[ranking["jaccard_threshold"].notna()]
    pure = ranking[ranking["jaccard_threshold"].isna()]
    ax.scatter(verified["cv_mean_recall"], verified["cv_mean_precision"], s=1.5, color=BLUE, alpha=0.25,
               linewidths=0, rasterized=True)
    ax.scatter(pure["cv_mean_recall"], pure["cv_mean_precision"], s=3, color=ORANGE, linewidths=0)
    ax.set_xlabel("(a) MinHash–LSH rules: recall")
    ax.set_ylabel("Precision")

    ax = axes[1]
    for model in MODELS:
        rows = treatments[treatments["model"] == model].set_index("representation")
        points = rows[["recall_mean", "precision_mean"]]
        ax.annotate("", xy=tuple(points.loc["question_choices"]), xytext=tuple(points.loc["question"]),
                    arrowprops={"arrowstyle": "-|>", "color": GRID, "linewidth": 0.6, "shrinkA": 2, "shrinkB": 2,
                                "mutation_scale": 5})
    for representation in REPRESENTATIONS:
        rows = treatments[treatments["representation"] == representation]
        ax.scatter(rows["recall_mean"], rows["precision_mean"], s=12, color=REPRESENTATION_COLORS[representation],
                   edgecolors="white", linewidths=0.3, zorder=3)
    chosen = treatments.loc[SELECTED_TREATMENT]
    ax.annotate("Octen-8B", (chosen["recall_mean"], chosen["precision_mean"]), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=5.5)
    ax.set_xlabel("(b) Embedding treatments: recall")

    for ax, point in [(axes[0], (selected_fuzzy["cv_mean_recall"], selected_fuzzy["cv_mean_precision"])),
                      (axes[1], (chosen["recall_mean"], chosen["precision_mean"]))]:
        ax.scatter(*point, marker="*", s=70, color=INK, edgecolors="white", linewidths=0.4, zorder=5)
    axes[1].scatter(selected_fuzzy["cv_mean_recall"], selected_fuzzy["cv_mean_precision"], marker="D", s=14,
                    facecolors="none", edgecolors=INK, linewidths=0.6, zorder=5)

    selected = Line2D([], [], linestyle="", marker="*", markersize=7, color=INK, label="Selected")
    legends = [
        [Line2D([], [], linestyle="", marker="o", markersize=3, color=BLUE, alpha=0.5,
                label=f"LSH + Jaccard filter ({len(verified):,})"),
         Line2D([], [], linestyle="", marker="o", markersize=3, color=ORANGE, label=f"LSH only ({len(pure)})"),
         selected],
        [*(Line2D([], [], linestyle="", marker="o", markersize=3.5, color=REPRESENTATION_COLORS[representation],
                  label=REPRESENTATION_NAMES[representation]) for representation in REPRESENTATIONS),
         selected,
         Line2D([], [], linestyle="", marker="D", markersize=3.5, markerfacecolor="none", markeredgecolor=INK,
                label="Selected fuzzy rule")],
    ]
    for ax, handles in zip(axes, legends, strict=True):
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0.3, 1.01)
        ax.grid(color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.legend(handles=handles, loc="lower left", handlelength=1)
    return figure


def plot_semantic_models(treatments: pd.DataFrame) -> Figure:
    """Figure: mean outer MCC of each model with and without the answer choices."""
    mean = treatments.pivot(index="model", columns="representation", values="mcc_mean")
    std = treatments.pivot(index="model", columns="representation", values="mcc_std")
    order = mean["question_choices"].sort_values(ascending=False).index

    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3), layout="constrained")
    for row, model in enumerate(order):
        ax.plot(mean.loc[model, REPRESENTATIONS], [row, row], color=GRID, linewidth=1.5, zorder=1)
    for representation in REPRESENTATIONS:
        ax.errorbar(mean.loc[order, representation], range(len(order)), xerr=std.loc[order, representation],
                    fmt="o", markersize=3.5, color=REPRESENTATION_COLORS[representation], ecolor=MUTED,
                    elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=3, label=REPRESENTATION_NAMES[representation])
    ax.set_yticks(range(len(order)), [model.removesuffix("-BF16") for model in order])
    selected_model = SELECTED_TREATMENT.split("__")[0]
    ax.get_yticklabels()[list(order).index(selected_model)].set_fontweight("bold")
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.35, 1)
    ax.set_xlabel("Mean outer-CV MCC (± sd over 10 repetitions)")
    ax.grid(axis="x", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    figure.legend(loc="outside upper center", ncol=2)
    return figure


def plot_score_separation(pairs: pd.DataFrame) -> Figure:
    """Figure: lexical similarity against embedding similarity for every panel pair."""
    distance = 1 - pairs[SELECTED_TREATMENT]
    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.6), layout="constrained")
    for label in LABELS:
        mask = pairs["label"] == label
        ax.scatter(pairs.loc[mask, "jaccard"], distance[mask], s=5, marker=LABEL_MARKERS[label],
                   color=LABEL_COLORS[label], edgecolors="white", linewidths=0.2, alpha=0.85,
                   label=f"{LABEL_NAMES[label]} ({int(mask.sum())})", zorder=3)
    ax.axvline(fuzzy.JACCARD_THRESHOLD, color=INK, linestyle="--", linewidth=0.6)
    ax.axhline(1 - semantic.COSINE_THRESHOLD, color=INK, linestyle="--", linewidth=0.6)

    # Counts of duplicates / non-duplicates in each quadrant of the two decision rules.
    semantic_positive = pairs[SELECTED_TREATMENT] >= semantic.COSINE_THRESHOLD
    duplicate = pairs["label"] == "duplicate"
    quadrants = [
        (True, True, (0.99, 0.985), "right", "top", "both"),
        (True, False, (0.02, 0.985), "left", "top", "semantic only"),
        (False, True, (0.99, 0.03), "right", "bottom", "fuzzy only"),
        (False, False, (0.02, 0.03), "left", "bottom", "neither"),
    ]
    for semantic_flag, fuzzy_flag, position, horizontal, vertical, name in quadrants:
        mask = (semantic_positive == semantic_flag) & (pairs["fuzzy_prediction"] == fuzzy_flag)
        ax.text(*position, f"{name}: {int((mask & duplicate).sum())} dup. / {int((mask & ~duplicate).sum())} other",
                transform=ax.transAxes, ha=horizontal, va=vertical, fontsize=5.5, color=MUTED,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.8}, zorder=4)

    ax.set_yscale("log")
    ax.set_ylim(1.6, 4e-5)
    cosine_ticks = [0, 0.5, 0.9, 0.99, 0.999, 0.9999]
    ax.set_yticks([1 - value for value in cosine_ticks], [f"{value:g}" for value in cosine_ticks])
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(0.4, 1)
    ax.set_xlabel(r"Character 8-gram Jaccard $J_8$")
    ax.set_ylabel(r"Octen-8B cosine (log scale of $1 - \cos$)")
    ax.grid(color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    figure.legend(loc="outside upper center", ncol=3, handletextpad=0.2, columnspacing=0.8)
    return figure


def plot_removals_by_exam(stages: pd.DataFrame) -> Figure:
    """Figure: records removed per exam by each deduplication stage, as a share of the filtered exam."""
    shares = 100 * stages[STAGES].div(stages["filtered"], axis=0)
    order = shares.sum(axis=1).sort_values(kind="stable").index

    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.9), layout="constrained")
    left = np.zeros(len(order))
    for stage in STAGES:
        values = shares.loc[order, stage].to_numpy()
        ax.barh(range(len(order)), values, left=left, height=0.7, color=STAGE_COLORS[stage],
                edgecolor="white", linewidth=0.4, label=STAGE_NAMES[stage])
        left += values
    for row, exam in enumerate(order):
        removed = stages.loc[exam, STAGES].sum()
        ax.text(left[row] + 0.4, row, f"{removed:,} / {stages.loc[exam, 'filtered']:,}",
                ha="left", va="center", fontsize=5.5, color=MUTED)

    ax.set_yticks(range(len(order)), [EXAM_LABELS.get(exam, exam) for exam in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, left.max() * 1.3)
    ax.set_xlabel("Removed (% of the exam's filtered records)")
    ax.grid(axis="x", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    figure.legend(loc="outside upper center", ncol=2)
    return figure


def save_figure(figure: Figure, name: str) -> None:
    figure.savefig(FIGURE_DIR / name, metadata={"CreationDate": None}, dpi=600)
    plt.close(figure)


def main() -> int:
    inputs = (FILTERED_DIR, EXACT_DIR, SEMANTIC_OUTPUT_PATH, SWEEP_DIR / "runs")
    missing = [path for path in inputs if not path.exists()]
    if missing:
        print(f"Missing inputs: {missing}. Run src/mmlu_pt/pipeline_minimal.py and the fuzzy ablation first.")
        return 1

    labels = load_labels()
    snapshot = load_snapshot()
    sweep = pd.read_csv(SWEEP_DIR / "metrics.csv")
    exam_removal = sweep_exam_removal(snapshot, sweep["slug"].tolist())
    ranking = pd.read_parquet(SEARCH_DIR / "cv_ranking.parquet")
    pairs = load_semantic_pairs()
    treatments = treatment_summary(pairs)
    stages = stage_counts()
    manifest, removals = production_run()
    print_summary(labels, snapshot, exam_removal, pairs, treatments, stages, manifest, removals)

    plt.rcParams.update(STYLE)
    FIGURE_DIR.mkdir(exist_ok=True)
    save_figure(plot_panel_composition(labels), "panel_composition.pdf")
    save_figure(plot_lsh_sweep(sweep), "lsh_sweep.pdf")
    save_figure(plot_lsh_exam_removal(exam_removal, snapshot), "lsh_exam_removal.pdf")
    save_figure(plot_fuzzy_search(ranking), "fuzzy_search.pdf")
    save_figure(plot_precision_recall(ranking, treatments), "precision_recall.pdf")
    save_figure(plot_semantic_models(treatments), "semantic_models.pdf")
    save_figure(plot_score_separation(pairs), "score_separation.pdf")
    save_figure(plot_removals_by_exam(stages), "removals_by_exam.pdf")
    print(f"Figures written to {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
