"""Generate the figures and verification summary for the length-filter appendix."""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from nemo_curator.stages.text.filters.heuristic import WordCountFilter

from mmlu_pt.mcqa_minimal import serialize_choices, serialize_question_and_choices

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO_ROOT / "output" / "03 - pre-word-filter"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"

# Mirrors the inclusive limits in mmlu_pt.pipelines.definition._word_filter_stages.
QUESTION_MIN_WORDS = 4
QUESTION_MAX_WORDS = 1_000
CHOICES_MAX_WORDS = 300
TOTAL_MAX_WORDS = 1_000

# Threshold grids from the ablation notebooks.
QUESTION_MIN_GRID = [0, 1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50]
QUESTION_MAX_GRID = [100, 200, 300, 500, 750, 1_000, 1_500, 2_000, 3_000]
CHOICES_MAX_GRID = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500]
TOTAL_MAX_GRID = [300, 400, 500, 600, 700, 750, 800, 850, 900, 950, 1_000, 1_050, 1_100, 1_150, 1_200]
QUESTION_MIN_HEATMAP = [3, 4, 5, 8, 10, 20]
CHOICES_MAX_HEATMAP = [200, 225, 250, 275, 300, 350, 400]
TOTAL_MAX_HEATMAP = [750, 800, 850, 900, 950, 1_000, 1_050, 1_100, 1_150, 1_200]
PERCENTILES = [0.5, 0.95, 0.99, 0.995, 0.999]

SEED = 42
WORD_COUNTER = WordCountFilter(min_words=0, max_words=1_000_000, lang="pt")

# ACL page geometry: 7.7 cm columns inside a 16 cm text block.
COLUMN_WIDTH = 7.7 / 2.54
TEXT_WIDTH = 16.0 / 2.54

FILTERS = ["question_short", "question_long", "choices_long", "total_long"]
CAUSE_LABELS = {
    "question_short": r"$w_q < 4$",
    "question_long": r"$w_q$ > 1,000",
    "choices_long": r"$w_c$ > 300",
    "total_long": r"$w_t$ > 1,000",
}
CAUSE_COLORS = {
    "question_short": "#2a78d6",
    "question_long": "#eb6834",
    "choices_long": "#1baf7a",
    "total_long": "#eda100",
}
LEVEL_LABELS = {"high_school": "High school", "undergraduate": "Undergraduate"}
EXAM_LABELS = {"RESIDENCIA_USP_UNICAMP": "Res. USP/Unicamp"}

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
SHADE = "#f0efec"
BOX = "#c3c2b7"
OVERALL_COLOR = "#2a78d6"
WORST_COLOR = "#eb6834"
SWEEP_LINE = {"linewidth": 1, "marker": "o", "markersize": 2.2}
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

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
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
}


def snapshot_digest(paths: list[Path]) -> str:
    """Hash the stage files exactly as the ablation notebooks do."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_records(paths: list[Path]) -> pd.DataFrame:
    """Read the structurally valid records analyzed by the ablations."""
    frames = [pd.read_json(path, lines=True, convert_dates=False) for path in paths]
    return pd.concat(frames, ignore_index=True)[["exam", "academic_level", "question", "choices"]]


def add_word_counts(records: pd.DataFrame) -> pd.DataFrame:
    """Count words with the pipeline's filter over the pipeline's serialized fields."""
    count = WORD_COUNTER.score_document
    return records.assign(
        question_words=records["question"].map(count).astype(int),
        choices_words=records["choices"].map(lambda choices: count(serialize_choices(choices))).astype(int),
        total_words=[
            int(count(serialize_question_and_choices(question=question, choices=choices)))
            for question, choices in zip(records["question"], records["choices"], strict=True)
        ],
        choice_word_sum=records["choices"].map(lambda choices: sum(map(count, choices))).astype(int),
    )


def failed_filters(records: pd.DataFrame) -> pd.DataFrame:
    """Flag each record against every length filter, in pipeline order."""
    return pd.DataFrame(
        {
            "question_short": records["question_words"] < QUESTION_MIN_WORDS,
            "question_long": records["question_words"] > QUESTION_MAX_WORDS,
            "choices_long": records["choices_words"] > CHOICES_MAX_WORDS,
            "total_long": records["total_words"] > TOTAL_MAX_WORDS,
        }
    )


def add_removal_stage(records: pd.DataFrame) -> pd.DataFrame:
    """Attribute each removed record to the first filter it fails."""
    failed = failed_filters(records)
    stage = failed.idxmax(axis=1).where(failed.any(axis=1), "kept")
    return pd.concat([records, failed], axis=1).assign(stage=stage)


def combined_filter_population(records: pd.DataFrame) -> pd.DataFrame:
    """Return D2, the records that reach the combined-length filter."""
    return records[records["stage"].isin(["kept", "total_long"])]


def threshold_mask(words: pd.Series, threshold: int, side: str) -> pd.Series:
    """Keep records on the inclusive side of a lower or upper bound."""
    return words >= threshold if side == "lower" else words <= threshold


def retention_metrics(population: pd.DataFrame, kept: pd.Series) -> dict[str, float | str]:
    """Summarize retention overall, by academic level, and for the worst exam."""
    by_exam = kept.groupby(population["exam"]).mean()
    by_level = kept.groupby(population["academic_level"]).mean()
    share_before = population["exam"].value_counts(normalize=True)
    share_after = (
        population.loc[kept, "exam"].value_counts(normalize=True).reindex(share_before.index, fill_value=0)
    )
    return {
        "retention": 100 * kept.mean(),
        "high_school": 100 * by_level["high_school"],
        "undergraduate": 100 * by_level["undergraduate"],
        "worst_exam": by_exam.idxmin(),
        "worst_exam_retention": 100 * by_exam.min(),
        "share_shift_pp": 100 * (share_after - share_before).abs().max(),
    }


def sweep(population: pd.DataFrame, column: str, thresholds: list[int], side: str) -> pd.DataFrame:
    """Compute retention metrics for every threshold in a grid."""
    rows = {
        threshold: retention_metrics(population, threshold_mask(population[column], threshold, side))
        for threshold in thresholds
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def exam_rows(records: pd.DataFrame) -> list[tuple[str, str | None]]:
    """Order exams by level and median question length, with one header row per level."""
    medians = records.groupby(["academic_level", "exam"])["question_words"].median()
    rows = []
    for level, level_label in LEVEL_LABELS.items():
        rows.append((level_label, None))
        for exam in medians[level].sort_values(ascending=False).index:
            rows.append((EXAM_LABELS.get(exam, exam), exam))
    return rows


def format_thousands(value: float, _position: int = 0) -> str:
    return f"{value:,.0f}"


def style_row_axis(ax: plt.Axes, rows: list[tuple[str, str | None]]) -> None:
    """Label exam rows and set the level header rows in bold."""
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    for tick_label, (_, exam) in zip(ax.get_yticklabels(), rows, strict=True):
        if exam is None:
            tick_label.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)


def print_summary(records: pd.DataFrame, digest: str) -> None:
    """Print the numbers quoted in the appendix for comparison with the notebooks."""
    failed = records[FILTERS]
    kept = records["stage"] == "kept"
    eligible = combined_filter_population(records)

    print(f"Snapshot: {len(records):,} records, {records['exam'].nunique()} exams, sha256={digest}")
    print("Records by level:", records["academic_level"].value_counts().to_dict())
    print("Records by choice count:", records["choices"].map(len).value_counts().sort_index().to_dict())
    shortest_choice = records["choices"].map(lambda choices: min(map(WORD_COUNTER.score_document, choices)))
    print("Minimum words in a choice:", shortest_choice.min())
    print(
        "Additivity:",
        bool((records["choices_words"] == records["choice_word_sum"]).all()),
        bool((records["total_words"] == records["question_words"] + records["choice_word_sum"]).all()),
    )

    for column, population, name in [
        ("question_words", records, "D0"),
        ("choices_words", records, "D0"),
        ("total_words", eligible, "D2"),
    ]:
        quantiles = population[column].quantile(PERCENTILES).round(3).to_dict()
        words = population[column]
        print(f"{column} on {name}: min={words.min()} {quantiles} max={words.max()}")

    removed = records["stage"].value_counts().reindex(FILTERS, fill_value=0)
    only = {name: int((failed[name] & ~failed.drop(columns=name).any(axis=1)).sum()) for name in FILTERS}
    print("Removed by stage:", removed.to_dict())
    print("Retained after each stage:", (len(records) - removed.cumsum()).to_dict())
    print("Fails alone:", failed.sum().to_dict())
    print("Fails only this filter:", only)
    print("Fails any filter:", int(failed.any(axis=1).sum()))
    question_failed = failed["question_short"] | failed["question_long"]
    print("Choices overlap with question filter:", int((failed["choices_long"] & question_failed).sum()))

    total_removed = records[records["stage"] == "total_long"]
    print("Combined-filter removals by exam:", total_removed["exam"].value_counts().to_dict())
    print("Combined-filter removals, minimum question words:", total_removed["question_words"].min())
    long_choice_counts = records.loc[failed["choices_long"], "choices"].map(len)
    print("Choices over the limit by choice count:", long_choice_counts.value_counts().to_dict())
    print("End to end:", retention_metrics(records, kept))

    per_exam = records.groupby("exam").agg(
        level=("academic_level", "first"),
        before=("stage", "size"),
        after=("stage", lambda stage: int((stage == "kept").sum())),
    )
    per_exam["retention_pct"] = (100 * per_exam["after"] / per_exam["before"]).round(3)
    stage_counts = pd.crosstab(records["exam"], records["stage"]).reindex(columns=FILTERS, fill_value=0)
    print(per_exam.join(stage_counts).sort_values(["level", "before"], ascending=[True, False]).to_string())


def draw_length_panel(
    ax: plt.Axes,
    population: pd.DataFrame,
    column: str,
    rows: list[tuple[str, str | None]],
    bounds: list[tuple[int, str]],
    causes: list[str],
    xlabel: str,
) -> None:
    """Draw per-exam length boxes, the removed records, and the selected bounds."""
    positions = {exam: index for index, (_, exam) in enumerate(rows) if exam is not None}
    ax.boxplot(
        [population.loc[population["exam"] == exam, column] for exam in positions],
        positions=list(positions.values()),
        orientation="horizontal",
        whis=(5, 95),
        showfliers=False,
        widths=0.6,
        patch_artist=True,
        boxprops={"facecolor": GRID, "edgecolor": MUTED, "linewidth": 0.5},
        medianprops={"color": INK, "linewidth": 0.8},
        whiskerprops={"color": MUTED, "linewidth": 0.5},
        capprops={"color": MUTED, "linewidth": 0.5},
    )

    rng = np.random.default_rng(SEED)
    for cause in causes:
        removed = population[population[cause]]
        jitter = rng.uniform(-0.28, 0.28, len(removed))
        ax.scatter(
            removed[column],
            removed["exam"].map(positions) + jitter,
            s=2.5,
            color=CAUSE_COLORS[cause],
            linewidths=0,
            zorder=3,
        )

    ax.set_xscale("log")
    xmin, xmax = population[column].min() / 1.6, population[column].max() * 1.6
    ax.set_xlim(xmin, xmax)
    header_row = next(index for index, (_, exam) in enumerate(rows) if exam is None)
    for bound, side in bounds:
        removed_range = (xmin, bound) if side == "lower" else (bound, xmax)
        ax.axvspan(*removed_range, color=SHADE, zorder=0)
        ax.axvline(bound, color=INK, linestyle="--", linewidth=0.6, zorder=2)
        ax.annotate(
            format_thousands(bound),
            (bound, header_row),
            xytext=(2, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6,
        )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel(xlabel)


def plot_lengths_by_exam(records: pd.DataFrame, rows: list[tuple[str, str | None]]) -> Figure:
    """Figure: per-exam distributions of the three filtered lengths."""
    figure, axes = plt.subplots(1, 3, sharey=True, figsize=(TEXT_WIDTH, 2.9), layout="constrained")
    panels = [
        (records, "question_words", [(QUESTION_MIN_WORDS, "lower"), (QUESTION_MAX_WORDS, "upper")],
         ["question_short", "question_long"], r"(a) Question words $w_q$ on $D_0$"),
        (records, "choices_words", [(CHOICES_MAX_WORDS, "upper")],
         ["choices_long"], r"(b) Choice words $w_c$ on $D_0$"),
        (combined_filter_population(records), "total_words", [(TOTAL_MAX_WORDS, "upper")],
         ["total_long"], r"(c) Combined words $w_t$ on $D_2$"),
    ]
    for ax, (population, column, bounds, causes, xlabel) in zip(axes, panels, strict=True):
        draw_length_panel(ax, population, column, rows, bounds, causes, xlabel)
    style_row_axis(axes[0], rows)
    axes[0].invert_yaxis()

    handles = [Patch(facecolor=GRID, edgecolor=MUTED, linewidth=0.5, label="IQR, median, p5–p95")]
    handles += [
        Line2D([], [], linestyle="", marker="o", markersize=2.5, color=CAUSE_COLORS[cause],
               label=f"Removed: {CAUSE_LABELS[cause]}")
        for cause in FILTERS
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    return figure


def plot_threshold_sweeps(records: pd.DataFrame) -> Figure:
    """Figure: overall and worst-exam retention along each threshold grid."""
    eligible = combined_filter_population(records)
    panels = [
        (records, "question_words", QUESTION_MIN_GRID, "lower", QUESTION_MIN_WORDS,
         r"(a) Minimum $w_q$ on $D_0$", "linear"),
        (records, "question_words", QUESTION_MAX_GRID, "upper", QUESTION_MAX_WORDS,
         r"(b) Maximum $w_q$ on $D_0$", "log"),
        (records, "choices_words", CHOICES_MAX_GRID, "upper", CHOICES_MAX_WORDS,
         r"(c) Maximum $w_c$ on $D_0$", "linear"),
        (eligible, "total_words", TOTAL_MAX_GRID, "upper", TOTAL_MAX_WORDS,
         r"(d) Maximum $w_t$ on $D_2$", "linear"),
    ]
    figure, axes = plt.subplots(1, 4, sharey=True, figsize=(TEXT_WIDTH, 1.95), layout="constrained")
    for ax, (population, column, grid, side, selected, xlabel, scale) in zip(axes, panels, strict=True):
        metrics = sweep(population, column, grid, side)
        ax.plot(grid, metrics["retention"], color=OVERALL_COLOR, **SWEEP_LINE)
        ax.plot(grid, metrics["worst_exam_retention"], color=WORST_COLOR, **SWEEP_LINE)
        ax.axvline(selected, color=INK, linestyle="--", linewidth=0.6)

        # The label sits on the retained side of the bound, below the curves.
        chosen = metrics.loc[selected]
        ax.annotate(
            f"{format_thousands(selected)}\n{chosen['worst_exam']} {chosen['worst_exam_retention']:.1f}",
            (selected, 3),
            xytext=(2 if side == "lower" else -2, 0),
            textcoords="offset points",
            ha="left" if side == "lower" else "right",
            va="bottom",
            fontsize=5.5,
        )

        ax.set_xscale(scale)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylim(0, 103)
    axes[0].set_ylabel("Retention (%)")
    handles = [
        Line2D([], [], color=OVERALL_COLOR, label="All records", **SWEEP_LINE),
        Line2D([], [], color=WORST_COLOR, label="Least-retained exam", **SWEEP_LINE),
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=2)
    return figure


def format_retention(value: float) -> str:
    """Round to one decimal without printing 100 for an exam that lost records."""
    return "100" if value == 100 else f"{min(value, 99.94):.1f}"


def plot_exam_sensitivity(records: pd.DataFrame, rows: list[tuple[str, str | None]]) -> Figure:
    """Figure: per-exam retention for candidate thresholds of each filter."""
    panels = [
        (records, "question_words", QUESTION_MIN_HEATMAP, "lower", QUESTION_MIN_WORDS,
         r"(a) Minimum $w_q$ on $D_0$"),
        (records, "choices_words", CHOICES_MAX_HEATMAP, "upper", CHOICES_MAX_WORDS,
         r"(b) Maximum $w_c$ on $D_0$"),
        (combined_filter_population(records), "total_words", TOTAL_MAX_HEATMAP, "upper", TOTAL_MAX_WORDS,
         r"(c) Maximum $w_t$ on $D_2$"),
    ]
    exams = [exam for _, exam in rows]
    colormap = LinearSegmentedColormap.from_list("retention", BLUE_RAMP[::-1])
    colormap.set_bad("none")
    norm = Normalize(vmin=80, vmax=100, clip=True)

    figure, axes = plt.subplots(
        1, 3, sharey=True, figsize=(TEXT_WIDTH, 3.0), layout="constrained",
        gridspec_kw={"width_ratios": [len(panel[2]) for panel in panels]},
    )
    for ax, (population, column, thresholds, side, selected, xlabel) in zip(axes, panels, strict=True):
        retention = pd.DataFrame(
            {
                threshold: 100 * threshold_mask(population[column], threshold, side)
                .groupby(population["exam"])
                .mean()
                for threshold in thresholds
            }
        ).reindex(exams)
        values = retention.to_numpy(dtype=float)
        image = ax.imshow(np.ma.masked_invalid(values), cmap=colormap, norm=norm, aspect="auto")
        for (row, col), value in np.ndenumerate(values):
            if np.isnan(value):
                continue
            ax.text(col, row, format_retention(value), ha="center", va="center", fontsize=5,
                    color="white" if value < 91 else INK)

        column_index = thresholds.index(selected)
        ax.add_patch(
            Rectangle((column_index - 0.5, -0.5), 1, len(rows), fill=False, edgecolor=INK, linewidth=0.8)
        )
        ax.set_xticks(range(len(thresholds)), [format_thousands(threshold) for threshold in thresholds])
        ax.set_xlabel(xlabel)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    style_row_axis(axes[0], rows)
    colorbar = figure.colorbar(
        image, ax=axes, extend="min", shrink=0.7, aspect=30, pad=0.01, ticks=range(80, 101, 5)
    )
    colorbar.set_label("Exam retention (%)")
    colorbar.outline.set_visible(False)
    return figure


def plot_question_vs_choices(records: pd.DataFrame) -> Figure:
    """Figure: joint question and choice lengths in D2 and the region the combined filter removes."""
    eligible = combined_filter_population(records)
    corner = TOTAL_MAX_WORDS - CHOICES_MAX_WORDS
    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3), layout="constrained")
    ax.add_patch(
        Polygon(
            [(corner, CHOICES_MAX_WORDS), (QUESTION_MAX_WORDS, CHOICES_MAX_WORDS),
             (QUESTION_MAX_WORDS, TOTAL_MAX_WORDS - QUESTION_MAX_WORDS)],
            closed=True, facecolor=SHADE, edgecolor="none", zorder=0,
        )
    )
    hexbin = ax.hexbin(
        eligible["question_words"],
        eligible["choices_words"],
        gridsize=40,
        extent=(0, QUESTION_MAX_WORDS, 0, CHOICES_MAX_WORDS),
        mincnt=1,
        norm=LogNorm(),
        cmap=LinearSegmentedColormap.from_list("density", BLUE_RAMP),
        linewidths=0.1,
        edgecolors="face",
        zorder=1,
    )
    ax.plot([corner, QUESTION_MAX_WORDS], [CHOICES_MAX_WORDS, TOTAL_MAX_WORDS - QUESTION_MAX_WORDS],
            color=INK, linestyle="--", linewidth=0.6, zorder=2)

    removed = eligible[eligible["total_long"]]
    ax.scatter(removed["question_words"], removed["choices_words"], s=5, color=CAUSE_COLORS["total_long"],
               edgecolors=INK, linewidths=0.3, zorder=3)
    ax.annotate(
        f"{len(removed)} removed\n" + r"($w_t$ > 1,000)",
        xy=(930, 140),
        xytext=(560, 250),
        ha="center",
        va="center",
        fontsize=6,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5},
    )

    ax.set_xlim(0, QUESTION_MAX_WORDS + 15)
    ax.set_ylim(0, CHOICES_MAX_WORDS + 5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
    ax.set_xlabel(r"Question words $w_q$")
    ax.set_ylabel(r"Choice words $w_c$")
    colorbar = figure.colorbar(hexbin, ax=ax, pad=0.01)
    colorbar.set_label("Records per bin")
    colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))
    colorbar.ax.minorticks_off()
    colorbar.outline.set_visible(False)
    return figure


def plot_removals_by_exam(records: pd.DataFrame) -> Figure:
    """Figure: end-to-end removals per exam, stacked by the filter that removed them."""
    counts = pd.crosstab(records["exam"], records["stage"]).reindex(columns=FILTERS, fill_value=0)
    totals = records["exam"].value_counts()
    shares = 100 * counts.div(totals, axis=0)
    order = shares.sum(axis=1).sort_values(kind="stable").index

    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.9), layout="constrained")
    left = np.zeros(len(order))
    for cause in FILTERS:
        values = shares.loc[order, cause].to_numpy()
        ax.barh(range(len(order)), values, left=left, height=0.7, color=CAUSE_COLORS[cause],
                edgecolor="white", linewidth=0.4, label=CAUSE_LABELS[cause])
        left += values
    for row, exam in enumerate(order):
        ax.text(left[row] + 0.15, row, f"{counts.loc[exam].sum():,} / {totals[exam]:,}",
                ha="left", va="center", fontsize=5.5, color=MUTED)

    ax.set_yticks(range(len(order)), [EXAM_LABELS.get(exam, exam) for exam in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, left.max() * 1.35)
    ax.set_xlabel("Removed by the length filters (% of exam)")
    ax.grid(axis="x", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    figure.legend(loc="outside upper center", ncol=2)
    return figure


def save_figure(figure: Figure, name: str) -> None:
    figure.savefig(FIGURE_DIR / name, metadata={"CreationDate": None})
    plt.close(figure)


def main() -> int:
    paths = sorted(INPUT_DIR.glob("*.jsonl"))
    if not paths:
        print(f"No JSONL files found in {INPUT_DIR}. Run src/mmlu_pt/pipeline_minimal.py first.")
        return 1

    records = add_removal_stage(add_word_counts(load_records(paths)))
    print_summary(records, snapshot_digest(paths))

    plt.rcParams.update(STYLE)
    FIGURE_DIR.mkdir(exist_ok=True)
    rows = exam_rows(records)
    save_figure(plot_lengths_by_exam(records, rows), "lengths_by_exam.pdf")
    save_figure(plot_threshold_sweeps(records), "threshold_sweeps.pdf")
    save_figure(plot_exam_sensitivity(records, rows), "exam_sensitivity.pdf")
    save_figure(plot_question_vs_choices(records), "question_vs_choices.pdf")
    save_figure(plot_removals_by_exam(records), "removals_by_exam.pdf")
    print(f"Figures written to {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
