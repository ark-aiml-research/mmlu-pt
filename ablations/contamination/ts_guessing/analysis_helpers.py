"""Scoring, coverage and exports for the TS-Guessing analysis notebook."""

from collections import Counter
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from artifacts import read_json, read_jsonl

CONDITIONS = ("without_source", "with_source")
PRIMARY_METRIC = "em_normalized"
METRICS = (
    "em_strict", "em_normalized", "em_label_tolerant", "token_f1",
    "empty_output", "copied_visible_choice", "length_limited",
)
STATUSES = ("completed", "context_overflow", "request_error", "not_run")
QUESTION_KEYS = (
    "model", "condition", "question_id", "exam", "subject", "macro_area", "choice_count",
)
RESPONSE_COLUMNS = (
    "task_id", "status", "text", "finish_reason", "prompt_tokens", "completion_tokens",
    "stage", "error", "session_id", "attempt", "batch_id", "rank",
)
EXPORT_CHUNK_SIZE = 20_000


def save_frame(directory: Path, name: str, frame: pd.DataFrame,
               *, jsonl: bool = False, append: bool = False) -> None:
    """Write bounded chunks; JSONL retains nested alternatives and missing scores."""
    csv_frame = frame.drop(columns="visible_choices", errors="ignore")
    csv_frame.to_csv(directory / f"{name}.csv", index=False, mode="a" if append else "w",
                     header=not append, chunksize=EXPORT_CHUNK_SIZE)
    if not jsonl:
        return
    with (directory / f"{name}.jsonl").open("a" if append else "w", encoding="utf-8") as stream:
        for start in range(0, len(frame), EXPORT_CHUNK_SIZE):
            frame.iloc[start:start + EXPORT_CHUNK_SIZE].to_json(
                stream, orient="records", lines=True, force_ascii=False,
            )


def validate_tasks(sample: pd.DataFrame, tasks: pd.DataFrame, manifest: dict) -> None:
    if sample.empty or tasks.empty:
        raise ValueError("The recorded sample and tasks must be nonempty.")
    if sample.question_id.duplicated().any() or tasks.task_id.duplicated().any():
        raise ValueError("Duplicate question or task IDs in the recorded artifacts.")
    if len(sample) != manifest["sample_size"] or len(tasks) != manifest["tasks_per_model"]:
        raise ValueError("Sample/task counts disagree with the manifest.")
    if set(tasks.question_id) != set(sample.question_id) or set(tasks.condition) != set(CONDITIONS):
        raise ValueError("Tasks must cover the recorded sample in both prompt conditions.")
    if tasks.duplicated(["question_id", "condition", "mask_index"]).any():
        raise ValueError("Duplicate masked positions within a question and condition.")

    expected = sample.set_index("question_id").choices.map(len) - 1
    observed = tasks.groupby(["question_id", "condition"]).size().unstack()
    if not observed.reindex(expected.index).eq(expected, axis=0).all().all():
        raise ValueError("Each question must include every distractor in both conditions.")
    source = sample.set_index("question_id")
    choice_counts = tasks.question_id.map(source.choices.map(len))
    answers = tasks.question_id.map(source.answer)
    if (tasks.choice_count.ne(choice_counts).any() or tasks.mask_index.eq(answers).any()
            or tasks.mask_index.lt(0).any() or tasks.mask_index.ge(choice_counts).any()):
        raise ValueError("Masked positions disagree with the recorded question alternatives.")

    paired_fields = [
        "exam", "exam_edition", "num", "subject", "macro_area", "academic_level",
        "mask_label", "choice_count", "target", "target_words", "target_numeric",
        "target_in_context", "visible_choices",
    ]
    left = tasks[tasks.condition.eq(CONDITIONS[0])].set_index(["question_id", "mask_index"])
    right = tasks[tasks.condition.eq(CONDITIONS[1])].set_index(["question_id", "mask_index"])
    if not left[paired_fields].sort_index().equals(right[paired_fields].sort_index()):
        raise ValueError("Prompt conditions must have identical masks, targets and question metadata.")


def load_model(run_dir: Path, spec: dict, manifest: dict) -> tuple[dict, pd.DataFrame]:
    directory = run_dir / "models" / spec["name"]
    metadata_path = directory / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    if metadata and (metadata["spec"] != spec or metadata["run_identity"] != manifest["identity"]):
        raise ValueError(f"Model metadata mismatch: {spec['name']}")
    description = {**manifest["model_descriptions"].get(spec["name"], {}),
                   **metadata.get("checkpoint", {})}
    total = description.get("total_parameters")
    model = {
        "model": spec["name"], "family": description.get("family", spec["name"]),
        "repo": spec["repo"], "revision": description.get("revision"),
        "max_model_len": metadata.get("server", {}).get(
            "max_model_len", spec.get("max_model_len", manifest["config"]["server"]["max_model_len"])),
        "total_parameters_b": total / 1e9 if total else description.get("nominal_parameters_b"),
        "parameters_source": "checkpoint metadata" if total else "model plan / nominal",
        **{key: description.get(key) for key in (
            "nominal_parameters_b", "active_parameters_b", "effective_parameters_b")},
        "execution_status": manifest["models"].get(spec["name"], {}).get("status", "not_started"),
    }
    latest = {}
    for row in read_jsonl(directory / "responses.jsonl"):
        usage = row.get("usage") or {}
        latest[row["task_id"]] = {
            "task_id": row["task_id"], "status": row["status"], "text": row.get("text", ""),
            "finish_reason": row.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            **{key: row.get(key) for key in ("stage", "error", "session_id", "attempt", "batch_id", "rank")},
        }
    return model, pd.DataFrame(latest.values(), columns=RESPONSE_COLUMNS)


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def score_response(text: str, target: str, label: str, visible_choices: list[str],
                   finish_reason: str | None) -> dict:
    prediction, target = text.strip(), target.strip()
    predicted, expected = normalize_text(prediction), normalize_text(target)
    pattern = rf"^\s*(?:\({label}\)|\[{label}\]|{label}\s*[.):\-])\s+"
    tolerant = normalize_text(re.sub(pattern, "", prediction, count=1).strip())
    predicted_tokens, expected_tokens = Counter(predicted.split()), Counter(expected.split())
    overlap = sum((predicted_tokens & expected_tokens).values())
    denominator = sum(predicted_tokens.values()) + sum(expected_tokens.values())
    visible = [normalize_text(choice) for choice in visible_choices]
    return {
        "em_strict": float(bool(target) and prediction == target),
        "em_normalized": float(bool(expected) and predicted == expected),
        "em_label_tolerant": float(bool(expected) and tolerant == expected),
        "token_f1": 2 * overlap / denominator if denominator else 0.0,
        "empty_output": float(not prediction),
        "copied_visible_choice": float(bool(tolerant) and tolerant in visible),
        "length_limited": float(finish_reason == "length"),
    }


def analyze_responses(tasks: pd.DataFrame, responses: pd.DataFrame,
                      model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unknown_tasks = set(responses.task_id) - set(tasks.task_id)
    if unknown_tasks or not responses.status.isin(STATUSES[:-1]).all():
        raise ValueError(f"Unknown task IDs or response statuses for {model}.")
    grid = tasks.merge(responses, on="task_id", how="left", validate="one_to_one")
    grid.insert(0, "model", model)
    grid["status"] = grid.status.fillna("not_run")
    grid[["subject", "macro_area"]] = grid[["subject", "macro_area"]].fillna("<missing>")
    completed = grid.loc[grid.status.eq("completed")].reset_index(drop=True)
    inputs = ["text", "target", "mask_label", "visible_choices", "finish_reason"]
    metric_rows = [score_response(*row) for row in completed[inputs].itertuples(index=False, name=None)]
    scored = pd.concat([completed, pd.DataFrame(metric_rows, columns=METRICS)], axis=1)
    coverage = grid.assign(is_completed=grid.status.eq("completed")).groupby(
        list(QUESTION_KEYS), dropna=False,
    ).agg(expected_masks=("task_id", "size"), completed_masks=("is_completed", "sum")).reset_index()
    means = scored.groupby(list(QUESTION_KEYS), dropna=False)[list(METRICS)].mean().reset_index()
    questions = coverage.merge(means, on=list(QUESTION_KEYS), how="left", validate="one_to_one")
    questions["all_masks_completed"] = questions.completed_masks.eq(questions.expected_masks)
    questions.loc[~questions.all_masks_completed, list(METRICS)] = np.nan
    return grid, scored, questions


def predictability_summary(scored: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        group = scored[scored.condition.eq(condition)]
        restricted = group[group.target_words.ge(8) & ~group.target_numeric & ~group.target_in_context]
        rows.append({
            "model": model, "condition": condition,
            "completed_masks": len(group), "mean_em": group[PRIMARY_METRIC].mean(),
            "less_predictable_masks": len(restricted),
            "less_predictable_em": restricted[PRIMARY_METRIC].mean(),
        })
    return pd.DataFrame(rows)


def remember_matches(scored: pd.DataFrame, examples: dict) -> None:
    """Keep one representative exact reconstruction per question for later review."""
    columns = ["question_id", "model", "condition", "task_id", "mask_label", "target", "text",
               "target_words", "target_numeric", "target_in_context"]
    matches = scored.loc[scored[PRIMARY_METRIC].eq(1), columns].sort_values(
        ["target_words", "condition", "task_id"], ascending=[False, False, True],
    ).drop_duplicates("question_id")
    for row in matches.to_dict("records"):
        previous = examples.get(row["question_id"])
        priority = (row["target_words"], row["condition"] == "without_source")
        if previous is None or priority > (previous["target_words"], previous["condition"] == "without_source"):
            examples[row["question_id"]] = row


def bootstrap_interval(frame: pd.DataFrame, metric: str,
                       replicates: int, seed: int) -> tuple[float, float, float]:
    groups = [group[metric].dropna().to_numpy(dtype=float)
              for _, group in frame.groupby("exam", sort=True)]
    groups = [values for values in groups if len(values)]
    if not groups:
        return np.nan, np.nan, np.nan
    weights = np.array([len(values) for values in groups], dtype=float)
    weights /= weights.sum()
    estimate = sum(weight * values.mean() for weight, values in zip(weights, groups))
    rng = np.random.default_rng(seed)
    draws = np.zeros(replicates)
    for weight, values in zip(weights, groups):
        for start in range(0, replicates, 128):
            stop = min(start + 128, replicates)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            draws[start:stop] += weight * values[indices].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return estimate, lower, upper


def summarize_questions(questions: pd.DataFrame, models: pd.DataFrame, sample_size: int,
                        replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = questions[questions.all_masks_completed]
    summary_rows, hint_rows = [], []
    for model in models.model:
        model_questions = complete[complete.model.eq(model)]
        for condition in CONDITIONS:
            group = model_questions[model_questions.condition.eq(condition)]
            point, lower, upper = bootstrap_interval(group, PRIMARY_METRIC, replicates, seed)
            summary_rows.append({
                "model": model, "condition": condition, "complete_questions": len(group),
                "covered_exams": group.exam.nunique(), "question_coverage": len(group) / sample_size,
                **{metric: group[metric].mean() for metric in METRICS},
                "sample_em": point, "sample_lower": lower, "sample_upper": upper,
            })
        left = model_questions[model_questions.condition.eq(CONDITIONS[0])]
        right = model_questions[model_questions.condition.eq(CONDITIONS[1])]
        shared = left[["question_id", "exam", PRIMARY_METRIC]].merge(
            right[["question_id", PRIMARY_METRIC]], on="question_id",
            suffixes=("_left", "_right"), validate="one_to_one",
        )
        shared["difference"] = shared[f"{PRIMARY_METRIC}_right"] - shared[f"{PRIMARY_METRIC}_left"]
        point, lower, upper = bootstrap_interval(shared, "difference", replicates, seed)
        hint_rows.append({
            "model": model, "left": CONDITIONS[0], "right": CONDITIONS[1], "condition": "source_hint",
            "shared_questions": len(shared), "covered_exams": shared.exam.nunique(),
            "sample_difference": point, "sample_lower": lower, "sample_upper": upper,
        })
    summary = pd.DataFrame(summary_rows).merge(models, on="model", validate="many_to_one")
    return summary, pd.DataFrame(hint_rows)


def build_question_review(questions: pd.DataFrame, models: pd.DataFrame,
                          sample: pd.DataFrame, tasks: pd.DataFrame) -> tuple:
    per_model = questions.groupby(["question_id", "model"], dropna=False).agg(
        completed_conditions=("all_masks_completed", "sum"),
        reconstruction_score=(PRIMARY_METRIC, "mean"),
    ).reset_index()
    per_model["model_complete"] = per_model.completed_conditions.eq(len(CONDITIONS))
    per_model.loc[~per_model.model_complete, "reconstruction_score"] = np.nan
    per_model = per_model.merge(models[["model", "family"]], on="model", validate="many_to_one")
    family_sizes = models.groupby("family").size().rename("expected_models")
    family_scores = per_model.groupby(["question_id", "family"]).agg(
        completed_models=("model_complete", "sum"), family_score=("reconstruction_score", "mean"),
    ).reset_index().merge(family_sizes, on="family", validate="many_to_one")
    family_scores["family_complete"] = family_scores.completed_models.eq(family_scores.expected_models)
    family_scores.loc[~family_scores.family_complete, "family_score"] = np.nan

    review = family_scores.groupby("question_id").agg(
        complete_families=("family_complete", "sum"), complete_models=("completed_models", "sum"),
        consensus_score=("family_score", "mean"),
    ).reindex(sample.question_id).reset_index()
    review["expected_models"] = len(models)
    review["expected_families"] = len(family_sizes)
    review[["complete_models", "complete_families"]] = review[["complete_models", "complete_families"]].fillna(0).astype(int)
    review["model_coverage"] = review.complete_models / review.expected_models
    review["eligible"] = review.complete_models.eq(review.expected_models) & review.complete_families.eq(review.expected_families)
    review.loc[~review.eligible, "consensus_score"] = np.nan

    masks = tasks[tasks.condition.eq(CONDITIONS[0])].copy()
    masks["short_target"] = masks.target_words.between(1, 3)
    diagnostics = masks.groupby("question_id").agg(
        distractors=("task_id", "size"), short_targets=("short_target", "sum"),
        numeric_targets=("target_numeric", "sum"), targets_in_context=("target_in_context", "sum"),
    ).reset_index()
    locators = ["question_id", "source_row_index", "exam", "exam_edition", "num", "subject", "macro_area"]
    review = review.merge(sample[locators], on="question_id", validate="one_to_one").merge(
        diagnostics, on="question_id", how="left", validate="one_to_one",
    )
    review = review.merge(
        family_scores.pivot(index="question_id", columns="family", values="family_score").add_prefix("family_score_"),
        on="question_id", how="left", validate="one_to_one",
    )
    ranking = review[review.eligible].sort_values(
        ["consensus_score", "question_id"], ascending=[False, True],
    ).copy()
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    review = review.merge(ranking[["question_id", "rank"]], on="question_id", how="left", validate="one_to_one")
    review["rank"] = review["rank"].astype("Int64")
    return per_model, family_scores, review, ranking
