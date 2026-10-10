"""Generate the figures and verification summary for the knowledge-area annotation appendix."""

import argparse
import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from mmlu_pt.annotation.knowledge_area.config import DEFAULT_ALIASES, PACKAGE_DIR, UNCERTAIN
from mmlu_pt.annotation.knowledge_area.dataset import annotation_id
from mmlu_pt.annotation.knowledge_area.taxonomy import Taxonomy, load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = Path(__file__).resolve().parent
FIGURE_DIR = ABLATION_DIR / "figures"

WORK_DIR = REPO_ROOT / "output" / "knowledge-annotation-work"
FULL_RUNS = {"1.1": WORK_DIR / "full-run", "1.2": WORK_DIR / "full-run-2", "1.4": WORK_DIR / "full-run-taxonomy-v1.4"}
SUBSET_RUNS = {"1.3": WORK_DIR / "uncertain-taxonomy-v1.3", "1.4": WORK_DIR / "uncertain-taxonomy-v1.4"}
POST_DIR = FULL_RUNS["1.4"] / "post-annotation"
BENCHMARK_DIR = REPO_ROOT / "output" / "fewshot-work" / "main"
DEDUPLICATED_DIR = REPO_ROOT / "output" / "06 - revised" / "huggingface"
MANUAL_PATH = REPO_ROOT / "config" / "manual_annotations.json"
REMOVALS_PATH = REPO_ROOT / "config" / "removed_questions.json"
EXCLUDED_TEST_PATH = REPO_ROOT / "config" / "excluded_test_questions.json"

# Taxonomy 1.0 was committed as mmlu_pt_taxonomy.json in b2aec7e and deleted in 66d2f73.
TAXONOMY_V1_0_PATH = ABLATION_DIR / "assets" / "mmlu_pt_taxonomy_v1_0.json"
TAXONOMY_V1_0_SHA256 = "5be8e69c775b47698676538ea2fe622e833ba77952bf2be5bf4fd1c5b582f957"
TAXONOMY_PATHS = {"1.0": TAXONOMY_V1_0_PATH,
                  **{f"1.{n}": PACKAGE_DIR / f"mmlu_pt_taxonomy_v1_{n}.json" for n in (1, 2, 3, 4)}}
VERSIONS = list(TAXONOMY_PATHS)
EXPECTED_RUN_IDS = {"1.1": "a777aaac987d0966", "1.2": "4e053d1ce260f6ae", "1.4": "1640098255bb0bed",
                    "1.3-subset": "9f4f7174e7631904", "1.4-subset": "130ddfa5eae044b3"}

STAGES = [("independent", 1), ("independent", 2), ("adjudication", 0)]
STAGE_NAMES = {("independent", 1): "Pass 1", ("independent", 2): "Pass 2", ("adjudication", 0): "Adjudication"}
MIN_MACRO_AREA = 2000
TRANSITION_LIMIT = 10

LEVELS = ["high_school", "undergraduate"]
LEVEL_LABELS = {"high_school": "High school", "undergraduate": "Undergraduate"}
EXAM_LABELS = {"RESIDENCIA_USP_UNICAMP": "Res. USP/Unicamp", "USP_UNICAMP_MEDICAL_RESIDENCY": "Res. USP/Unicamp"}
MACRO_AREA_LABELS = {
    "Languages, Literature, Arts, and Communication": "Languages,\nLit., Arts,\nComm.",
    "Mathematics, Statistics, and Logic": "Mathematics,\nStatistics,\nLogic",
    "Computing, Engineering, Architecture, and Design": "Computing,\nEng., Arch.,\nDesign",
    "Natural, Earth, Agricultural and Veterinary Sciences": "Natural, Earth,\nAgric., Vet.\nSciences",
    "Humanities": "Humanities",
    "Social and Applied Social Sciences": "Social and\nApplied Social\nSciences",
    "Economics, Business, and Accounting": "Economics,\nBusiness,\nAccounting",
    "Law": "Law",
    "Health Sciences": "Health\nSciences",
}
UNCERTAIN_STAGE_ORDER = ["Full run, taxonomy 1.1", "Full run, taxonomy 1.2", "UNCERTAIN subset, taxonomy 1.3",
                         "Dataset revision", "UNCERTAIN subset, taxonomy 1.4", "Full run, taxonomy 1.4",
                         "Post-annotation"]

# ACL page geometry: 7.7 cm columns inside a 16 cm text block.
COLUMN_WIDTH = 7.7 / 2.54
TEXT_WIDTH = 16.0 / 2.54

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
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
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seconds_between(start: str, end: str) -> float:
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


# -----------------------------------------------------------------------------
# Loading


def load_taxonomies() -> dict[str, Taxonomy]:
    """Taxonomy versions 1.0 to 1.4, with the 1.0 snapshot checked against its pinned hash."""
    if file_sha256(TAXONOMY_V1_0_PATH) != TAXONOMY_V1_0_SHA256:
        raise ValueError("The taxonomy 1.0 snapshot does not match its pinned SHA-256.")
    taxonomies = {version: load_taxonomy(path, DEFAULT_ALIASES) for version, path in TAXONOMY_PATHS.items()}
    for version, taxonomy in taxonomies.items():
        if taxonomy.version != version:
            raise ValueError(f"Taxonomy file for {version} declares version {taxonomy.version}.")
        if any(entry.get("context_rules") for entry in taxonomy.document["exams"].values()):
            raise ValueError(f"Taxonomy {version} uses context rules; candidate counts would be ambiguous.")
    return taxonomies


def load_run(run_dir: Path, expected_run_id: str, packaged: Taxonomy) -> dict:
    """Manifest, inference session, QC report and the taxonomy snapshot of one run."""
    manifest = read_json(run_dir / "manifest.json")
    session = read_json(run_dir / "inference-session.json")
    report = read_json(run_dir / "reports" / "report.json")
    taxonomy = load_taxonomy(run_dir / "taxonomy.json", run_dir / "exam_aliases.json")
    if manifest["run_id"] != expected_run_id:
        raise ValueError(f"{run_dir.name}: run {manifest['run_id']} is not the expected {expected_run_id}.")
    if taxonomy.checksum != manifest["identity"]["taxonomy_hash"] or taxonomy.checksum != packaged.checksum:
        raise ValueError(f"{run_dir.name}: taxonomy snapshot differs from the manifest or the packaged file.")
    if session["status"] != "stopped" or session["exit_codes"] != [0]:
        raise ValueError(f"{run_dir.name}: inference session did not stop cleanly.")
    return {"manifest": manifest, "session": session, "report": report, "taxonomy": taxonomy}


def load_export(run_dir: Path, run: dict) -> pd.DataFrame:
    """Published export of a run, checked against the report and the taxonomy."""
    latest = read_json(run_dir / "exports" / "latest.json")
    frame = pd.read_parquet(run_dir / "exports" / latest["directory"] / "data.parquet")
    taxonomy = run["taxonomy"]
    if len(frame) != latest["rows"] or len(frame) != run["manifest"]["selected_rows"]:
        raise ValueError(f"{run_dir.name}: export row count does not match the manifest.")
    if not frame["annotation_id"].is_unique or frame["annotation_run_id"].nunique() != 1:
        raise ValueError(f"{run_dir.name}: export has duplicate ids or mixed runs.")
    if frame["annotation_status"].value_counts().to_dict() != run["report"]["statuses"]:
        raise ValueError(f"{run_dir.name}: export statuses differ from the QC report.")
    labelled = frame[frame["subject"] != UNCERTAIN]
    derived = labelled["subject"].map(taxonomy.macro_areas)
    if not derived.eq(labelled["macro_area"]).all():
        raise ValueError(f"{run_dir.name}: stored macro areas differ from the taxonomy.")
    candidates = {exam: set(taxonomy.resolve_candidates(exam, {})) for exam in frame["exam"].unique()}
    allowed = [subject in candidates[exam] for exam, subject in zip(labelled["exam"], labelled["subject"], strict=True)]
    if not all(allowed):
        raise ValueError(f"{run_dir.name}: a label is outside the exam's candidates.")
    return frame


def open_checkpoint(run_dir: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{run_dir / 'checkpoint.sqlite3'}?mode=ro", uri=True)


def load_tasks(run_dir: Path) -> pd.DataFrame:
    """Final task results (pass 1, pass 2 and adjudication) per annotation id."""
    with open_checkpoint(run_dir) as connection:
        rows = connection.execute("SELECT annotation_id, kind, pass_number, status, result_json FROM tasks").fetchall()
    frame = pd.DataFrame(rows, columns=["annotation_id", "kind", "pass_number", "status", "result_json"])
    frame["subject"] = frame["result_json"].map(lambda text: json.loads(text)["subject"] if text else None)
    return frame.drop(columns="result_json")


def load_attempts(run_dir: Path) -> pd.DataFrame:
    """Every model request with its token usage and batch timing."""
    with open_checkpoint(run_dir) as connection:
        rows = connection.execute("SELECT annotation_id, kind, pass_number, request_status, error, model_metadata_json "
                                  "FROM attempts").fetchall()
    records = []
    for annotation_id, kind, pass_number, status, error, metadata_json in rows:
        metadata = json.loads(metadata_json)
        usage = metadata.get("usage") or {}
        records.append({"annotation_id": annotation_id, "kind": kind, "pass_number": pass_number, "status": status,
                        "error": error, "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"), "finish_reason": metadata.get("finish_reason"),
                        "batch_id": metadata.get("batch_id"), "batch_seconds": metadata.get("batch_duration_seconds")})
    return pd.DataFrame(records)


def load_final() -> tuple[pd.DataFrame, dict]:
    """The post-annotation dataset and its manifest, checked against the versioned configuration files."""
    frame = pd.read_json(POST_DIR / "data.jsonl", lines=True)
    manifest = read_json(POST_DIR / "manifest.json")
    manual_ids = {entry["annotation_id"] for entry in read_json(MANUAL_PATH)["assignments"]}
    removed_ids = {entry["annotation_id"] for entry in read_json(REMOVALS_PATH)["removals"]}
    if len(frame) != manifest["output_rows"] or frame["subject"].eq(UNCERTAIN).any():
        raise ValueError("Post-annotation dataset has the wrong size or still contains UNCERTAIN rows.")
    if set(frame.loc[frame["annotation_status"] == "manual", "annotation_id"]) != manual_ids:
        raise ValueError("Manual rows differ from config/manual_annotations.json.")
    if not {entry["annotation_id"] for entry in manifest["removed"]} <= removed_ids:
        raise ValueError("Post-annotation removals are not listed in config/removed_questions.json.")
    if frame["annotation_id"].isin(removed_ids).any():
        raise ValueError("A removed question is still in the final dataset.")
    return frame, manifest


def load_benchmark() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load the published splits and verify selection and annotation provenance."""
    annotated = load_from_disk(str(POST_DIR / "huggingface"))
    originals = {row["annotation_id"]: row for row in annotated}
    deduplicated = load_from_disk(str(DEDUPLICATED_DIR))
    deduplicated_ids = {annotation_id(row) for row in deduplicated}
    if len(deduplicated_ids) != len(deduplicated):
        raise ValueError("Duplicate deduplicated IDs.")
    for row in deduplicated:
        source = originals[annotation_id(row)]
        if any(source[key] != value for key, value in row.items()):
            raise ValueError("Deduplicated question differs from its annotated source.")
    rows = {"test": [], "dev": []}
    for level in LEVELS:
        for path in sorted((BENCHMARK_DIR / "datasets" / level).iterdir()):
            if not path.is_dir():
                continue
            dataset = load_from_disk(str(path))
            for split in rows:
                for row in dataset[split]:
                    identifier = row["annotation_id"]
                    if row["id"] != identifier or annotation_id(row) != identifier:
                        raise ValueError("Published ID differs from its content hash.")
                    if row["academic_level"] != level:
                        raise ValueError("Published row is in the wrong level.")
                    source = originals[identifier]
                    if any(row[key] != value for key, value in source.items()):
                        raise ValueError("Published row differs from its annotation source.")
                    rows[split].append(row)
    frames = {split: pd.DataFrame(records) for split, records in rows.items()}
    ids = {split: set(frame["id"]) for split, frame in frames.items()}
    if any(len(ids[split]) != len(frame) for split, frame in frames.items()):
        raise ValueError("Duplicate published IDs.")
    excluded = {entry["annotation_id"] for entry in read_json(EXCLUDED_TEST_PATH)["exclusions"]}
    if ids["test"] & ids["dev"] or ids["test"] != deduplicated_ids - ids["dev"] - excluded:
        raise ValueError("Test selection or global dev separation is inconsistent.")
    expected = {"test": {"high_school": 8704, "undergraduate": 29094},
                "dev": {"high_school": 30, "undergraduate": 45}}
    for split, frame in frames.items():
        if frame["academic_level"].value_counts().to_dict() != expected[split]:
            raise ValueError(f"Unexpected {split} counts.")
    return frames["test"], frames["dev"], read_json(BENCHMARK_DIR / "manifest.json")


def export_aggregates(output_dir: Path, test: pd.DataFrame, dev: pd.DataFrame,
                      taxonomy: Taxonomy, manifest: dict, runs: dict, exports: dict,
                      attempts: dict, joined: pd.DataFrame, chain: pd.DataFrame) -> None:
    """Export the populations used by the appendix separately."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "test_level_area": level_area_table(test, taxonomy),
        "test_subjects": subject_table(test, taxonomy),
        "test_exams": exam_table(test, taxonomy),
        "test_exam_area_pct": 100 * pd.crosstab(test["exam"], test["macro_area"], normalize="index"),
        "test_exam_subject_pct": 100 * pd.crosstab(test["exam"], test["subject"], normalize="index"),
        "test_statuses": test.groupby(["academic_level", "annotation_status"]).size().rename("questions"),
        "split_counts": pd.concat([frame["academic_level"].value_counts().rename(split)
                                   for split, frame in (("test", test), ("dev", dev))], axis=1),
        "annotation_runs": run_outcomes(runs, exports),
        "annotation_label_changes": change_shares_by_exam(joined),
        "annotation_uncertain_chain": chain,
        "annotation_cost_1.4": stage_cost(attempts["1.4"]),
        "annotation_prompt_tokens_1.4": prompt_token_stats(attempts["1.4"], exports["1.4"]),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=not isinstance(table.index, pd.RangeIndex))
    subjects = tables["test_subjects"]
    summary = {"test_rows": len(test), "dev_rows": len(dev),
               "test_subjects": int((subjects["total"] > 0).sum()),
               "test_subjects_below_100": int((subjects["total"] < 100).sum()),
               "test_subjects_below_50": int((subjects["total"] < 50).sum()),
               "test_smallest_area": int(test["macro_area"].value_counts().min()),
               "test_statuses": test["annotation_status"].value_counts().to_dict(),
               "annotation_source": {key: manifest["source"][key] for key in ("identifier", "revision", "rows")},
               "selection_source": manifest["source"]["test_source"],
               "published": {repo: value["revision"] for repo, value in manifest["published"].items()},
               "checks": {"unique_ids": True, "global_dev_separation": True,
                          "exact_deduplicated_test_selection": True, "annotation_content_equal": True},
               "files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(output_dir.glob("*.csv"))}}
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def parse_engine_log(run_dir: Path) -> dict:
    """Engine facts recorded by vLLM at start-up."""
    text = (run_dir / "vllm-rank-0000.log").read_text(encoding="utf-8", errors="replace")
    patterns = {"weights_seconds": r"Loading weights took ([\d.]+) seconds",
                "weights_gib": r"Model loading took ([\d.]+) GiB memory",
                "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
                "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
                "vllm_version": r"Initializing a V1 LLM engine \(v([\d.]+)\)"}
    return {key: (match.group(1) if (match := re.search(pattern, text)) else None) for key, pattern in patterns.items()}


# -----------------------------------------------------------------------------
# Taxonomy versions


def subject_definitions(taxonomy: Taxonomy) -> dict[str, str]:
    return {subject["name"]: subject["description"] for area in taxonomy.document["macro_areas"] for subject in area["subjects"]}


def taxonomy_summary(taxonomies: dict[str, Taxonomy]) -> pd.DataFrame:
    """Subjects, macro areas, exams and policy rules of every version."""
    rows = [{"version": version, "subjects": len(taxonomy.definitions),
             "macro_areas": len(taxonomy.document["macro_areas"]), "exams": len(taxonomy.document["exams"]),
             "policy_rules": len(taxonomy.document["annotation_policy"])}
            for version, taxonomy in taxonomies.items()]
    return pd.DataFrame(rows).set_index("version")


def candidate_matrix(taxonomies: dict[str, Taxonomy]) -> pd.DataFrame:
    """Candidates per exam key and version; NaN where the exam is not covered."""
    exams = list(taxonomies[VERSIONS[-1]].document["exams"])
    matrix = pd.DataFrame(index=exams, columns=VERSIONS, dtype=float)
    for version, taxonomy in taxonomies.items():
        for exam in exams:
            entry = taxonomy.document["exams"].get(exam)
            matrix.at[exam, version] = len(entry["allowed_subjects"]) if entry else np.nan
    return matrix


def version_diff(previous: Taxonomy, current: Taxonomy) -> dict:
    """What changed between two consecutive versions."""
    old_definitions, new_definitions = subject_definitions(previous), subject_definitions(current)
    old_policy, new_policy = previous.document["annotation_policy"], current.document["annotation_policy"]
    candidates_added = {}
    for exam, entry in current.document["exams"].items():
        before = set(previous.document["exams"].get(exam, {}).get("allowed_subjects", []))
        added = [subject for subject in entry["allowed_subjects"] if subject not in before]
        if exam in previous.document["exams"] and added:
            candidates_added[exam] = (len(before), len(entry["allowed_subjects"]), added)
    return {
        "added_subjects": sorted(set(new_definitions) - set(old_definitions)),
        "removed_subjects": sorted(set(old_definitions) - set(new_definitions)),
        "added_exams": {exam: len(current.document["exams"][exam]["allowed_subjects"])
                        for exam in current.document["exams"] if exam not in previous.document["exams"]},
        "candidates_added": candidates_added,
        "changed_definitions": sorted(s for s in new_definitions if s in old_definitions and old_definitions[s] != new_definitions[s]),
        "policy_added": sorted(set(new_policy) - set(old_policy)),
        "policy_changed": sorted(k for k in new_policy if k in old_policy and old_policy[k] != new_policy[k]),
        "area_changes": {s: (previous.macro_areas[s], current.macro_areas[s]) for s in old_definitions
                         if s in current.macro_areas and previous.macro_areas[s] != current.macro_areas[s]},
    }


# -----------------------------------------------------------------------------
# Runs and label changes


def adjudication_choices(export: pd.DataFrame) -> dict[str, int]:
    """Whether the adjudicator sided with pass 1, pass 2, a third subject or abstained."""
    adjudicated = export[export["annotation_agreement"].isin(["disagreement", "uncertain"])]
    choices = {"pass_1": 0, "pass_2": 0, "third": 0, "uncertain": 0}
    for subject, first, second in zip(adjudicated["subject"], adjudicated["annotation_pass_1_subject"],
                                      adjudicated["annotation_pass_2_subject"], strict=True):
        if subject == UNCERTAIN:
            choices["uncertain"] += 1
        elif subject == first:
            choices["pass_1"] += 1
        elif subject == second:
            choices["pass_2"] += 1
        else:
            choices["third"] += 1
    return choices


def run_outcomes(runs: dict[str, dict], exports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outcome of each full run: statuses, agreement, adjudication choices and session time."""
    rows = []
    for version, run in runs.items():
        report, session, manifest = run["report"], run["session"], run["manifest"]
        choices = adjudication_choices(exports[version])
        if sum(choices.values()) != report["adjudication_count"]:
            raise ValueError(f"Run {version}: adjudication choices do not add up to the report.")
        rows.append({"version": version, "run_id": manifest["run_id"], "source": manifest["identity"]["source"]["identifier"],
                     "source_revision": manifest["identity"]["source"]["revision"][:8], "rows": report["total_rows"],
                     **report["statuses"], "agreement_pct": 100 * report["agreement_rate"],
                     "adjudicated_pct": 100 * report["adjudication_rate"], **{f"adj_{k}": v for k, v in choices.items()},
                     "max_model_len": session["settings"]["max_model_len"],
                     "ready_seconds": seconds_between(session["started_at"], session["ready_at"]),
                     "session_seconds": seconds_between(session["started_at"], session["stopped_at"])})
    return pd.DataFrame(rows).set_index("version")


def joined_labels(exports: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The 1.4 rows with the labels every full run gave them."""
    base = exports["1.4"][["annotation_id", "exam", "academic_level", "subject", "macro_area", "annotation_agreement"]]
    joined = base.rename(columns={"subject": "subject_1.4", "macro_area": "macro_area_1.4", "annotation_agreement": "agreement_1.4"})
    for version in ("1.1", "1.2"):
        other = exports[version][["annotation_id", "exam", "academic_level", "subject", "macro_area", "annotation_agreement"]]
        other = other.rename(columns={"subject": f"subject_{version}", "macro_area": f"macro_area_{version}",
                                      "annotation_agreement": f"agreement_{version}", "exam": "exam_other",
                                      "academic_level": "level_other"})
        joined = joined.merge(other, on="annotation_id", how="inner")
        if len(joined) != len(base) or not joined["exam"].eq(joined["exam_other"]).all() \
                or not joined["academic_level"].eq(joined["level_other"]).all():
            raise ValueError(f"Join with run {version} is incomplete or inconsistent.")
        joined = joined.drop(columns=["exam_other", "level_other"])
    return joined


def change_shares_by_exam(joined: pd.DataFrame) -> pd.DataFrame:
    """Share of each exam's labels that changed between consecutive full runs, next to the within-run disagreement."""
    groups = [(exam, part) for exam, part in joined.groupby("exam")] + [("All", joined)]
    rows = []
    for exam, part in groups:
        rows.append({"exam": exam, "n": len(part),
                     "changed_1.1_to_1.2": int((part["subject_1.1"] != part["subject_1.2"]).sum()),
                     "changed_1.2_to_1.4": int((part["subject_1.2"] != part["subject_1.4"]).sum()),
                     "changed_1.1_to_1.2_pct": 100 * (part["subject_1.1"] != part["subject_1.2"]).mean(),
                     "changed_1.2_to_1.4_pct": 100 * (part["subject_1.2"] != part["subject_1.4"]).mean(),
                     "disagreement_1.2_pct": 100 * part["agreement_1.2"].eq("disagreement").mean(),
                     "disagreement_1.4_pct": 100 * part["agreement_1.4"].eq("disagreement").mean()})
    return pd.DataFrame(rows).set_index("exam")


def top_transitions(joined: pd.DataFrame, before: str, after: str, limit: int = TRANSITION_LIMIT) -> pd.DataFrame:
    """Most frequent (old subject, new subject) pairs between two runs."""
    changed = joined[joined[f"subject_{before}"] != joined[f"subject_{after}"]]
    counts = changed.groupby([f"subject_{before}", f"subject_{after}"]).size().sort_values(ascending=False)
    table = counts.head(limit).reset_index()
    table.columns = ["from", "to", "count"]
    return table


def new_label_uptake(exports: dict[str, pd.DataFrame], taxonomies: dict[str, Taxonomy]) -> tuple[pd.Series, pd.Series]:
    """Rows of run 1.2 on the eight new subjects, and rows per exam outside the 1.1 candidates of that exam."""
    new_subjects = set(taxonomies["1.2"].definitions) - set(taxonomies["1.1"].definitions)
    export = exports["1.2"]
    on_new = export.loc[export["subject"].isin(new_subjects), "subject"].value_counts()
    widened = {}
    for exam, part in export.groupby("exam"):
        old_candidates = set(taxonomies["1.1"].resolve_candidates(exam, {}))
        outside = part["subject"].ne(UNCERTAIN) & ~part["subject"].isin(old_candidates)
        if outside.any():
            widened[exam] = int(outside.sum())
    return on_new, pd.Series(widened).sort_values(ascending=False)


def macro_area_stability(joined: pd.DataFrame, taxonomies: dict[str, Taxonomy]) -> dict:
    """Stability of subjects and macro areas between consecutive full runs on the joined rows."""
    remapped_1_1 = joined["subject_1.1"].map(taxonomies["1.2"].macro_areas)
    both_labelled = joined["subject_1.1"].ne(UNCERTAIN) & joined["subject_1.2"].ne(UNCERTAIN)
    return {"same_subject_1.1_1.2_pct": 100 * joined["subject_1.1"].eq(joined["subject_1.2"]).mean(),
            "same_area_1.1_1.2_pct": 100 * remapped_1_1[both_labelled].eq(joined.loc[both_labelled, "macro_area_1.2"]).mean(),
            "same_subject_1.2_1.4_pct": 100 * joined["subject_1.2"].eq(joined["subject_1.4"]).mean(),
            "same_area_1.2_1.4_pct": 100 * joined["macro_area_1.2"].eq(joined["macro_area_1.4"]).mean()}


def uncertain_chain(exports: dict[str, pd.DataFrame], subsets: dict[str, pd.DataFrame],
                    final_manifest: dict) -> pd.DataFrame:
    """How the abstentions evolved from the first full run to the final dataset, computed from ids."""
    removed_ids = {entry["annotation_id"] for entry in read_json(REMOVALS_PATH)["removals"]}
    uncertain = {v: set(e.loc[e["subject"] == UNCERTAIN, "annotation_id"]) for v, e in exports.items()}
    subset_ids = {v: set(s["annotation_id"]) for v, s in subsets.items()}
    subset_uncertain = {v: set(s.loc[s["subject"] == UNCERTAIN, "annotation_id"]) for v, s in subsets.items()}
    if subset_ids["1.3"] != uncertain["1.2"]:
        raise ValueError("The 1.3 subset is not the UNCERTAIN set of run 1.2.")
    removed_from_subset = subset_uncertain["1.3"] & removed_ids
    if subset_ids["1.4"] != subset_ids["1.3"] - removed_from_subset:
        raise ValueError("The 1.4 subset is not the 1.3 subset minus the removed questions.")
    if uncertain["1.4"] & (uncertain["1.1"] | uncertain["1.2"]):
        raise ValueError("Run 1.4 abstained on a question that an earlier run had already abstained on.")
    post_removed = {entry["annotation_id"] for entry in final_manifest["removed"]}
    post_manual = {entry["annotation_id"] for entry in final_manifest["manual_assignments"]}
    if post_removed | post_manual != uncertain["1.4"]:
        raise ValueError("Post-annotation did not resolve exactly the 1.4 abstentions.")
    rows = [
        {"stage": UNCERTAIN_STAGE_ORDER[0], "items": len(exports["1.1"]), "uncertain": len(uncertain["1.1"]),
         "labelled": len(exports["1.1"]) - len(uncertain["1.1"]), "removed": 0},
        {"stage": UNCERTAIN_STAGE_ORDER[1], "items": len(exports["1.2"]), "uncertain": len(uncertain["1.2"]),
         "labelled": len(exports["1.2"]) - len(uncertain["1.2"]), "removed": 0,
         "previous_uncertain_labelled": len(uncertain["1.1"] - uncertain["1.2"]),
         "previous_uncertain_kept": len(uncertain["1.1"] & uncertain["1.2"]),
         "new_uncertain": len(uncertain["1.2"] - uncertain["1.1"])},
        {"stage": UNCERTAIN_STAGE_ORDER[2], "items": len(subset_ids["1.3"]), "uncertain": len(subset_uncertain["1.3"]),
         "labelled": len(subset_ids["1.3"]) - len(subset_uncertain["1.3"]), "removed": 0},
        {"stage": UNCERTAIN_STAGE_ORDER[3], "items": len(subset_uncertain["1.3"]), "uncertain": len(subset_uncertain["1.3"] - removed_ids),
         "labelled": 0, "removed": len(removed_from_subset)},
        {"stage": UNCERTAIN_STAGE_ORDER[4], "items": len(subset_ids["1.4"]), "uncertain": len(subset_uncertain["1.4"]),
         "labelled": len(subset_ids["1.4"]) - len(subset_uncertain["1.4"]), "removed": 0},
        {"stage": UNCERTAIN_STAGE_ORDER[5], "items": len(exports["1.4"]), "uncertain": len(uncertain["1.4"]),
         "labelled": len(exports["1.4"]) - len(uncertain["1.4"]), "removed": 0},
        {"stage": UNCERTAIN_STAGE_ORDER[6], "items": len(uncertain["1.4"]), "uncertain": 0,
         "labelled": len(post_manual), "removed": len(post_removed)},
    ]
    return pd.DataFrame(rows).set_index("stage")


# -----------------------------------------------------------------------------
# Final distribution


def level_area_table(final: pd.DataFrame, taxonomy: Taxonomy) -> pd.DataFrame:
    """Questions, share and distinct subjects per (level, macro area), in taxonomy order."""
    areas = [area["name"] for area in taxonomy.document["macro_areas"]]
    rows = []
    for level in LEVELS:
        part = final[final["academic_level"] == level]
        for area in areas:
            cell = part[part["macro_area"] == area]
            rows.append({"level": level, "macro_area": area, "subjects": cell["subject"].nunique(),
                         "questions": len(cell), "share": 100 * len(cell) / len(part)})
    overall = final["macro_area"].value_counts()
    if overall.min() < MIN_MACRO_AREA:
        raise ValueError(f"A macro area has fewer than {MIN_MACRO_AREA} questions.")
    return pd.DataFrame(rows)


def subject_table(final: pd.DataFrame, taxonomy: Taxonomy) -> pd.DataFrame:
    """Questions per subject and level, in taxonomy order."""
    counts = pd.crosstab(final["subject"], final["academic_level"])
    rows = []
    for area in taxonomy.document["macro_areas"]:
        for subject in area["subjects"]:
            name = subject["name"]
            row = {"macro_area": area["name"], "subject": name}
            for level in LEVELS:
                row[level] = int(counts.at[name, level]) if name in counts.index and level in counts.columns else 0
            row["total"] = row["high_school"] + row["undergraduate"]
            rows.append(row)
    return pd.DataFrame(rows)


def exam_table(final: pd.DataFrame, taxonomy: Taxonomy) -> pd.DataFrame:
    """Size, candidates, label diversity and the two most frequent subjects of each exam."""
    rows = []
    for exam, part in final.groupby("exam"):
        ranked = part["subject"].value_counts()
        rows.append({"level": part["academic_level"].iat[0], "exam": exam, "n": len(part),
                     "candidates": len(taxonomy.resolve_candidates(exam, {})), "macro_areas": part["macro_area"].nunique(),
                     "subjects": len(ranked), "top": ranked.index[0], "top_share": 100 * ranked.iat[0] / len(part),
                     "second": ranked.index[1], "second_share": 100 * ranked.iat[1] / len(part)})
    table = pd.DataFrame(rows)
    return table.sort_values(["level", "n"], ascending=[True, False], ignore_index=True)


# -----------------------------------------------------------------------------
# Cost


def stage_cost(attempts: pd.DataFrame) -> pd.DataFrame:
    """Requests, tokens and deduplicated batch time per stage of one run."""
    rows = []
    for stage in STAGES:
        part = attempts[(attempts["kind"] == stage[0]) & (attempts["pass_number"] == stage[1])]
        success = part[part["status"] == "success"]
        batches = success.drop_duplicates("batch_id")
        rows.append({"stage": STAGE_NAMES[stage], "requests": len(success), "failed": int((part["status"] != "success").sum()),
                     "prompt_tokens": int(success["prompt_tokens"].sum()),
                     "completion_tokens": int(success["completion_tokens"].sum()),
                     "completion_mean": success["completion_tokens"].mean(), "completion_max": int(success["completion_tokens"].max()),
                     "batches": len(batches), "batch_seconds": float(batches["batch_seconds"].sum())})
    table = pd.DataFrame(rows).set_index("stage")
    table.loc["Total"] = table.sum(numeric_only=True)
    table.loc["Total", "completion_mean"] = attempts.loc[attempts["status"] == "success", "completion_tokens"].mean()
    return table


def prompt_token_stats(attempts: pd.DataFrame, export: pd.DataFrame) -> pd.DataFrame:
    """Prompt tokens of the first pass per academic level."""
    first = attempts[(attempts["kind"] == "independent") & (attempts["pass_number"] == 1) & (attempts["status"] == "success")]
    merged = first.merge(export[["annotation_id", "academic_level"]], on="annotation_id")
    groups = {"all": merged, **{level: merged[merged["academic_level"] == level] for level in LEVELS}}
    rows = [{"group": name, "items": len(part), "mean": part["prompt_tokens"].mean(), "median": part["prompt_tokens"].median(),
             "min": int(part["prompt_tokens"].min()), "p99": part["prompt_tokens"].quantile(0.99),
             "max": int(part["prompt_tokens"].max())} for name, part in groups.items()]
    return pd.DataFrame(rows).set_index("group")


# -----------------------------------------------------------------------------
# Reporting


def print_frame(frame: pd.DataFrame, index: bool = True) -> None:
    with pd.option_context("display.width", 220, "display.max_columns", 40, "display.max_rows", 200,
                           "display.max_colwidth", 70, "display.float_format", "{:.2f}".format):
        print(frame.to_string(index=index))


def print_taxonomy_section(taxonomies: dict[str, Taxonomy]) -> None:
    print("== Taxonomy versions")
    print_frame(taxonomy_summary(taxonomies))
    print("Candidates per exam and version:")
    print_frame(candidate_matrix(taxonomies))
    for previous, current in zip(VERSIONS, VERSIONS[1:], strict=False):
        diff = version_diff(taxonomies[previous], taxonomies[current])
        print(f"-- {previous} -> {current}")
        print("  added subjects:", diff["added_subjects"], "| removed:", diff["removed_subjects"])
        print("  added exams:", diff["added_exams"])
        for exam, (before, after, added) in diff["candidates_added"].items():
            print(f"  candidates {exam}: {before} -> {after}, added {added}")
        print("  changed definitions:", len(diff["changed_definitions"]), diff["changed_definitions"])
        print("  policy added:", diff["policy_added"], "| policy changed:", diff["policy_changed"])
        areas = sorted({pair for pair in diff["area_changes"].values()})
        print("  macro-area moves:", len(diff["area_changes"]), "subjects;", areas)


def print_runs_section(runs: dict[str, dict], exports: dict[str, pd.DataFrame], joined: pd.DataFrame,
                       taxonomies: dict[str, Taxonomy]) -> None:
    print("\n== Runs per version")
    print_frame(run_outcomes(runs, exports).T)
    print("\n== Label changes between versions")
    print("Joined rows:", len(joined))
    print_frame(change_shares_by_exam(joined))
    print("Stability:", {k: round(v, 2) for k, v in macro_area_stability(joined, taxonomies).items()})
    for before, after in (("1.1", "1.2"), ("1.2", "1.4")):
        print(f"Top transitions {before} -> {after}:")
        print_frame(top_transitions(joined, before, after), index=False)
    on_new, widened = new_label_uptake(exports, taxonomies)
    print("Run 1.2 rows on the eight new subjects:", int(on_new.sum()), on_new.to_dict())
    print("Run 1.2 rows outside the 1.1 candidates of the exam:", int(widened.sum()), widened.to_dict())
    abstained = exports["1.2"][exports["1.2"]["subject"] == UNCERTAIN]
    print("Run 1.2 abstentions with confidence >= 0.90:", int((abstained["subject_confidence"] >= 0.9).sum()), "of", len(abstained))


def print_uncertain_section(chain: pd.DataFrame) -> None:
    print("\n== UNCERTAIN chain")
    print_frame(chain)


def print_distribution_section(final: pd.DataFrame, taxonomy: Taxonomy) -> None:
    print("\n== Final distribution")
    print("Rows:", len(final), "| statuses:", final["annotation_status"].value_counts().to_dict(),
          "| by level:", final["academic_level"].value_counts().to_dict(), "| exams:", final["exam"].nunique())
    areas = level_area_table(final, taxonomy)
    print_frame(areas, index=False)
    overall = final["macro_area"].value_counts()
    print_frame(pd.DataFrame({"questions": overall, "share": 100 * overall / len(final)}))
    subjects = subject_table(final, taxonomy)
    print_frame(subjects, index=False)
    print("Subjects used:", int((subjects["total"] > 0).sum()), "of", len(subjects),
          "| fewer than 50:", int((subjects["total"] < 50).sum()), "| fewer than 100:", int((subjects["total"] < 100).sum()))
    print("Smallest subjects:", subjects.nsmallest(5, "total")[["subject", "total"]].to_dict("records"))
    print("Top subjects:", final["subject"].value_counts().head(8).to_dict())
    print_frame(exam_table(final, taxonomy), index=False)
    shares = 100 * pd.crosstab(final["exam"], final["macro_area"], normalize="index")
    print_frame(shares.round(1))


def print_post_annotation_section(final_manifest: dict) -> None:
    print("\n== Post-annotation")
    categories = pd.Series([entry["category"] for entry in final_manifest["manual_assignments"]]).value_counts()
    print("Manual assignments:", len(final_manifest["manual_assignments"]), categories.to_dict())
    print("Removed:", len(final_manifest["removed"]), [entry["category"] for entry in final_manifest["removed"]])
    print("Revision list:", len(read_json(REMOVALS_PATH)["removals"]), "ids |",
          pd.Series([e["category"] for e in read_json(REMOVALS_PATH)["removals"]]).value_counts().to_dict())
    print("Input rows:", final_manifest["input_rows"], "| output rows:", final_manifest["output_rows"],
          "| hub:", final_manifest["hub"]["repository"], final_manifest["hub"]["revision"][:8])


def print_cost_section(runs: dict[str, dict], attempts: dict[str, pd.DataFrame], exports: dict[str, pd.DataFrame]) -> None:
    print("\n== Cost")
    for version, run in runs.items():
        session = run["session"]
        print(f"-- Run {version}: device {session['devices'][0]['name']} ({session['devices'][0]['total_bytes'] / 1e9:.1f} GB), "
              f"settings {session['settings']}")
        print("   engine:", parse_engine_log(FULL_RUNS[version]))
        print(f"   session {seconds_between(session['started_at'], session['stopped_at']):.0f} s, "
              f"ready after {seconds_between(session['started_at'], session['ready_at']):.0f} s")
        finish = attempts[version].loc[attempts[version]["status"] == "success", "finish_reason"].value_counts().to_dict()
        errors = attempts[version].loc[attempts[version]["status"] != "success", "error"].value_counts().to_dict()
        print("   finish reasons:", finish, "| failed attempts:", errors)
        print_frame(stage_cost(attempts[version]))
    print("Prompt tokens of pass 1 (run 1.4) by level:")
    print_frame(prompt_token_stats(attempts["1.4"], exports["1.4"]))
    first = attempts["1.4"][(attempts["1.4"]["kind"] == "independent") & (attempts["1.4"]["pass_number"] == 1)]
    longest = first.merge(exports["1.4"][["annotation_id", "exam"]], on="annotation_id").nlargest(1, "prompt_tokens")
    print("Longest pass-1 prompt:", longest[["exam", "prompt_tokens"]].to_dict("records"))


# -----------------------------------------------------------------------------
# Figures


def exam_rows(exams: pd.DataFrame) -> list[tuple[str, str | None]]:
    """Exams grouped by level and sorted by size, with one header row per level."""
    rows = []
    for level, label in LEVEL_LABELS.items():
        rows.append((label, None))
        for exam in exams.loc[exams["level"] == level, "exam"]:
            rows.append((EXAM_LABELS.get(exam, exam), exam))
    return rows


def style_row_axis(ax: plt.Axes, rows: list[tuple[str, str | None]]) -> None:
    """Label grouped rows and set the header rows in bold."""
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    for tick_label, (_, key) in zip(ax.get_yticklabels(), rows, strict=True):
        if key is None:
            tick_label.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)


def plot_candidates_by_version(matrix: pd.DataFrame, exams: pd.DataFrame, taxonomy: Taxonomy) -> Figure:
    """Figure: candidate subjects per exam across taxonomy versions."""
    rows = exam_rows(exams)
    keys = [taxonomy.exam_key(exam) if exam else None for _, exam in rows]
    values = np.array([[matrix.at[key, version] if key else np.nan for version in VERSIONS] for key in keys], dtype=float)

    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.9), layout="constrained")
    colormap = SEQUENTIAL.copy()
    colormap.set_bad("none")
    ax.imshow(np.ma.masked_invalid(values), cmap=colormap, norm=Normalize(vmin=0, vmax=np.nanmax(values) * 1.15), aspect="auto")
    for (row, col), value in np.ndenumerate(values):
        if keys[row] is None:
            continue
        text = "–" if np.isnan(value) else f"{int(value)}"
        colour = "white" if not np.isnan(value) and value >= 45 else INK
        ax.text(col, row, text, ha="center", va="center", fontsize=6, color=colour)
    ax.set_xticks(range(len(VERSIONS)), [f"v{version}" for version in VERSIONS])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    style_row_axis(ax, rows)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Candidate subjects per exam and taxonomy version")
    return figure


def plot_label_changes(shares: pd.DataFrame, exams: pd.DataFrame) -> Figure:
    """Figure: share of labels changed between consecutive full runs, with the within-run disagreement as reference."""
    rows = exam_rows(exams)
    positions = [row for row, (_, exam) in enumerate(rows) if exam is not None]
    part = shares.loc[[exam for _, exam in rows if exam is not None]]

    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.9), layout="constrained")
    ax.plot(part["disagreement_1.4_pct"], positions, "o", markersize=3.2, markerfacecolor="white", markeredgecolor=MUTED,
            markeredgewidth=0.7, zorder=2, label="Pass 1 vs pass 2 disagreement, run 1.4")
    ax.plot(part["changed_1.1_to_1.2_pct"], positions, "o", markersize=3, color=BLUE, zorder=3, label="Changed from 1.1 to 1.2")
    ax.plot(part["changed_1.2_to_1.4_pct"], positions, "s", markersize=2.8, color=ORANGE, zorder=3, label="Changed from 1.2 to 1.4")
    style_row_axis(ax, rows)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, max(part[["changed_1.1_to_1.2_pct", "changed_1.2_to_1.4_pct", "disagreement_1.4_pct"]].max()) * 1.12)
    ax.set_xlabel("Share of the exam's questions (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    figure.legend(loc="outside upper center", ncol=1)
    return figure


def plot_uncertain_chain(chain: pd.DataFrame) -> Figure:
    """Figure: what became of the abstentions at every stage after the first full run."""
    stages = chain.index[1:]
    parts = []
    for stage in stages:
        row = chain.loc[stage]
        if stage == UNCERTAIN_STAGE_ORDER[1]:
            parts.append((row["previous_uncertain_labelled"], row["previous_uncertain_kept"], 0))
        elif stage == UNCERTAIN_STAGE_ORDER[5]:
            parts.append((0, row["uncertain"], 0))
        else:
            parts.append((row["labelled"], row["uncertain"], row["removed"]))
    labels = [stage.replace(", taxonomy", ",\ntaxonomy") for stage in stages]
    series = (("Received a subject", BLUE), ("Still UNCERTAIN", ORANGE), ("Removed", MUTED))

    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.2), layout="constrained")
    positions = np.arange(len(stages))
    left = np.zeros(len(stages))
    for index, (name, colour) in enumerate(series):
        values = np.array([part[index] for part in parts], dtype=float)
        ax.barh(positions, values, left=left, height=0.62, color=colour, edgecolor="white", linewidth=0.4, label=name, zorder=2)
        for position, value, start in zip(positions, values, left, strict=True):
            if value >= 6:
                ax.text(start + value / 2, position, f"{int(value)}", ha="center", va="center", fontsize=5.5, color="white")
        left += values
    for position, part in zip(positions, parts, strict=True):
        small = [f"{int(v)} {name.lower()}" for v, (name, _) in zip(part, series, strict=True) if 0 < v < 6]
        if small:
            ax.text(sum(part) + 1.5, position, ", ".join(small), ha="left", va="center", fontsize=5.5, color=MUTED)
    ax.set_yticks(positions, labels)
    ax.tick_params(axis="y", length=0, labelsize=5.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xlabel("Questions")
    ax.grid(axis="x", color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    figure.legend(loc="outside upper center", ncol=3)
    return figure


def plot_macro_area_by_exam(final: pd.DataFrame, exams: pd.DataFrame, taxonomy: Taxonomy) -> Figure:
    """Figure: share of each exam's questions in every macro area."""
    areas = [area["name"] for area in taxonomy.document["macro_areas"]]
    rows = exam_rows(exams)
    shares = 100 * pd.crosstab(final["exam"], final["macro_area"], normalize="index")
    shares = shares.reindex(columns=areas, fill_value=0)
    values = shares.reindex([exam for _, exam in rows]).to_numpy(dtype=float)
    values[values == 0] = np.nan

    figure, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.4), layout="constrained")
    colormap = SEQUENTIAL.copy()
    colormap.set_bad("none")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=colormap, norm=Normalize(vmin=0, vmax=100), aspect="auto")
    for (row, col), value in np.ndenumerate(values):
        if not np.isnan(value):
            ax.text(col, row, f"{value:.1f}" if value >= 1 else "<1", ha="center", va="center",
                    fontsize=5, color="white" if value >= 45 else INK)
    ax.set_xticks(range(len(areas)), [MACRO_AREA_LABELS[area] for area in areas])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelsize=5.5)
    style_row_axis(ax, rows)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=ax, shrink=0.6, aspect=25, pad=0.02, ticks=range(0, 101, 25))
    colorbar.set_label("Share of the exam's questions (%)")
    colorbar.outline.set_visible(False)
    return figure


def subject_rows(table: pd.DataFrame, areas: list[str]) -> list[tuple[str, str | None]]:
    rows = []
    for area in areas:
        rows.append((area, None))
        rows.extend((subject, subject) for subject in table.loc[table["macro_area"] == area, "subject"])
    return rows


def plot_subjects_by_level(table: pd.DataFrame, taxonomy: Taxonomy) -> Figure:
    """Figure: questions per subject, stacked by level, in two panels of macro areas."""
    areas = [area["name"] for area in taxonomy.document["macro_areas"]]
    panels = [[a for a in areas if a in ("Law", "Health Sciences", "Economics, Business, and Accounting",
                                         "Social and Applied Social Sciences")],
              [a for a in areas if a not in ("Law", "Health Sciences", "Economics, Business, and Accounting",
                                             "Social and Applied Social Sciences")]]
    counts = table.set_index("subject")
    limit = counts["total"].max() * 1.2

    figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 4.6), layout="constrained")
    for ax, panel in zip(axes, panels, strict=True):
        rows = subject_rows(table, panel)
        positions = [row for row, (_, subject) in enumerate(rows) if subject is not None]
        subjects = [subject for _, subject in rows if subject is not None]
        high = counts.loc[subjects, "high_school"].to_numpy()
        under = counts.loc[subjects, "undergraduate"].to_numpy()
        ax.barh(positions, high, height=0.68, color=BLUE, edgecolor="white", linewidth=0.3, zorder=2, label="High school")
        ax.barh(positions, under, left=high, height=0.68, color=ORANGE, edgecolor="white", linewidth=0.3, zorder=2,
                label="Undergraduate")
        for position, total in zip(positions, high + under, strict=True):
            ax.text(total + 0.01 * limit, position, f"{int(total):,}", va="center", ha="left", fontsize=5, color=MUTED)
        style_row_axis(ax, rows)
        ax.tick_params(axis="y", labelsize=5.5)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_xlim(0, limit)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="x", color=GRID, linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Questions per subject")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    return figure


def save_figure(figure: Figure, name: str) -> None:
    figure.savefig(FIGURE_DIR / name, metadata={"CreationDate": None}, dpi=600)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ABLATION_DIR,
                        help="Directory for figures/ and aggregates/.")
    args = parser.parse_args()
    global FIGURE_DIR
    FIGURE_DIR = args.output_dir / "figures"
    inputs = [*FULL_RUNS.values(), *SUBSET_RUNS.values(), POST_DIR, TAXONOMY_V1_0_PATH, MANUAL_PATH, REMOVALS_PATH]
    missing = [path for path in inputs if not path.exists()]
    if missing:
        print(f"Missing inputs: {missing}")
        return 1

    taxonomies = load_taxonomies()
    runs = {version: load_run(path, EXPECTED_RUN_IDS[version], taxonomies[version]) for version, path in FULL_RUNS.items()}
    exports = {version: load_export(path, runs[version]) for version, path in FULL_RUNS.items()}
    subset_runs = {version: load_run(path, EXPECTED_RUN_IDS[f"{version}-subset"], taxonomies[version])
                   for version, path in SUBSET_RUNS.items()}
    subsets = {version: load_export(path, subset_runs[version]) for version, path in SUBSET_RUNS.items()}
    attempts = {version: load_attempts(path) for version, path in FULL_RUNS.items()}
    annotated, final_manifest = load_final()
    final, dev, benchmark_manifest = load_benchmark()
    joined = joined_labels(exports)
    chain = uncertain_chain(exports, subsets, final_manifest)
    taxonomy = taxonomies["1.4"]

    print_taxonomy_section(taxonomies)
    print_runs_section(runs, exports, joined, taxonomies)
    print_uncertain_section(chain)
    print_distribution_section(final, taxonomy)
    print("Dev counts:", dev["academic_level"].value_counts().to_dict())
    print_post_annotation_section(final_manifest)
    print_cost_section(runs, attempts, exports)

    plt.rcParams.update(STYLE)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    annotation_exams = exam_table(annotated, taxonomy)
    exams = exam_table(final, taxonomy)
    save_figure(plot_candidates_by_version(candidate_matrix(taxonomies), annotation_exams, taxonomy), "taxonomy_versions.pdf")
    save_figure(plot_label_changes(change_shares_by_exam(joined), annotation_exams), "label_changes_by_version.pdf")
    save_figure(plot_uncertain_chain(chain), "uncertain_flow.pdf")
    save_figure(plot_macro_area_by_exam(final, exams, taxonomy), "macro_area_by_exam.pdf")
    save_figure(plot_subjects_by_level(subject_table(final, taxonomy), taxonomy), "subjects_by_level.pdf")
    export_aggregates(args.output_dir / "aggregates", final, dev, taxonomy, benchmark_manifest,
                      runs, exports, attempts, joined, chain)
    print(f"\nFigures written to {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
