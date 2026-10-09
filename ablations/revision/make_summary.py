"""Print and verify every number quoted in the dataset-revision appendix.

Reads config/removed_questions.json, the original and revised pipeline stages, the knowledge-area
annotation runs that detected the defects and the audit of run 1.1. Nothing is written; any
inconsistency between the artifacts raises ValueError. Run with .venv/bin/python from the repo root.
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from datasets import load_from_disk

from mmlu_pt.annotation.knowledge_area.dataset import annotation_id

REPO_ROOT = Path(__file__).resolve().parents[2]
REMOVALS_PATH = REPO_ROOT / "config" / "removed_questions.json"
OUTPUT_DIR = REPO_ROOT / "output"
STAGE_04 = OUTPUT_DIR / "04 - filtered-huggingface"
STAGE_06 = OUTPUT_DIR / "06 - semantic-deduplicated" / "huggingface"
REVISED_04 = OUTPUT_DIR / "04 - revised"
REVISED_06 = OUTPUT_DIR / "06 - revised"

WORK_DIR = OUTPUT_DIR / "knowledge-annotation-work"
SUBSET_INPUT = WORK_DIR / "subsets" / "full-run-2-uncertain" / "dataset"
SUBSET_REVISED = WORK_DIR / "subsets" / "full-run-2-uncertain-revised"
FULL_RUNS = {"1.2": WORK_DIR / "full-run-2", "1.4": WORK_DIR / "full-run-taxonomy-v1.4"}
SUBSET_RUNS = {"1.3": WORK_DIR / "uncertain-taxonomy-v1.3", "1.4": WORK_DIR / "uncertain-taxonomy-v1.4"}
POST_DIR = FULL_RUNS["1.4"] / "post-annotation"
AUDIT_PATH = WORK_DIR / "taxonomy-v1.2-audit" / "reviewed_cases.json"
EXPECTED_RUN_IDS = {"1.2": "4e053d1ce260f6ae", "1.4": "1640098255bb0bed",
                    "1.3-subset": "9f4f7174e7631904", "1.4-subset": "130ddfa5eae044b3"}

UNCERTAIN = "UNCERTAIN"
IDENTITY_FIELDS = ["exam", "exam_edition", "num"]
# The evidence field of each listed question starts with one of these prefixes.
SOURCES = {"full-run-2 UNCERTAIN": "abstention_1.2",
           "varredura por padrão de defeito": "pattern_sweep",
           "full-run-taxonomy-v1.4 UNCERTAIN": "abstention_1.4"}
SOURCE_ORDER = ["abstention_1.2", "pattern_sweep", "abstention_1.4"]
# Strings that reproduce the pattern sweep over the question text.
SWEEP_PATTERNS = {"stem_replaced_by_latex_instruction": "Seu único objetivo é detectar as expressões matemáticas",
                  "self_referential_block": "A resposta da questão"}
INSTRUCTION_PREFIXES = ("INSTRUÇÕES", "Instruções:")
NEXT_PASSAGE_PATTERN = re.compile(r"texto para as? (?:quest|pr[óo]xim)", re.IGNORECASE)
KEPT_EDGE_CASES = [("AFA", "AFA_2019", "45"), ("COMVEST", "COMVEST 2015", "38"), ("POSCOMP", "POSCOMP_2024", "66")]
IME_PAIR = {("IME", "IME 2010 - Portugues", "36"): "2010", ("IME", "IME 2011 - Portugues", "36"): "2011"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def print_frame(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(frame.to_string())


def item_label(row: dict) -> str:
    return f"{row['exam']} | {row['exam_edition']} | #{row['num']}"


# -----------------------------------------------------------------------------
# Removal list


def source_of(evidence: str) -> str:
    for prefix, source in SOURCES.items():
        if evidence.startswith(prefix):
            return source
    raise ValueError(f"Evidence does not name a known detection source: {evidence!r}")


def load_removals() -> pd.DataFrame:
    """The versioned list as a frame indexed by annotation_id, checked for internal consistency."""
    document = read_json(REMOVALS_PATH)
    frame = pd.DataFrame(document["removals"]).set_index("annotation_id")
    if not frame.index.is_unique:
        raise ValueError("The removal list repeats an annotation_id.")
    unknown = set(frame["category"]) - set(document["categories"])
    if unknown:
        raise ValueError(f"Categories without a description: {sorted(unknown)}")
    if (frame["reason"] != frame["category"].map(document["categories"])).any():
        raise ValueError("A removal reason differs from the description of its category.")
    frame["source"] = frame["evidence"].map(source_of)
    frame.attrs["created_at"] = document["created_at"]
    frame.attrs["updated_at"] = document["updated_at"]
    return frame


# -----------------------------------------------------------------------------
# Pipeline stages and revised outputs


def load_stage(path: Path) -> pd.DataFrame:
    """Identity fields, text and annotation_id of every row of a dataset saved with save_to_disk."""
    data = load_from_disk(str(path))
    frame = data.to_pandas()[IDENTITY_FIELDS + ["question", "choices", "answer"]]
    frame["annotation_id"] = [annotation_id(row) for row in data]
    if not frame["annotation_id"].is_unique:
        raise ValueError(f"{path}: annotation ids are not unique.")
    return frame


def check_listed_rows(stage: pd.DataFrame, removals: pd.DataFrame) -> None:
    rows = stage.set_index("annotation_id")
    missing = removals.index.difference(rows.index)
    if len(missing):
        raise ValueError(f"Listed ids absent from stage 04: {list(missing)}")
    if not rows.loc[removals.index, IDENTITY_FIELDS].equals(removals[IDENTITY_FIELDS]):
        raise ValueError("Exam, edition or item number of a listed question differs from stage 04.")


def check_revision(directory: Path, source: pd.DataFrame, removals: pd.DataFrame, check_list_hash: bool) -> dict:
    """Manifest and removals.jsonl of a revised output, checked against its input and the list."""
    manifest = read_json(directory / "manifest.json")
    removed = read_jsonl(directory / "removals.jsonl")
    removed_ids = {entry["annotation_id"] for entry in removed}
    absent_ids = set(manifest["listed_ids_absent_from_input"])
    listed_ids = set(removals.index)
    if manifest["input_rows"] != len(source) or manifest["removed_rows"] != len(removed) != len(removed_ids):
        raise ValueError(f"{directory.name}: row counts disagree with the input or removals.jsonl.")
    if manifest["output_rows"] != manifest["input_rows"] - manifest["removed_rows"]:
        raise ValueError(f"{directory.name}: output_rows is not input_rows minus removed_rows.")
    # An output built from an earlier version of the list covers only the ids listed at the time.
    if removed_ids & absent_ids or not removed_ids | absent_ids <= listed_ids:
        raise ValueError(f"{directory.name}: removed and absent ids are not disjoint parts of the list.")
    if check_list_hash and removed_ids | absent_ids != listed_ids:
        raise ValueError(f"{directory.name}: removed and absent ids do not partition the list.")
    if absent_ids & set(source["annotation_id"]):
        raise ValueError(f"{directory.name}: an id reported absent is present in the input.")
    for entry in removed:
        if source["annotation_id"].iat[entry["input_row_index"]] != entry["annotation_id"]:
            raise ValueError(f"{directory.name}: input_row_index {entry['input_row_index']} points at another row.")
    if check_list_hash and manifest["removals_sha256"] != file_sha256(REMOVALS_PATH):
        raise ValueError(f"{directory.name}: built from a different removal list than the versioned one.")
    output = load_stage(directory / "huggingface")
    expected = source.loc[~source["annotation_id"].isin(removed_ids), "annotation_id"]
    if len(output) != manifest["output_rows"] or output["annotation_id"].tolist() != expected.tolist():
        raise ValueError(f"{directory.name}: output rows are not the kept input rows in input order.")
    return manifest


def normalized_statement(question: str) -> str:
    """The exact-deduplication key of the pipeline: NFKC, casefold and collapsed whitespace."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", question)).casefold().strip()


def absent_twins(stage: pd.DataFrame, removals: pd.DataFrame, revision: dict) -> pd.DataFrame:
    """For each listed id absent from a stage, the listed row with the same statement that the stage kept."""
    listed = stage[stage["annotation_id"].isin(removals.index)].set_index("annotation_id")
    keys = listed["question"].map(normalized_statement)
    rows = []
    for identity in revision["listed_ids_absent_from_input"]:
        twins = [other for other in keys.index if other != identity and keys[other] == keys[identity]]
        kept_twins = [other for other in twins if other not in revision["listed_ids_absent_from_input"]]
        if len(kept_twins) != 1:
            raise ValueError(f"Absent id {identity[:8]} is not an exact-deduplication twin of one kept listed row.")
        rows.append({"absent": item_label(removals.loc[identity]), "twin_in_stage": item_label(removals.loc[kept_twins[0]])})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Annotation runs that detected the defects


def load_export(run_dir: Path, expected_run_id: str) -> tuple[pd.DataFrame, dict]:
    manifest = read_json(run_dir / "manifest.json")
    latest = read_json(run_dir / "exports" / "latest.json")
    frame = pd.read_parquet(run_dir / "exports" / latest["directory"] / "data.parquet")
    if manifest["run_id"] != expected_run_id:
        raise ValueError(f"{run_dir.name}: run {manifest['run_id']} is not the expected {expected_run_id}.")
    if len(frame) != latest["rows"] or len(frame) != manifest["selected_rows"]:
        raise ValueError(f"{run_dir.name}: export row count does not match the manifest.")
    if not frame["annotation_id"].is_unique:
        raise ValueError(f"{run_dir.name}: annotation ids are not unique.")
    return frame, manifest


def uncertain_ids(frame: pd.DataFrame) -> set[str]:
    return set(frame.loc[frame["subject"] == UNCERTAIN, "annotation_id"])


def ids_by_source(removals: pd.DataFrame, source: str) -> set[str]:
    return set(removals.index[removals["source"] == source])


def ime_table(removals: pd.DataFrame, stage: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Answer key of each copy of the IME item and the subject given to it by every run that saw it."""
    pair = removals.reset_index().merge(pd.DataFrame([{**dict(zip(IDENTITY_FIELDS, key)), "copy": year}
                                                      for key, year in IME_PAIR.items()]), on=IDENTITY_FIELDS)
    if len(pair) != 2:
        raise ValueError("The IME pair is not in the removal list.")
    rows = stage.set_index("annotation_id").loc[pair["annotation_id"]]
    if rows["question"].nunique() != 1 or rows["choices"].map(tuple).nunique() != 1:
        raise ValueError("The IME copies do not share the same question and choices.")
    table = pd.DataFrame({"answer": rows["answer"].tolist()}, index=pair["copy"].tolist())
    for name, frame in frames.items():
        subjects = frame.set_index("annotation_id")["subject"]
        table[name] = [subjects.get(identity, "--") for identity in pair["annotation_id"]]
    return table


def check_detection(removals: pd.DataFrame, full: dict[str, pd.DataFrame], subsets: dict[str, pd.DataFrame],
                    post: dict) -> dict:
    """Follow the abstentions from run 1.2 to the post-annotation and check them against the list."""
    listed = set(removals.index)
    uncertain = {name: uncertain_ids(frame) for name, frame in {**full, **{f"{k}-subset": v for k, v in subsets.items()}}.items()}
    abstention_12, sweep, abstention_14 = (ids_by_source(removals, source) for source in SOURCE_ORDER)
    if uncertain["1.2"] & listed != abstention_12:
        raise ValueError("The listed abstentions of run 1.2 are not the abstentions of its export.")
    if sweep & uncertain["1.2"] or not sweep <= set(full["1.2"]["annotation_id"]):
        raise ValueError("A question found by the sweep was not labelled by run 1.2.")
    if set(subsets["1.3"]["annotation_id"]) != uncertain["1.2"]:
        raise ValueError("The 1.3 subset is not the abstention set of run 1.2.")
    removed_from_subset = uncertain["1.3-subset"] & listed
    kept_from_subset = uncertain["1.3-subset"] - listed
    if set(subsets["1.4"]["annotation_id"]) != uncertain["1.2"] - removed_from_subset:
        raise ValueError("The 1.4 subset is not the 1.3 subset minus the removed questions.")
    ime_ids = set(removals.index[removals["category"] == "missing_supporting_text"])
    if removed_from_subset != abstention_12 - ime_ids or not uncertain["1.4-subset"] < ime_ids:
        raise ValueError("The IME pair did not stay in the subset until run 1.4 split it.")
    if uncertain["1.4"] & uncertain["1.2"]:
        raise ValueError("Run 1.4 abstained on a question run 1.2 had abstained on.")
    post_removed = {entry["annotation_id"] for entry in post["removed"]}
    post_manual = {entry["annotation_id"] for entry in post["manual_assignments"]}
    if post_removed != abstention_14 or set(full["1.4"]["annotation_id"]) & listed != abstention_14:
        raise ValueError("The removals of the post-annotation are not the listed abstentions of run 1.4.")
    if post_removed | post_manual != uncertain["1.4"] or post_removed & post_manual:
        raise ValueError("Post-annotation did not resolve exactly the abstentions of run 1.4.")
    if post["input_rows"] != len(full["1.4"]) or post["output_rows"] != post["input_rows"] - len(post_removed):
        raise ValueError("Post-annotation row counts disagree with the export of run 1.4.")
    if post["removals_sha256"] != file_sha256(REMOVALS_PATH):
        raise ValueError("Post-annotation was built from a different removal list than the versioned one.")
    return {"uncertain": uncertain, "kept_from_subset": kept_from_subset, "post_manual": post_manual,
            "post_removed": post_removed, "removed_from_subset": removed_from_subset}


def funnel_table(removals: pd.DataFrame, full: dict[str, pd.DataFrame], subsets: dict[str, pd.DataFrame],
                 detection: dict) -> pd.DataFrame:
    """The steps that built the list, with the questions examined and added at each one."""
    uncertain = detection["uncertain"]
    sweep = ids_by_source(removals, "pattern_sweep")
    ime = set(removals.index[removals["category"] == "missing_supporting_text"])
    rows = [("Full run 1.2", len(full["1.2"]), len(uncertain["1.2"]), 0),
            ("Run 1.3 on the abstentions", len(subsets["1.3"]), len(uncertain["1.3-subset"]), 0),
            ("Reading of the remaining abstentions", len(uncertain["1.3-subset"]), len(detection["kept_from_subset"]),
             len(detection["removed_from_subset"])),
            ("Pattern sweep of stages 04 and 06", len(full["1.2"]), 0, len(sweep)),
            ("Run 1.4 on the remaining abstentions", len(subsets["1.4"]), len(uncertain["1.4-subset"]), 0),
            ("Reading of the identical copies", len(ime), 0, len(ime)),
            ("Full run 1.4", len(full["1.4"]), len(uncertain["1.4"]), 0),
            ("Post-annotation", len(uncertain["1.4"]), 0, len(detection["post_removed"]))]
    frame = pd.DataFrame(rows, columns=["step", "examined", "uncertain", "added"]).set_index("step")
    frame["list_size"] = frame["added"].cumsum()
    if frame["list_size"].iat[-1] != len(removals):
        raise ValueError("The funnel does not add up to the removal list.")
    return frame


# -----------------------------------------------------------------------------
# Pattern sweep, edge cases and the audit of run 1.1


def check_sweep(stage: pd.DataFrame, revised_ids: set[str], removals: pd.DataFrame) -> dict:
    """Reproduce the string patterns of the sweep and count the edge-case groups that were read and kept."""
    by_category = removals.groupby("category").groups
    for category, pattern in SWEEP_PATTERNS.items():
        hits = set(stage.loc[stage["question"].str.contains(pattern, regex=False), "annotation_id"])
        if hits != set(by_category[category]):
            raise ValueError(f"The pattern for {category} does not select exactly the listed questions.")
    instruction = stage[stage["question"].str.lstrip().str.startswith(INSTRUCTION_PREFIXES)]
    if not set(by_category["truncated_stem"]) <= set(instruction["annotation_id"]):
        raise ValueError("A truncated stem does not start with the instruction prefix.")
    last_choice = stage["choices"].map(lambda choices: choices[-1])
    next_passage = stage[last_choice.str.contains(NEXT_PASSAGE_PATTERN)]
    if set(next_passage["annotation_id"]) & set(removals.index):
        raise ValueError("A row with the next passage in its last choice is in the removal list.")
    for key in KEPT_EDGE_CASES:
        matches = stage[(stage[IDENTITY_FIELDS] == key).all(axis=1)]["annotation_id"]
        if len(matches) != 1 or matches.iat[0] not in revised_ids or matches.iat[0] in removals.index:
            raise ValueError(f"Edge case {key} was not kept in the revised stage 04.")
    return {"instruction_rows": instruction, "instruction_by_exam": instruction["exam"].value_counts(),
            "instruction_by_edition": instruction.groupby(["exam", "exam_edition"]).size(),
            "instruction_kept": len(instruction) - len(by_category["truncated_stem"]),
            "next_passage_rows": next_passage, "next_passage_by_edition": next_passage["exam_edition"].value_counts()}


def check_audit(removals: pd.DataFrame) -> dict:
    """Source issues tagged by the engineering audit of run 1.1, split into removed and kept."""
    audit = read_json(AUDIT_PATH)
    cases = audit["cases"]
    source_issues = {identity: case for identity, case in cases.items() if "source_issue" in case["categories"]}
    removed = sorted(set(source_issues) & set(removals.index))
    kept = [source_issues[identity] for identity in sorted(set(source_issues) - set(removals.index))]
    not_audited = sorted(set(removals.index) - set(cases))
    return {"reviewer": audit["reviewer"], "cases": len(cases), "source_issues": len(source_issues),
            "removed": removed, "kept": kept, "not_audited": not_audited}


# -----------------------------------------------------------------------------
# Tables


def category_table(removals: pd.DataFrame, revisions: dict[str, dict]) -> pd.DataFrame:
    removed_06 = {entry["annotation_id"] for entry in read_jsonl(REVISED_06 / "removals.jsonl")}
    rows = []
    for category, group in removals.groupby("category", sort=False):
        sources = group["source"].value_counts()
        rows.append({"category": category, "listed": len(group), "stage_04": len(group),
                     "stage_06": len(set(group.index) & removed_06),
                     **{source: int(sources.get(source, 0)) for source in SOURCE_ORDER},
                     "exams": ", ".join(sorted(set(group["exam"])))})
    frame = pd.DataFrame(rows).set_index("category")
    if frame["stage_04"].sum() != revisions["04"]["removed_rows"] or frame["stage_06"].sum() != revisions["06"]["removed_rows"]:
        raise ValueError("Per-category counts do not add up to the revised manifests.")
    return frame


def dataset_table(revisions: dict[str, dict], post: dict, run_14: dict, full_rows: int) -> pd.DataFrame:
    source = run_14["identity"]["source"]
    rows = [{"dataset": "filtered corpus, 17-item list", "input": full_rows, "removed": full_rows - source["rows"],
             "absent": 0, "output": source["rows"], "hub": source["identifier"], "revision": source["revision"][:8]}]
    for name, label in [("04", "filtered corpus, 19-item list"), ("06", "semantically deduplicated corpus"),
                        ("subset", "abstentions of run 1.2, 15-item list")]:
        manifest = revisions[name]
        hub = manifest.get("hub", {})
        rows.append({"dataset": label, "input": manifest["input_rows"], "removed": manifest["removed_rows"],
                     "absent": len(manifest["listed_ids_absent_from_input"]), "output": manifest["output_rows"],
                     "hub": hub.get("repository", "--"), "revision": hub.get("revision", "--")[:8]})
    rows.append({"dataset": "annotated export of run 1.4", "input": post["input_rows"], "removed": len(post["removed"]),
                 "absent": 0, "output": post["output_rows"], "hub": post["hub"]["repository"],
                 "revision": post["hub"]["revision"][:8]})
    return pd.DataFrame(rows).set_index("dataset")


# -----------------------------------------------------------------------------
# Main


def check_inputs() -> list[Path]:
    required = [REMOVALS_PATH, STAGE_04, STAGE_06, REVISED_04, REVISED_06, SUBSET_INPUT, SUBSET_REVISED, POST_DIR,
                AUDIT_PATH, *FULL_RUNS.values(), *SUBSET_RUNS.values()]
    return [path for path in required if not path.exists()]


def main() -> int:
    missing = check_inputs()
    if missing:
        print("Missing inputs:", *missing, sep="\n  ")
        return 1
    removals = load_removals()
    stage_04 = load_stage(STAGE_04)
    stage_06 = load_stage(STAGE_06)
    subset_input = load_stage(SUBSET_INPUT)
    check_listed_rows(stage_04, removals)
    revisions = {"04": check_revision(REVISED_04, stage_04, removals, check_list_hash=True),
                 "06": check_revision(REVISED_06, stage_06, removals, check_list_hash=True),
                 "subset": check_revision(SUBSET_REVISED, subset_input, removals, check_list_hash=False)}
    full, subsets, manifests = {}, {}, {}
    for version, run_dir in FULL_RUNS.items():
        full[version], manifests[version] = load_export(run_dir, EXPECTED_RUN_IDS[version])
    for version, run_dir in SUBSET_RUNS.items():
        subsets[version], _ = load_export(run_dir, EXPECTED_RUN_IDS[f"{version}-subset"])
    post = read_json(POST_DIR / "manifest.json")
    detection = check_detection(removals, full, subsets, post)
    revised_ids = set(load_stage(REVISED_04 / "huggingface")["annotation_id"])
    sweep = check_sweep(stage_04, revised_ids, removals)
    audit = check_audit(removals)

    print("== Removal list")
    print(f"File: {REMOVALS_PATH.relative_to(REPO_ROOT)}, sha256 {file_sha256(REMOVALS_PATH)}")
    print(f"Created {removals.attrs['created_at']}, updated {removals.attrs['updated_at']}")
    print(f"Listed questions: {len(removals)} ({100 * len(removals) / len(stage_04):.3f}% of {len(stage_04):,} rows of stage 04)")
    print("By detection source:", removals["source"].value_counts().reindex(SOURCE_ORDER).to_dict())
    print("By exam:", removals["exam"].value_counts().to_dict())
    print_frame("By category (listed, removed from stage 04 and 06, detection sources, exams):",
                category_table(removals, revisions))
    print_frame("Listed questions:", removals[IDENTITY_FIELDS + ["category", "source"]].reset_index(drop=True))

    print("\n== Detection")
    print_frame("Funnel from run 1.2 to the post-annotation:", funnel_table(removals, full, subsets, detection))
    print("Run ids:", EXPECTED_RUN_IDS)
    source = manifests["1.4"]["identity"]["source"]
    print(f"Run 1.4 source: {source['identifier']} revision {source['revision']} ({source['rows']:,} rows)")
    kept = subsets["1.3"][subsets["1.3"]["annotation_id"].isin(detection["kept_from_subset"])]
    print("Abstention of run 1.3 kept in the dataset:", [item_label(row) for row in kept.to_dict("records")])
    print_frame("Answer key and subject of each IME copy by run:",
                ime_table(removals, stage_04, {"1.2": full["1.2"], "1.3": subsets["1.3"], "1.4-subset": subsets["1.4"]}))
    print(f"Abstentions of run 1.4: {len(detection['uncertain']['1.4'])} -> manual {len(detection['post_manual'])}, removed {len(detection['post_removed'])}")

    print("\n== Sweep and edge cases")
    sweep_rows = full["1.2"].set_index("annotation_id").loc[list(ids_by_source(removals, "pattern_sweep"))]
    print_frame("Label given by run 1.2 to the questions found by the sweep:",
                sweep_rows[IDENTITY_FIELDS + ["subject", "subject_confidence", "annotation_status"]].reset_index(drop=True))
    for category, pattern in SWEEP_PATTERNS.items():
        print(f"Pattern {pattern!r}: {len(removals[removals['category'] == category])} rows, all listed ({category})")
    print(f"Rows starting with {INSTRUCTION_PREFIXES}: {len(sweep['instruction_rows'])}, by exam {sweep['instruction_by_exam'].to_dict()}; kept {sweep['instruction_kept']}")
    print_frame("Rows with the instruction header by edition:", sweep["instruction_by_edition"].to_frame("rows"))
    print(f"Rows whose last choice announces the next passage: {len(sweep['next_passage_rows'])}, by edition {sweep['next_passage_by_edition'].to_dict()}; all kept")
    print("Edge cases confirmed present in the revised stage 04:", KEPT_EDGE_CASES)

    print("\n== Audit of run 1.1")
    print(f"Reviewer: {audit['reviewer']}; reviewed cases {audit['cases']}, tagged source_issue {audit['source_issues']}")
    print(f"Source issues later removed: {len(audit['removed'])}; kept: {len(audit['kept'])}; listed questions not in the audit: {len(audit['not_audited'])}")
    for case in audit["kept"]:
        print("  kept:", item_label(case), "|", case["assessment"][:120])
    for identity in audit["not_audited"]:
        print("  not audited:", item_label(removals.loc[identity]))

    print("\n== Revised datasets")
    print_frame("Rows per dataset:", dataset_table(revisions, post, manifests["1.4"], len(stage_04)))
    print_frame("Listed ids absent from stage 06, each an exact-deduplication twin of a listed row the stage kept:",
                absent_twins(stage_04, removals, revisions["06"]))
    print("Kept rows of every revised output are identical to the input rows, in input order: checked")

    print("\n== Hub")
    for name, manifest in [("04 - revised", revisions["04"]), ("06 - revised", revisions["06"])]:
        hub = manifest["hub"]
        print(f"{name}: {hub['repository']} revision {hub['revision']} pushed {hub['pushed_at']}")
    print(f"post-annotation: {post['hub']['repository']} revision {post['hub']['revision']} pushed {post['hub']['pushed_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
