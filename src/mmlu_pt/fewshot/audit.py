"""Automatic checks on few-shot candidates and their rationales, plus the audit report."""

import re
from pathlib import Path

from datasets import Dataset

from .config import LETTERS, RunConfig, write_json
from .selection import normalize_text

ANSWER_LINE = re.compile(r"^Resposta: ([A-E])$")


def pre_rationale_checks(data: Dataset, ids: list[str], index: int, text_index: dict,
                         config: RunConfig) -> list[str]:
    """Reasons a source row cannot serve as a few-shot example; empty means accepted."""
    row = data[index]
    question, choices, answer = text_index["questions"][index], row["choices"], row["answer"]
    reasons = []
    if not question:
        reasons.append("empty_question")
    if not isinstance(choices, list) or len(choices) not in (4, 5):
        reasons.append("choice_count")
    elif any(not normalize_text(choice) for choice in choices):
        reasons.append("empty_choice")
    elif len({normalize_text(choice) for choice in choices}) != len(choices):
        reasons.append("duplicate_choices")
    if not isinstance(answer, int) or not 0 <= answer < len(choices or []):
        reasons.append("answer_out_of_range")
    if text_index["counts"][question] > 1:
        reasons.append("duplicate_question_in_source")
    reasons += shared_support_text(row, index, ids, text_index, config.shared_prefix_chars)
    return reasons


def shared_support_text(row: dict, index: int, ids: list[str], text_index: dict, min_chars: int) -> list[str]:
    """Other rows of the same exam edition whose question starts with the same long prefix."""
    questions = text_index["questions"]
    prefix = questions[index][:min_chars]
    if len(prefix) < min_chars:
        return []
    group = text_index["groups"][(row["exam"], row["exam_edition"] or "")]
    return [f"shared_support_text:{ids[other][:12]}" for other in group
            if other != index and questions[other].startswith(prefix)]


def rationale_checks(record: dict | None, gold: str, choice_count: int, config: RunConfig) -> list[str]:
    """Reasons a generated rationale is unusable; empty means accepted."""
    if record is None or record.get("status") != "ok":
        return ["rationale_missing"]
    explanation, answer, rationale = record["explanation"], record["answer_letter"], record["rationale"]
    reasons = []
    if not explanation.strip():
        reasons.append("rationale_empty")
    if len(rationale) > config.rationale_max_chars:
        reasons.append("rationale_too_long")
    if "Resposta:" in explanation:
        reasons.append("rationale_extra_answer_line")
    if answer not in LETTERS[:choice_count]:
        reasons.append("rationale_letter_not_in_choices")
    if answer != gold:
        reasons.append("rationale_letter_mismatch")
    match = ANSWER_LINE.match(rationale.splitlines()[-1]) if rationale else None
    if match is None or match.group(1) != gold:
        reasons.append("rationale_final_line_invalid")
    return reasons


def final_checks(built: dict, source_rows: int, shots: int,
                 expected_test_ids: set[str] | None = None) -> dict:
    """Disjointness and count checks over the built configs of every level."""
    dev_ids, test_ids, problems = set(), set(), []
    for level, configs in built.items():
        for name, dataset_dict in configs.items():
            dev, test = dataset_dict["dev"], dataset_dict["test"]
            if len(dev) != shots:
                problems.append(f"{level}/{name}: dev has {len(dev)} rows, expected {shots}")
            if dev.features != test.features:
                problems.append(f"{level}/{name}: dev and test features differ")
            for label, subset, seen in (("dev", dev, dev_ids), ("test", test, test_ids)):
                identifiers = subset["id"]
                if len(set(identifiers)) != len(identifiers) or seen.intersection(identifiers):
                    problems.append(f"{level}/{name}: duplicate {label} IDs")
                seen.update(identifiers)
    overlap = dev_ids & test_ids
    if overlap:
        problems.append(f"{len(overlap)} ids appear in both dev and test")
    if expected_test_ids is not None and test_ids != expected_test_ids:
        problems.append(f"Test IDs differ: missing={len(expected_test_ids - test_ids)}, extra={len(test_ids - expected_test_ids)}")
    if expected_test_ids is None and len(dev_ids) + len(test_ids) != source_rows:
        problems.append(f"dev + test = {len(dev_ids) + len(test_ids)} rows, source has {source_rows}")
    return {"dev_rows": len(dev_ids), "test_rows": len(test_ids), "source_rows": source_rows,
            "overlap": len(overlap), "problems": problems}


def write_reports(run_dir: Path, selection: list[dict], final: dict) -> None:
    write_json(run_dir / "audit.json", {"strata": selection, "final": final})
    lines = ["# Few-shot audit", "", f"- dev rows: {final['dev_rows']}", f"- test rows: {final['test_rows']}",
             f"- source rows: {final['source_rows']}", f"- dev/test overlap: {final['overlap']}"]
    lines += [f"- PROBLEM: {problem}" for problem in final["problems"]]
    for stratum in selection:
        lines += ["", f"## {stratum['level']} / {stratum['macro_area']} (`{stratum['config_name']}`)", "",
                  f"rows: {stratum['stratum_rows']}; candidates considered: {stratum['ranking_considered']}; "
                  f"accepted: {len(stratum['dev'])}; excluded: {len(stratum['excluded'])}", "",
                  "| rank | id | exam | edition | num | answer | status |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for entry in stratum["dev"]:
            lines.append(f"| {entry['rank_position']} | {entry['id'][:12]} | {entry['exam']} | {entry['exam_edition']} "
                         f"| {entry['num']} | {entry['answer_letter']} | {entry['status']} |")
        for entry in stratum["excluded"]:
            lines.append(f"| {entry['rank_position']} | {entry['id'][:12]} | {entry['exam']} | {entry['exam_edition']} "
                         f"| {entry['num']} | {entry['answer_letter']} | excluded ({entry['stage']}): "
                         f"{', '.join(entry['reasons'])} |")
    (run_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
