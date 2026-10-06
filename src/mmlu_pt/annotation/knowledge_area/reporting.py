"""Quality-control summaries and answer-free human review exports."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset

from .config import UNCERTAIN, canonical_json, timestamp, write_json


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical_json(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    temporary.replace(path)


def generate_report(data: Dataset, audits: list[dict], validation: dict, directory: Path,
                    run_id: str, generation: int) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    subjects, areas, pairs, statuses = Counter(), Counter(), Counter(), Counter()
    by_exam = defaultdict(lambda: Counter(total=0, uncertain=0, errors=0, agreements=0,
                                         disagreements=0, comparable_pairs=0, adjudications=0))
    distributions = defaultdict(Counter)
    confidences, histogram = [], Counter({f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)})
    for row, audit in zip(data, audits, strict=True):
        exam, status = row["exam"], row["annotation_status"]
        stats = by_exam[exam]
        stats["total"] += 1
        statuses[status] += 1
        stats["errors"] += status == "error"
        stats["uncertain"] += row["subject"] == UNCERTAIN
        stats["adjudications"] += audit["adjudication_requested"]
        first, second = row["annotation_pass_1_subject"], row["annotation_pass_2_subject"]
        if first is not None and second is not None:
            stats["comparable_pairs"] += 1
            stats["agreements"] += first == second and first != UNCERTAIN
            stats["disagreements"] += first != second
            if first != second:
                pairs[tuple(sorted((first, second)))] += 1
        if row["subject"] is not None:
            subjects[row["subject"]] += 1
            distributions[exam][row["subject"]] += 1
        if row["macro_area"] is not None:
            areas[row["macro_area"]] += 1
        confidence = row["subject_confidence"]
        if confidence is not None:
            confidences.append(confidence)
            bucket = min(int(confidence * 10), 9)
            histogram[f"{bucket/10:.1f}-{(bucket+1)/10:.1f}"] += 1
    exams = []
    for exam, stats in sorted(by_exam.items()):
        exams.append({"exam": exam, **stats,
                      "uncertain_rate": rate(stats["uncertain"], stats["total"]),
                      "disagreement_rate": rate(stats["disagreements"], stats["comparable_pairs"]),
                      "agreement_rate": rate(stats["agreements"], stats["comparable_pairs"])})
    totals = Counter()
    for stats in by_exam.values():
        totals.update(stats)
    report = {
        "run_id": run_id, "generation": generation, "generated_at": timestamp(), "total_rows": len(data),
        "successfully_annotated_rows": statuses["accepted"] + statuses["adjudicated"],
        "errors": statuses["error"], "uncertain_count": totals["uncertain"],
        "uncertain_rate": rate(totals["uncertain"], len(data)),
        "comparable_independent_pairs": totals["comparable_pairs"],
        "agreement_count": totals["agreements"], "agreement_rate": rate(totals["agreements"], totals["comparable_pairs"]),
        "disagreement_count": totals["disagreements"], "disagreement_rate": rate(totals["disagreements"], totals["comparable_pairs"]),
        "adjudication_count": totals["adjudications"], "adjudication_rate": rate(totals["adjudications"], len(data)),
        "statuses": dict(statuses), "subject_counts": dict(subjects.most_common()),
        "macro_area_counts": dict(areas.most_common()), "by_exam": exams,
        "subject_distribution_by_exam": {e: dict(counts) for e, counts in sorted(distributions.items())},
        "confidence": {"count": len(confidences), "mean": sum(confidences) / len(confidences) if confidences else None,
                       "min": min(confidences) if confidences else None, "max": max(confidences) if confidences else None,
                       "histogram": dict(histogram), "calibrated": False},
        "disagreement_pairs": [{"subject_1": pair[0], "subject_2": pair[1], "count": count}
                               for pair, count in pairs.most_common()],
        "unmapped_exam_values": validation.get("unmapped_exams", []),
        "denominators": {"uncertain_rate": "all selected rows", "adjudication_rate": "all selected rows",
                         "agreement_rate": "rows with two successful independent structured results; UNCERTAIN is never agreement",
                         "disagreement_rate": "rows with two successful independent results whose labels differ"}}
    write_json(directory / "report.json", report)
    write_csv(directory / "by_exam.csv", exams, ["exam", "total", "uncertain", "errors", "agreements",
              "disagreements", "comparable_pairs", "adjudications", "uncertain_rate", "disagreement_rate", "agreement_rate"])
    distribution_rows = [{"exam": e, "subject": s, "count": n} for e, counts in sorted(distributions.items())
                         for s, n in counts.most_common()]
    write_csv(directory / "subjects_by_exam.csv", distribution_rows, ["exam", "subject", "count"])
    write_csv(directory / "disagreement_pairs.csv", report["disagreement_pairs"], ["subject_1", "subject_2", "count"])
    lines = [f"# Subject annotation QC: {run_id}", "", f"Generation: {generation}", "",
             f"- Total rows: {len(data)}", f"- Accepted subjects: {report['successfully_annotated_rows']}",
             f"- Errors: {report['errors']}", f"- UNCERTAIN: {report['uncertain_count']} ({report['uncertain_rate']})",
             f"- Independent agreement: {report['agreement_rate']} ({totals['comparable_pairs']} comparable pairs)",
             f"- Adjudications requested: {report['adjudication_count']} ({report['adjudication_rate']})", "",
             "Confidence is model-reported and uncalibrated. Null rates have no eligible denominator.", "",
             "Agreement excludes UNCERTAIN. Disagreement compares different labels. Adjudication counts include failed requests.", "",
             "## By exam", "", "| Exam | Rows | Errors | UNCERTAIN | Agreement | Disagreement |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in exams:
        lines.append(f"| {row['exam']} | {row['total']} | {row['errors']} | {row['uncertain']} | {row['agreement_rate']} | {row['disagreement_rate']} |")
    for title, counts in (("Subjects", subjects), ("Macro areas", areas)):
        lines.extend(["", f"## {title}", "", "| Label | Count |", "|---|---:|"])
        lines.extend(f"| {label} | {count} |" for label, count in counts.most_common())
    lines.extend(["", "## Confidence distribution", "", "```json", canonical_json(report["confidence"]), "```",
                  "", "## Common disagreement pairs", "", "| Subject 1 | Subject 2 | Count |", "|---|---|---:|"])
    lines.extend(f"| {a} | {b} | {n} |" for (a, b), n in pairs.most_common(30))
    lines.extend(["", "## Subject distribution by exam", "", "See `subjects_by_exam.csv` for the complete distribution.",
                  "", f"Unmapped exams: {report['unmapped_exam_values']}", ""])
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def export_review(data: Dataset, directory: Path, confidence_below: float | None = None,
                  include_disagreements: bool = False, include_answer: bool = False) -> int:
    if confidence_below is not None and not 0 <= confidence_below <= 1:
        raise ValueError("confidence_below must be in [0, 1]")
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in data:
        confidence = row["subject_confidence"]
        required = row["annotation_status"] in ("uncertain", "error")
        low = confidence_below is not None and confidence is not None and confidence < confidence_below
        disagreement = include_disagreements and row["annotation_agreement"] == "disagreement"
        if not (required or low or disagreement):
            continue
        selected = {name: row.get(name) for name in ("annotation_id", "annotation_row_index", "exam", "exam_edition",
                    "question", "choices", "subject", "alternative_subject", "subject_confidence", "subject_justification",
                    "annotation_status", "annotation_agreement", "annotation_pass_1_subject", "annotation_pass_2_subject")}
        if include_answer:
            selected["answer"] = row["answer"]
        rows.append(selected)
    fields = ["annotation_id", "annotation_row_index", "exam", "exam_edition", "question", "choices", "subject",
              "alternative_subject", "subject_confidence", "subject_justification", "annotation_status",
              "annotation_agreement", "annotation_pass_1_subject", "annotation_pass_2_subject"]
    if include_answer:
        fields.append("answer")
    write_csv(directory / "manual_review.csv", rows, fields)
    temporary = directory / "manual_review.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(directory / "manual_review.jsonl")
    return len(rows)
