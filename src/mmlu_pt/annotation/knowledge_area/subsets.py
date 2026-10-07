"""Prepare reproducible abstention experiments and compare their exported results."""

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk

from .config import PACKAGE_DIR, UNCERTAIN, digest, timestamp, write_json
from .dataset import annotation_id, metadata
from .export import ANNOTATION_FEATURES
from .reporting import write_csv
from .taxonomy import load_taxonomy
from .validation import Annotation

DEFAULT_REVIEW_NOTES = PACKAGE_DIR / "mmlu_pt_taxonomy_v1_3_evidence.json"
REMOVED_CATEGORY = "removed_from_dataset"
RESULT_FIELDS = ("subject", "macro_area", "alternative_subject", "subject_confidence",
                 "subject_justification", "annotation_status", "annotation_agreement",
                 "annotation_pass_1_subject", "annotation_pass_2_subject", "taxonomy_version")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def exported_run(run_dir: Path) -> tuple[dict, dict, list[dict], Path]:
    """Read published artifacts only; never open or migrate the checkpoint."""
    manifest = read_json(run_dir / "manifest.json")
    latest = read_json(run_dir / "exports" / "latest.json")
    directory = latest["directory"]
    if not isinstance(directory, str) or Path(directory).name != directory:
        raise ValueError("Invalid export directory in latest.json")
    path = run_dir / "exports" / directory / "data.jsonl"
    if not path.resolve().is_relative_to((run_dir / "exports").resolve()):
        raise ValueError("Export path escapes the run directory")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != latest["rows"] or len(rows) != manifest["selected_rows"]:
        raise ValueError("Export row count differs from its manifest")
    ids = [r["annotation_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate annotation IDs in published export")
    if any(r["annotation_run_id"] != manifest["run_id"] for r in rows):
        raise ValueError("Export contains a different run ID")
    return manifest, latest, rows, path


def new_destination(path: Path, source_runs: tuple[Path, ...]) -> None:
    resolved = path.resolve()
    if path.exists() or path.is_symlink():
        raise ValueError(f"Output already exists; choose a new directory: {path}")
    if any(resolved.is_relative_to(run.resolve()) for run in source_runs):
        raise ValueError("Output must be outside the source runs")


def prepare_subset(source_run: Path, output_dir: Path) -> dict:
    new_destination(output_dir, (source_run,))
    manifest, latest, rows, export_path = exported_run(source_run)
    source = load_from_disk(str(source_run / "source"))
    if not isinstance(source, Dataset) or len(source) != len(rows):
        raise ValueError("Source snapshot does not match the export")
    reserved = set(source.column_names) & (set(ANNOTATION_FEATURES) | {"source_row_index", "source_run_id"})
    if reserved:
        raise ValueError(f"Source has reserved columns: {sorted(reserved)}")
    positions, mapping = [], []
    indices = manifest["selected_indices"]
    if len(indices) != len(rows):
        raise ValueError("Source index mapping is incomplete")
    for position, row in enumerate(rows):
        if row["subject"] != UNCERTAIN:
            continue
        original = source[position]
        identity = annotation_id(original)
        if identity != row["annotation_id"] or row["annotation_row_index"] != indices[position]:
            raise ValueError(f"Source identity/index mismatch at position {position}")
        # Answer is checked for preservation only; it is never exported to review material.
        if any(key not in row or row[key] != value for key, value in original.items()):
            raise ValueError(f"Source content differs from export at position {position}")
        positions.append(position)
        mapping.append({"position": len(mapping), "source_row_index": indices[position],
                        "annotation_id": identity})
    if not positions:
        raise ValueError("Source run has no final UNCERTAIN rows")
    # add_column otherwise flattens the selection into a cache beside the source Arrow file.
    subset = source.select(positions).flatten_indices(keep_in_memory=True)
    subset = subset.add_column("source_row_index", [r["source_row_index"] for r in mapping])
    subset = subset.add_column("source_run_id", [manifest["run_id"]] * len(mapping))
    provenance = {"format_version": 1, "created_at": timestamp(), "selection": {"subject": UNCERTAIN},
                  "source_run_id": manifest["run_id"], "source_run": str(source_run.resolve()),
                  "source_generation": latest["generation"], "source": manifest["source"],
                  "source_taxonomy_version": manifest["taxonomy_version"],
                  "source_manifest_sha256": file_hash(source_run / "manifest.json"),
                  "source_export_sha256": file_hash(export_path),
                  "source_snapshot_sha256": {p.name: file_hash(p) for p in sorted((source_run / "source").iterdir()) if p.is_file()},
                  "rows": len(mapping), "mapping": mapping,
                  "identity_note": "annotation_row_index is local to this subset; source_row_index preserves the original index."}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        subset.save_to_disk(str(stage / "dataset"))
        provenance["dataset_sha256"] = {p.name: file_hash(p) for p in sorted((stage / "dataset").iterdir()) if p.is_file()}
        write_json(stage / "subset.json", provenance)
        (stage / "README.md").write_text(
            f"# UNCERTAIN subset\n\n{len(subset)} original records from run `{manifest['run_id']}`.\n\n"
            "Use `--dataset-path <this-directory>/dataset` without `--dry-run`.\n"
            "Copy this entire directory to the inference machine. No access to the parent run is required for annotation.\n"
            "The dataset preserves answer keys; classification input excludes them. No old annotation columns are included.\n",
            encoding="utf-8")
        if output_dir.exists():
            raise ValueError(f"Output already exists: {output_dir}")
        os.rename(stage, output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return provenance


def label_issues(row: dict, taxonomy) -> list[str]:
    candidates = taxonomy.resolve_candidates(row["exam"], metadata(row))
    issues = []
    if row["annotation_status"] == "error":
        if row["subject"] is not None or row["macro_area"] is not None:
            issues.append("error_with_assigned_label")
    else:
        try:
            Annotation.model_validate({"subject": row["subject"], "alternative_subject": row["alternative_subject"],
                                       "confidence": row["subject_confidence"], "justification": row["subject_justification"]})
        except ValueError:
            issues.append("invalid_final_schema")
        if row["subject"] not in candidates and row["subject"] != UNCERTAIN:
            issues.append("subject_outside_candidates")
        if row["alternative_subject"] is not None and row["alternative_subject"] not in candidates:
            issues.append("alternative_outside_candidates")
        if row["macro_area"] != taxonomy.macro_areas.get(row["subject"]):
            issues.append("macro_area_mismatch")
        if row["annotation_status"] not in {"accepted", "adjudicated", "uncertain"}:
            issues.append("invalid_annotation_status")
        if (row["subject"] == UNCERTAIN) != (row["annotation_status"] == "uncertain"):
            issues.append("status_subject_mismatch")
    for key in ("annotation_pass_1_subject", "annotation_pass_2_subject"):
        if row[key] is not None and row[key] not in candidates and row[key] != UNCERTAIN:
            issues.append(f"{key}_outside_candidates")
    if row["taxonomy_version"] != taxonomy.version:
        issues.append("taxonomy_version_mismatch")
    return issues


def compare_subset(source_run: Path, run_dir: Path, output_dir: Path,
                   review_notes: Path = DEFAULT_REVIEW_NOTES) -> dict:
    new_destination(output_dir, (source_run, run_dir))
    baseline, _, original_rows, original_path = exported_run(source_run)
    experimental, _, new_rows, new_path = exported_run(run_dir)
    expected = {r["annotation_id"]: r for r in original_rows if r["subject"] == UNCERTAIN}
    if not expected:
        raise ValueError("Baseline contains no UNCERTAIN rows")
    observed = {r["annotation_id"]: r for r in new_rows}
    taxonomy = load_taxonomy(run_dir / "taxonomy.json", run_dir / "exam_aliases.json")
    if taxonomy.checksum != experimental["identity"]["taxonomy_hash"]:
        raise ValueError("Experimental taxonomy differs from its manifest")
    if digest(taxonomy.aliases) != experimental["identity"]["aliases_hash"]:
        raise ValueError("Experimental aliases differ from their manifest")
    notes_document = read_json(review_notes)
    notes = {r["annotation_id"]: r for r in notes_document["cases"]}
    counts, cases, violations = Counter(), [], []
    for identity, old in expected.items():
        row = observed.get(identity)
        note = notes.get(identity, {})
        item = {"annotation_id": identity, "source_row_index": old["annotation_row_index"],
                "exam": old["exam"], "exam_edition": old.get("exam_edition"), "num": old.get("num"),
                "question": old["question"], "choices": old["choices"],
                "review_category": note.get("category", "not_reviewed"),
                "source_issue": note.get("source_issue", False), "review_note": note.get("assessment"),
                **{"previous_" + k: old.get(k) for k in RESULT_FIELDS},
                **{"new_" + k: row.get(k) if row else None for k in RESULT_FIELDS}}
        issues = []
        if row is None:
            # Rows removed by the dataset revision are expected to be absent from later experiments.
            outcome = "removed" if note.get("category") == REMOVED_CATEGORY else "missing"
        else:
            if annotation_id(row) != identity or any(
                k not in row or k not in old or row[k] != old[k] for k in baseline["source"]["features"]
            ):
                issues.append("source_content_mismatch")
            if row.get("source_row_index") != old["annotation_row_index"] or row.get("source_run_id") != baseline["run_id"]:
                issues.append("source_provenance_mismatch")
            try:
                issues.extend(label_issues(row, taxonomy))
            except (ValueError, KeyError, TypeError):
                issues.append("invalid_annotation_record")
            outcome = ("invalid" if issues else "error" if row["annotation_status"] == "error"
                       else "uncertain" if row["subject"] == UNCERTAIN else "received_subject")
        counts[outcome] += 1
        if issues:
            violations.append({"annotation_id": identity, "issues": issues})
        item.update({"outcome": outcome, "issues": issues})
        cases.append(item)
    missing = sorted(identity for identity in expected.keys() - observed.keys()
                     if notes.get(identity, {}).get("category") != REMOVED_CATEGORY)
    unexpected = sorted(observed.keys() - expected.keys())
    summary = {"created_at": timestamp(), "baseline_run_id": baseline["run_id"],
               "experimental_run_id": experimental["run_id"], "taxonomy_version": taxonomy.version,
               "expected_rows": len(expected), "observed_rows": len(observed),
               "counts": {k: counts[k] for k in ("received_subject", "uncertain", "error", "missing", "removed", "invalid")},
               "missing_ids": missing, "unexpected_ids": unexpected, "violations": violations,
               "valid": not (missing or unexpected or violations),
               "source_issue_cases": sum(c["source_issue"] for c in cases),
               "baseline_export_sha256": file_hash(original_path), "experimental_export_sha256": file_hash(new_path),
               "review_notes_sha256": file_hash(review_notes),
               "interpretation": "Received subject is not a verified correction; confidence is uncalibrated. Review notes are engineering assessments, not gold labels."}
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "comparison.json", summary)
    write_csv(output_dir / "cases.csv", cases, list(cases[0]))
    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    lines = ["# Comparação do subset UNCERTAIN", "", f"Baseline: `{baseline['run_id']}`; experimento: `{experimental['run_id']}`.",
             "", f"Integridade: {'válida' if summary['valid'] else 'FALHOU; consulte comparison.json'}.",
             f"Receberam disciplina: {counts['received_subject']}; continuam UNCERTAIN: {counts['uncertain']}; erros: {counts['error']}.",
             f"Ausentes: {len(missing)}; removidos do dataset: {counts['removed']}; inesperados: {len(unexpected)}; inválidos: {counts['invalid']}.",
             "", "Receber disciplina não comprova correção. Confiança não é calibrada. Nenhum gabarito é exportado.", ""]
    for source_issue in (True, False):
        lines.extend(["## " + ("Problemas conhecidos de fonte" if source_issue else "Cobertura e classificação"), "",
                      "| Índice original | Exame | Resultado | Disciplina |", "|---:|---|---|---|"])
        for case in cases:
            if case["source_issue"] == source_issue:
                lines.append(f"| {case['source_row_index']} | {case['exam']} | {case['outcome']} | {case['new_subject'] or '—'} |")
        lines.append("")
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
