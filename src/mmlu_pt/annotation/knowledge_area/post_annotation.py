"""Final annotated dataset: manual assignments and dataset removals applied to a published export."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from datasets import Dataset, load_from_disk

from .config import UNCERTAIN, timestamp, write_json
from .dataset import annotation_id, metadata
from .subsets import exported_run, file_hash, new_destination
from .taxonomy import load_taxonomy

DEFAULT_MANUAL = Path("config/manual_annotations.json")
DEFAULT_REMOVALS = Path("config/removed_questions.json")
MANUAL_STATUS = "manual"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def indexed(entries: list[dict]) -> dict[str, dict]:
    result = {}
    for entry in entries:
        if entry["annotation_id"] in result:
            raise ValueError(f"Duplicate annotation_id: {entry['annotation_id']}")
        result[entry["annotation_id"]] = entry
    return result


def check_identity(row: dict, entry: dict) -> None:
    if any(row[field] != entry[field] for field in ("exam", "exam_edition", "num")):
        raise ValueError(f"Configuration metadata differs from the export for {entry['annotation_id']}")


def apply_manual(row: dict, entry: dict, taxonomy) -> dict:
    """Replace the model result by the manual assignment; passes and agreement are kept for audit."""
    candidates = taxonomy.resolve_candidates(row["exam"], metadata(row))
    if entry["subject"] not in candidates:
        raise ValueError(f"Manual subject outside exam candidates for {entry['annotation_id']}")
    alternative = entry.get("alternative_subject")
    if alternative is not None and alternative not in candidates:
        raise ValueError(f"Manual alternative outside exam candidates for {entry['annotation_id']}")
    if row["subject"] != UNCERTAIN:
        raise ValueError(f"Manual assignment targets a row that is not UNCERTAIN: {entry['annotation_id']}")
    return row | {"subject": entry["subject"], "macro_area": taxonomy.macro_areas[entry["subject"]],
                  "alternative_subject": alternative, "subject_confidence": None,
                  "subject_justification": entry["justification"], "annotation_status": MANUAL_STATUS}


def finalize(run_dir: Path, output_dir: Path, manual_path: Path = DEFAULT_MANUAL,
             removals_path: Path = DEFAULT_REMOVALS) -> dict:
    new_destination(output_dir, (run_dir / "exports",))
    manifest, latest, rows, export_path = exported_run(run_dir)
    taxonomy = load_taxonomy(run_dir / "taxonomy.json", run_dir / "exam_aliases.json")
    manual = indexed(read_json(manual_path)["assignments"])
    removals = indexed(read_json(removals_path)["removals"])
    if manual.keys() & removals.keys():
        raise ValueError("An annotation_id is both manually assigned and removed")
    kept, applied, removed = [], [], []
    for row in rows:
        identity = row["annotation_id"]
        if annotation_id(row) != identity:
            raise ValueError(f"Export identity mismatch at row {row['annotation_row_index']}")
        if identity in removals:
            check_identity(row, removals[identity])
            removed.append({"annotation_row_index": row["annotation_row_index"], "annotation_id": identity,
                            "exam": row["exam"], "exam_edition": row["exam_edition"], "num": row["num"],
                            "model_subject": row["subject"], "category": removals[identity]["category"]})
            continue
        if identity in manual:
            check_identity(row, manual[identity])
            row = apply_manual(row, manual[identity], taxonomy)
            applied.append({"annotation_row_index": row["annotation_row_index"], "annotation_id": identity,
                            "exam": row["exam"], "exam_edition": row["exam_edition"], "num": row["num"],
                            "subject": row["subject"], "category": manual[identity]["category"]})
        kept.append(row)
    unapplied = sorted(manual.keys() - {a["annotation_id"] for a in applied})
    if unapplied:
        raise ValueError(f"Manual assignments not found in the export: {unapplied}")
    remaining = [r["annotation_row_index"] for r in kept if r["subject"] == UNCERTAIN]
    source = load_from_disk(str(run_dir / "source"))
    final = Dataset.from_list(kept, features=load_from_disk(str(export_path.parent / "dataset")).features)
    summary = {"created_at": timestamp(), "run_id": manifest["run_id"], "source_generation": latest["generation"],
               "taxonomy_version": taxonomy.version, "export_sha256": file_hash(export_path),
               "manual_file": str(manual_path.resolve()), "manual_sha256": file_hash(manual_path),
               "removals_file": str(removals_path.resolve()), "removals_sha256": file_hash(removals_path),
               "input_rows": len(rows), "output_rows": len(final), "manual_assignments": applied,
               "removed": removed, "remaining_uncertain_row_indices": remaining,
               "note": "annotation_status 'manual' marks rows whose subject was assigned by a human after the run; "
                       "subject_confidence is null for them. Removed rows are absent from the source dataset revision."}
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        final.save_to_disk(str(stage / "huggingface"), num_shards=1)
        saved = load_from_disk(str(stage / "huggingface"))
        if len(saved) != len(source) - len(removed) or any(r["subject"] == UNCERTAIN for r in saved) != bool(remaining):
            raise ValueError("Final dataset count or abstention state differs from the expectation")
        with (stage / "data.jsonl").open("w", encoding="utf-8") as handle:
            for row in saved:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        for name, items in (("manual_assignments.jsonl", applied), ("removals.jsonl", removed)):
            with (stage / name).open("w", encoding="utf-8") as handle:
                for item in items:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        summary["output_sha256"] = {p.name: file_hash(p) for p in sorted((stage / "huggingface").iterdir()) if p.is_file()}
        write_json(stage / "manifest.json", summary)
        new_destination(output_dir, (run_dir / "exports",))
        os.rename(stage, output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return summary
