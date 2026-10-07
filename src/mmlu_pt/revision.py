"""Remove questions with unrecoverable defects from a local dataset, with an audit trail."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset, load_from_disk
from huggingface_hub import HfApi

from mmlu_pt.annotation.knowledge_area.dataset import annotation_id

DEFAULT_REMOVALS = Path("config/removed_questions.json")
IDENTITY_FIELDS = ("exam", "exam_edition", "num")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def directory_sha256(path: Path) -> dict[str, str]:
    return {entry.name: file_sha256(entry) for entry in sorted(path.iterdir()) if entry.is_file()}


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_removals(path: Path) -> dict[str, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    removals = {}
    for entry in document["removals"]:
        if entry["annotation_id"] in removals:
            raise ValueError(f"Duplicate annotation_id in removal list: {entry['annotation_id']}")
        removals[entry["annotation_id"]] = entry
    if not removals:
        raise ValueError("Removal list is empty")
    return removals


def new_destination(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"Output already exists; choose a new directory: {path}")


def split_rows(data: Dataset, removals: dict[str, dict]) -> tuple[list[int], list[dict]]:
    """Return kept positions and the audit record of every removed row, in input order."""
    kept, removed = [], []
    for position, row in enumerate(data):
        identity = annotation_id(row)
        entry = removals.get(identity)
        if entry is None:
            kept.append(position)
            continue
        if any(row[field] != entry[field] for field in IDENTITY_FIELDS):
            raise ValueError(f"Removal metadata differs from the dataset at input row {position}")
        removed.append({"input_row_index": position, "annotation_id": identity,
                        **{field: row[field] for field in IDENTITY_FIELDS},
                        "category": entry["category"], "reason": entry["reason"]})
    return kept, removed


def check_output(saved: Dataset, data: Dataset, kept: list[int], removals: dict[str, dict]) -> None:
    if len(saved) != len(kept) or saved.features != data.features:
        raise ValueError("Revised dataset count or schema differs from the expectation")
    for position, row in zip(kept, saved, strict=True):
        if annotation_id(row) in removals or row != data[position]:
            raise ValueError(f"Revised dataset diverges from its input at input row {position}")


def revise(input_dir: Path, output_dir: Path, removals_path: Path) -> dict:
    new_destination(output_dir)
    removals = read_removals(removals_path)
    data = load_from_disk(str(input_dir))
    if not isinstance(data, Dataset):
        raise ValueError("Expected a single Hugging Face Dataset, not a DatasetDict")
    kept, removed = split_rows(data, removals)
    found = {row["annotation_id"] for row in removed}
    # flatten_indices keeps the Arrow cache of the selection out of the input directory.
    revised = data.select(kept).flatten_indices(keep_in_memory=True)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "input_dir": str(input_dir.resolve()),
                "input_rows": len(data), "input_sha256": directory_sha256(input_dir),
                "removals_file": str(removals_path.resolve()), "removals_sha256": file_sha256(removals_path),
                "removed_rows": len(removed), "removed": removed,
                "listed_ids_absent_from_input": sorted(removals.keys() - found),
                "output_rows": len(revised), "columns": revised.column_names}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        revised.save_to_disk(str(stage / "huggingface"), num_shards=1)
        saved = load_from_disk(str(stage / "huggingface"))
        check_output(saved, data, kept, removals)
        write_jsonl(stage / "data.jsonl", saved)
        write_jsonl(stage / "removals.jsonl", removed)
        manifest["output_sha256"] = directory_sha256(stage / "huggingface")
        manifest["data_jsonl_sha256"] = file_sha256(stage / "data.jsonl")
        write_manifest(stage / "manifest.json", manifest)
        new_destination(output_dir)
        os.rename(stage, output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def publish(output_dir: Path, repository: str) -> dict:
    """Push a revised dataset to the Hub and record the resulting revision in its manifest."""
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if directory_sha256(output_dir / "huggingface") != manifest["output_sha256"]:
        raise ValueError("Revised dataset changed after its manifest was written")
    data = load_from_disk(str(output_dir / "huggingface"))
    commit = data.push_to_hub(repository, split="train")
    revision = HfApi().dataset_info(repository).sha
    manifest["hub"] = {"repository": repository, "commit": commit.oid, "revision": revision,
                       "pushed_at": datetime.now(timezone.utc).isoformat()}
    write_manifest(output_dir / "manifest.json", manifest)
    return manifest["hub"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove listed questions from a local Hugging Face dataset")
    commands = parser.add_subparsers(dest="command", required=True)
    revise_command = commands.add_parser("revise", help="Write a revised copy of a dataset; no network access")
    revise_command.add_argument("--input", type=Path, required=True, help="directory saved with save_to_disk")
    revise_command.add_argument("--output", type=Path, required=True, help="new directory; existing ones are refused")
    revise_command.add_argument("--removals", type=Path, default=DEFAULT_REMOVALS)
    publish_command = commands.add_parser("publish", help="Push a revised dataset to the Hub as split 'train'")
    publish_command.add_argument("--output", type=Path, required=True, help="directory written by 'revise'")
    publish_command.add_argument("--repo", required=True, help="dataset repository, e.g. org/name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "publish":
            hub = publish(args.output, args.repo)
            print(f"Pushed to {hub['repository']} at revision {hub['revision']}")
            return 0
        manifest = revise(args.input, args.output, args.removals)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 2
    absent = len(manifest["listed_ids_absent_from_input"])
    print(f"Removed {manifest['removed_rows']} of {manifest['input_rows']} rows; "
          f"{manifest['output_rows']} kept; {absent} listed IDs absent from input: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
