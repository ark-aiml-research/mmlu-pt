"""Validação, checkpoints e publicação de um resultado completo."""

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
from datasets import Dataset, DatasetDict, Value, load_from_disk

from mmlu_pt.classification.config import LABEL_COLUMNS
from mmlu_pt.classification.taxonomy import LEVELS, macro_area

REQUIRED_COLUMNS = ("question", "choices", "academic_level")
SINGLE_SPLIT = "__dataset__"


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_splits(dataset: Dataset | DatasetDict) -> dict[str, Dataset]:
    if isinstance(dataset, Dataset):
        return {SINGLE_SPLIT: dataset}
    if isinstance(dataset, DatasetDict):
        return dict(dataset)
    raise TypeError("dataset deve ser Dataset ou DatasetDict (não streaming).")


def arrow_row_hash(table: pa.Table) -> str:
    sink = pa.BufferOutputStream()
    table = table.combine_chunks()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue()).hexdigest()


def inspect_input(dataset: Dataset | DatasetDict) -> tuple[list[dict], dict]:
    records = []
    metadata = {"type": type(dataset).__name__, "splits": []}
    for split, data in as_splits(dataset).items():
        missing = set(REQUIRED_COLUMNS).difference(data.column_names)
        if len(data) and missing:
            raise ValueError(f"Split {split!r}: colunas obrigatórias ausentes: {sorted(missing)}.")
        digest = hashlib.sha256()
        arrow_data = data.with_format("arrow")
        rows = data.select_columns(list(REQUIRED_COLUMNS)).with_format(None) if len(data) else []
        for index, row in enumerate(rows):
            location = f"split={split!r}, index={index}"
            if not isinstance(row["question"], str) or not row["question"].strip():
                raise ValueError(f"{location}: question deve ser texto não vazio.")
            choices = row["choices"]
            if not isinstance(choices, list) or not choices or any(
                not isinstance(choice, str) or not choice.strip() for choice in choices
            ):
                raise ValueError(f"{location}: choices deve ser lista não vazia de textos não vazios.")
            if row["academic_level"] not in LEVELS:
                raise ValueError(f"{location}: academic_level inválido: {row['academic_level']!r}.")
            row_hash = arrow_row_hash(arrow_data[index])
            digest.update(bytes.fromhex(row_hash))
            records.append({"split": split, "index": index, "row_hash": row_hash, **row})
        metadata["splits"].append({
            "name": split, "count": len(data), "features": data.features.to_dict(),
            "records_sha256": digest.hexdigest(),
        })
    return records, metadata


def start_checkpoint(output_dir: Path, identity: dict, *, resume: bool) -> Path:
    root = output_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    pointer = root / "latest.json"
    if resume:
        if not pointer.is_file():
            raise ValueError("Não existe checkpoint para --resume neste output-dir.")
        name = json.loads(pointer.read_text(encoding="utf-8"))["run"]
        if Path(name).name != name or not name.startswith("run-"):
            raise ValueError("Identificador inválido no checkpoint.")
        checkpoint = root / name
        saved = json.loads((checkpoint / "identity.json").read_text(encoding="utf-8"))
        if saved != identity:
            differences = [key for key in set(saved) | set(identity) if saved.get(key) != identity.get(key)]
            raise ValueError(f"Checkpoint incompatível; campos alterados: {', '.join(sorted(differences))}.")
        return checkpoint
    checkpoint = root / f"run-{uuid.uuid4().hex}"
    checkpoint.mkdir()
    write_json(checkpoint / "identity.json", identity)
    write_json(checkpoint / "started.json", {"started_at": utc_now()})
    write_json(pointer, {"run": checkpoint.name})
    return checkpoint


def read_predictions(checkpoint: Path, records: list[dict]) -> dict[tuple[str, int], dict]:
    expected = {(row["split"], row["index"]): row for row in records}
    predictions = {}
    for path in sorted(checkpoint.glob("rank-*.jsonl")):
        with path.open("r+b") as handle:
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    # Só a última escrita interrompida pode ser descartada.
                    handle.truncate(position)
                    break
                try:
                    prediction = json.loads(line)
                    key = (prediction["split"], prediction["index"])
                    row = expected[key]
                    if prediction["row_hash"] != row["row_hash"]:
                        raise ValueError("hash de registro divergente")
                    if prediction["macro_area"] != macro_area(row["academic_level"], prediction["subject"]):
                        raise ValueError("macroárea divergente")
                    if key in predictions:
                        raise ValueError("registro duplicado")
                    if type(prediction["prompt_tokens"]) is not int or prediction["prompt_tokens"] < 1:
                        raise ValueError("contagem de tokens inválida")
                    predictions[key] = prediction
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"Checkpoint inválido em {path.name}, byte {position}: {error}") from error
    return predictions


def append_predictions(path: Path, predictions: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(canonical_json(prediction) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def add_labels(dataset: Dataset | DatasetDict, predictions: dict) -> Dataset | DatasetDict:
    outputs = {}
    for split, data in as_splits(dataset).items():
        column_order = data.column_names + [name for name in LABEL_COLUMNS if name not in data.column_names]
        output = data.remove_columns([name for name in LABEL_COLUMNS if name in data.column_names])
        for name in LABEL_COLUMNS:
            values = [predictions[(split, index)][name] for index in range(len(data))]
            output = output.add_column(name, values, feature=Value("string"))
        outputs[split] = output.select_columns(column_order)
    return DatasetDict(outputs) if isinstance(dataset, DatasetDict) else outputs[SINGLE_SPLIT]


def summarize(dataset: Dataset | DatasetDict) -> dict:
    summary = {"total": 0, "splits": {}, "by_level": {}, "subjects": {}, "macro_areas": {}, "by_exam": {}}
    subjects, areas = Counter(), Counter()
    for split, data in as_splits(dataset).items():
        summary["splits"][split] = len(data)
        fields = [name for name in ("academic_level", "subject", "macro_area", "exam") if name in data.column_names]
        for row in data.select_columns(fields).with_format(None):
            summary["total"] += 1
            subject, area = row["subject"], row["macro_area"]
            subjects[subject] += 1
            areas[area] += 1
            groups = [(summary["by_level"], row["academic_level"])]
            if "exam" in row:
                groups.append((summary["by_exam"], str(row["exam"])))
            for mapping, key in groups:
                group = mapping.setdefault(key, {"count": 0, "subjects": {}, "macro_areas": {}, "academic_levels": {}})
                group["count"] += 1
                for category, value in (("subjects", subject), ("macro_areas", area), ("academic_levels", row["academic_level"])):
                    group[category][value] = group[category].get(value, 0) + 1
    summary["subjects"], summary["macro_areas"] = dict(sorted(subjects.items())), dict(sorted(areas.items()))
    return summary


def validate_export(original: Dataset | DatasetDict, saved: Dataset | DatasetDict) -> None:
    if type(original) is not type(saved) or list(as_splits(original)) != list(as_splits(saved)):
        raise ValueError("Tipo ou ordem dos splits alterados na exportação.")
    for split, data in as_splits(original).items():
        output = as_splits(saved)[split]
        if len(data) != len(output):
            raise ValueError(f"Quantidade de linhas alterada no split {split!r}.")
        column_order = data.column_names + [name for name in LABEL_COLUMNS if name not in data.column_names]
        if output.column_names != column_order:
            raise ValueError(f"Ordem de colunas alterada no split {split!r}.")
        fields = [name for name in data.column_names if name not in LABEL_COLUMNS]
        for name in fields:
            if data.features[name] != output.features[name]:
                raise ValueError(f"Tipo de {name!r} alterado no split {split!r}.")
        if fields:
            before = data.select_columns(fields).with_format("arrow")[:]
            after = output.select_columns(fields).with_format("arrow")[:]
            if not before.equals(after, check_metadata=False):
                # Arrow considera NaN != NaN; a serialização também permite validar esses valores.
                if arrow_row_hash(before.replace_schema_metadata(None)) != arrow_row_hash(after.replace_schema_metadata(None)):
                    raise ValueError(f"Conteúdo ou ordem alterado no split {split!r}.")
        if any(output.features[name] != Value("string") for name in LABEL_COLUMNS):
            raise ValueError("Colunas de classificação devem ter tipo string.")
        fields = ["academic_level", *LABEL_COLUMNS] if len(output) else []
        for index, row in enumerate(output.select_columns(fields).with_format(None)):
            if row["macro_area"] != macro_area(row["academic_level"], row["subject"]):
                raise ValueError(f"Classificação inválida no split {split!r}, index={index}.")


def publish(output_dir: Path, original: Dataset | DatasetDict, classified: Dataset | DatasetDict, manifest: dict, summary: dict) -> Path | None:
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-staging-", dir=output_dir.parent))
    backup = None
    try:
        if isinstance(classified, DatasetDict):
            shards = {name: 1 if not len(data) else None for name, data in classified.items()}
        else:
            shards = 1 if not len(classified) else None
        # datasets 5.0.1 grava zero shards por padrão em entradas vazias, que não recarregam.
        classified.save_to_disk(str(staging / "dataset"), num_shards=shards)
        validate_export(original, load_from_disk(str(staging / "dataset")))
        shutil.copytree(output_dir / "checkpoints", staging / "checkpoints")
        write_json(staging / "checkpoints" / manifest["checkpoint"] / "status.json",
                   {"status": "complete", "updated_at": manifest["completed_at"]})
        write_json(staging / "manifest.json", manifest)
        write_json(staging / "summary.json", summary)
        backup = output_dir.with_name(f".{output_dir.name}-backup-{uuid.uuid4().hex}")
        os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except BaseException:
            os.replace(backup, output_dir)
            raise
        return backup
    finally:
        if staging.exists():
            shutil.rmtree(staging)
