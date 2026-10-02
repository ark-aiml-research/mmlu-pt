"""API pública de classificação e coordenação da persistência."""

import fcntl
import hashlib
import json
import logging
import platform
import signal
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path
from threading import current_thread, main_thread

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from mmlu_pt.classification.config import MAX_OUTPUT_TOKENS, MODEL_ID, ClassificationConfig
from mmlu_pt.classification.inference import check_hardware, run_inference
from mmlu_pt.classification.persistence import (
    add_labels, as_splits, inspect_input, json_hash, publish, read_predictions,
    start_checkpoint, summarize, utc_now, write_json,
)
from mmlu_pt.classification.taxonomy import LEVELS, load_taxonomy, resource_hash, system_prompt

LOGGER = logging.getLogger(__name__)
OUTPUT_ENTRIES = {"dataset", "manifest.json", "summary.json", "checkpoints"}


def interrupted(signum, frame) -> None:
    raise KeyboardInterrupt(f"Sinal {signum}; checkpoints preservados.")


def library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("mmlu-pt", "datasets", "pyarrow", "huggingface-hub", "vllm", "torch", "transformers"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def implementation_hash() -> str:
    digest = hashlib.sha256()
    for name in ("config.py", "taxonomy.py", "inference.py", "persistence.py", "runner.py", "cli.py"):
        digest.update(name.encode("utf-8"))
        digest.update(files("mmlu_pt.classification").joinpath(name).read_bytes())
    return digest.hexdigest()


def resolve_model_revision(requested: str) -> str:
    resolved = HfApi().model_info(MODEL_ID, revision=requested).sha
    if not resolved:
        raise RuntimeError("Não foi possível resolver a revisão dos pesos do modelo.")
    return resolved


def limit_dataset(dataset: Dataset | DatasetDict, limit: int | None) -> Dataset | DatasetDict:
    if limit is None:
        return dataset
    splits = {name: data.select(range(min(limit, len(data)))) for name, data in as_splits(dataset).items()}
    return DatasetDict(splits) if isinstance(dataset, DatasetDict) else next(iter(splits.values()))


def classify_dataset(dataset: Dataset | DatasetDict, *, output_dir: str | Path,
                     config: ClassificationConfig) -> Dataset | DatasetDict:
    """Acrescenta subject/macro_area e publica um dataset local completo."""
    config.validate()
    load_taxonomy()
    dataset = limit_dataset(dataset, config.limit)
    records, input_metadata = inspect_input(dataset)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.with_name(f".{output_dir.name}.classification.lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Outra classificação está usando este output-dir.") from error
        install_handler = current_thread() is main_thread()
        previous = signal.signal(signal.SIGTERM, interrupted) if install_handler else None
        try:
            return _classify(dataset, records, input_metadata, output_dir=output_dir, config=config)
        finally:
            if install_handler:
                signal.signal(signal.SIGTERM, previous)


def _classify(dataset: Dataset | DatasetDict, records: list[dict], input_metadata: dict,
              *, output_dir: Path, config: ClassificationConfig) -> Dataset | DatasetDict:
    if output_dir.exists():
        unknown = {path.name for path in output_dir.iterdir()} - OUTPUT_ENTRIES
        if unknown:
            raise ValueError(f"output-dir contém arquivos alheios à classificação: {sorted(unknown)}.")
    output_dir.mkdir(exist_ok=True)
    devices = check_hardware(config) if records and not config.resume else []
    resolved_revision = resolve_model_revision(config.model_revision) if records else None
    identity = {
        "schema_version": 1, "input": input_metadata, "source": config.dataset_source,
        "model": {"id": MODEL_ID, "requested_revision": config.model_revision, "resolved_revision": resolved_revision},
        "parameters": config.identity_parameters(), "versions": library_versions(),
        "taxonomy": {"version": load_taxonomy()["version"], "sha256": resource_hash("taxonomy.json")},
        "prompt_sha256": resource_hash("prompt.md"), "implementation_sha256": implementation_hash(),
        "generation": {"temperature": 0, "max_tokens": MAX_OUTPUT_TOKENS, "enable_thinking": False,
                       "language_model_only": True, "structured_outputs": "choice", "enable_expert_parallel": False},
    }
    checkpoint = start_checkpoint(output_dir, identity, resume=config.resume)
    write_json(checkpoint / "status.json", {"status": "running", "updated_at": utc_now()})
    try:
        predictions = read_predictions(checkpoint, records)
        resumed_count = len(predictions)
        LOGGER.info("Entrada: %s questões; %s já classificadas em checkpoint", len(records), resumed_count)
        runtime = {"generated": 0, "rounds": 0}
        runtime_path = checkpoint / "runtime.json"
        if resumed_count < len(records):
            if not devices:
                devices = check_hardware(config)
            runtime = run_inference(
                records, set(predictions), checkpoint=checkpoint,
                model_revision=resolved_revision, config=config,
            )
            predictions = read_predictions(checkpoint, records)
        elif runtime_path.is_file():
            previous = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime, devices = previous["runtime"], previous["devices"]
        if len(predictions) != len(records):
            raise RuntimeError(f"Resultado incompleto: {len(predictions)}/{len(records)} registros.")
        classified = add_labels(dataset, predictions)
        summary = summarize(classified)
        started = json.loads((checkpoint / "started.json").read_text(encoding="utf-8"))["started_at"]
        manifest = {
            "status": "complete", "started_at": started, "completed_at": utc_now(),
            "identity": identity, "identity_sha256": json_hash(identity),
            "system_prompt_sha256": {level: hashlib.sha256(system_prompt(level).encode("utf-8")).hexdigest() for level in LEVELS},
            "counts": {"input": len(records), "classified": len(predictions), "reused": resumed_count,
                       "generated_this_invocation": len(records) - resumed_count,
                       "splits": summary["splits"]},
            "checkpoint_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted(checkpoint.glob("rank-*.jsonl"))},
            "runtime": runtime, "devices": devices, "checkpoint": checkpoint.name,
            "output": "dataset", "summary": "summary.json",
        }
        write_json(runtime_path, {"runtime": runtime, "devices": devices})
        write_json(checkpoint / "status.json", {"status": "ready_for_publication", "updated_at": utc_now()})
        backup = publish(output_dir, dataset, classified, manifest, summary)
        LOGGER.info("Dataset completo salvo em %s", output_dir / "dataset")
        LOGGER.info("Diretório anterior preservado em %s", backup)
        return classified
    except BaseException as error:
        write_json(checkpoint / "status.json", {"status": "failed", "updated_at": utc_now(),
                                               "error": f"{type(error).__name__}: {error}"})
        LOGGER.error("Execução interrompida; checkpoints preservados em %s", checkpoint)
        raise
