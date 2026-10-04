"""Generate reproducible TS-Guessing results with offline vLLM batches."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import unicodedata
import uuid
from tempfile import TemporaryDirectory
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from artifacts import (
    ARTIFACT_VERSION, BACKEND, TERMINAL_STATUSES, TRANSFORMERS_VERSION, VLLM_VERSION,
    digest, file_hash, latest_responses, read_json, read_jsonl, record_operation, response_summary,
    utc_now, write_json, write_jsonl,
)


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
DEFAULT_CONFIG = STUDY_DIR / "config.json"
WORKER_PATH = STUDY_DIR / "offline_worker.py"
MODEL_PLAN = REPO_ROOT / "docs/experimental_evaluation.md"
# Effective PLE capacity from the model-plan footnote; these are not MoE counts.
EFFECTIVE_PLE_PARAMETERS_B = {"google/gemma-4-E2B-it": 2.3, "google/gemma-4-E4B-it": 4.5}
HTTP_OPTIONS = {"port", "host", "request_timeout_seconds", "startup_timeout_seconds"}
CONDITIONS = ("without_source", "with_source")
PROTOCOL_VERSION = 1
INSTRUCTION = (
    "Reconstrua exatamente o texto da alternativa {label} substituído por [MASK]. "
    "Não copie as demais alternativas. Responda apenas com o texto ausente, "
    "sem letra ou explicação."
)


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def runtime_versions() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("vllm", "torch", "transformers", "datasets", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def serving_command_prefix(server: dict) -> list[str]:
    executable = shutil.which("uv")
    if not executable:
        raise FileNotFoundError("uv is required to prepare the isolated vLLM runtime.")
    return [executable, "run", "--isolated", "--no-project", "--python", sys.executable,
            "--with", f"vllm=={server['vllm_version']}",
            "--with", f"transformers=={server['transformers_version']}",
            "--with", "tqdm>=4.67,<5"]


def serving_runtime_versions(server: dict) -> dict:
    """Prepare the isolated inference environment when generation is requested."""
    program = (
        "import importlib.metadata,json,platform; "
        "print(json.dumps({'python':platform.python_version(), **{"
        "p:importlib.metadata.version(p) for p in "
        "('vllm','torch','transformers','huggingface-hub','tqdm')}}))"
    )
    output = subprocess.check_output([*serving_command_prefix(server), "python", "-c", program], text=True)
    return json.loads(output)


def load_input(spec: dict) -> list[dict]:
    from datasets import DatasetDict, load_dataset, load_from_disk

    if spec["kind"] == "huggingface":
        return list(load_dataset(
            spec["dataset"], name=spec["config"], split=spec["split"],
            revision=spec["revision"],
        ))

    path = resolve_path(spec["path"])
    if path.is_dir() and any((path / name).exists() for name in ("state.json", "dataset_dict.json")):
        dataset = load_from_disk(str(path))
        return list(dataset[spec["split"]] if isinstance(dataset, DatasetDict) else dataset)

    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if path.is_dir() and not files:
        files = sorted(path.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No Parquet or JSONL files found in {path}")
    if files[0].suffix == ".parquet":
        return list(load_dataset("parquet", data_files=[str(file) for file in files], split="train"))
    rows = []
    for file in files:
        with file.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def identify_records(rows: list[dict]) -> tuple[list[dict], str]:
    occurrences = Counter()
    identified = []
    for source_row_index, row in enumerate(rows):
        content_id = digest(row)
        occurrence = occurrences[content_id]
        occurrences[content_id] += 1
        identified.append({**row, "question_id": f"{content_id}:{occurrence}",
                           "source_row_index": source_row_index})
    return identified, digest(sorted(row["question_id"] for row in identified))


def sample_records(rows: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["exam"]].append(row)
    fraction = Fraction(str(fraction))
    sample, counts = [], []
    for exam, group in sorted(groups.items()):
        size = (len(group) * fraction.numerator + fraction.denominator - 1) // fraction.denominator
        ordered = sorted(group, key=lambda row: digest([seed, exam, row["question_id"]]))
        sample.extend(ordered[:size])
        counts.append({"exam": exam, "population": len(group), "sample": size})
    return sample, counts


def strip_choice_label(text: str, label: str) -> str:
    """Strip one editorial label only when it agrees with the option position."""
    pattern = rf"^\s*(?:\({label}\)|\[{label}\]|{label}\s*[.):\-])\s+"
    return re.sub(pattern, "", text, count=1).strip()


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def build_tasks(sample: list[dict]) -> list[dict]:
    tasks = []
    for row in sample:
        choices = [strip_choice_label(text, chr(65 + index)) for index, text in enumerate(row["choices"])]
        for index, target in enumerate(choices):
            if index == row["answer"]:
                continue
            label = chr(65 + index)
            visible = [choice for other, choice in enumerate(choices) if other != index]
            options = "\n".join(
                f"{chr(65 + other)}: {'[MASK]' if other == index else choice}"
                for other, choice in enumerate(choices)
            )
            instruction = INSTRUCTION.format(label=label)
            body = f"{instruction}\n\nEnunciado:\n{row['question']}\n\nAlternativas:\n{options}"
            normalized = normalize_text(target)
            for condition in CONDITIONS:
                source = (
                    f"Prova: {row['exam']}\nEdição: {row['exam_edition']}\n"
                    f"Número da questão: {row['num']}\n\n"
                    if condition == "with_source" else ""
                )
                prompt = source + body
                tasks.append({
                    "task_id": digest([row["question_id"], index, condition, prompt]),
                    "question_id": row["question_id"], "condition": condition,
                    "exam": row["exam"], "exam_edition": row["exam_edition"],
                    "num": row["num"], "subject": row["subject"],
                    "macro_area": row["macro_area"], "academic_level": row["academic_level"],
                    "mask_index": index, "mask_label": label, "choice_count": len(choices),
                    "target": target, "target_words": len(target.split()),
                    "target_numeric": bool(re.fullmatch(r"[\d\s.,+−\-*/=()%²³]+", target)) and bool(re.search(r"\d", target)),
                    "target_in_context": bool(normalized) and any(
                        normalized in normalize_text(text) for text in [row["question"], *visible]
                    ),
                    "visible_choices": visible, "prompt": prompt,
                })
    return tasks


def model_descriptions(models: list[dict]) -> dict:
    """Snapshot analysis metadata from the model plan, outside inference settings."""
    descriptions = {}
    for line in MODEL_PLAN.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 9 or cells[6] != "Open-weight":
            continue
        repo = cells[1].strip("`")
        nominal = float(re.match(r"[\d,]+", cells[2])[0].replace(",", "."))
        active = float(cells[3].replace(",", ".")) if cells[3] != "—" else None
        descriptions[repo] = {
            "family": cells[0], "nominal_parameters_b": nominal,
            "active_parameters_b": active,
            "effective_parameters_b": EFFECTIVE_PLE_PARAMETERS_B.get(repo, active),
            "parameters_description_source": "docs/experimental_evaluation.md, section 2.1",
        }
    return {spec["name"]: descriptions.get(spec["repo"], {}) for spec in models}


def prepare_run(config: dict, versions: dict) -> tuple[Path, dict, list[dict]]:
    descriptions = model_descriptions(config["models"])
    rows, snapshot_hash = identify_records(load_input(config["input"]))
    source_order_hash = digest([row["question_id"] for row in rows])
    sample, counts = sample_records(rows, **config["sampling"])
    tasks = build_tasks(sample)
    experiment = {key: value for key, value in config.items() if key != "output_dir"}
    implementation = {path.name: file_hash(path)
                      for path in (Path(__file__), WORKER_PATH, STUDY_DIR / "artifacts.py")}
    identity = digest({
        "config": experiment, "snapshot_hash": snapshot_hash, "versions": versions,
        "source_order_sha256": source_order_hash, "artifact_version": ARTIFACT_VERSION,
        "model_descriptions": descriptions,
        "implementation_sha256": implementation,
        "backend": BACKEND,
        "protocol_version": PROTOCOL_VERSION, "instruction": INSTRUCTION,
        "task_ids": [task["task_id"] for task in tasks],
    })
    run_dir = resolve_path(config["output_dir"]) / identity[:20]
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError(f"Experiment identity mismatch in {run_dir}")
        for filename, field in (("sample.jsonl", "sample_sha256"), ("tasks.jsonl", "tasks_sha256")):
            if file_hash(run_dir / filename) != manifest[field]:
                raise ValueError(f"Artifact checksum mismatch: {run_dir / filename}")
        return run_dir, manifest, tasks
    manifest = {
        "identity": identity, "created_at": utc_now(), "status": "prepared",
        "artifact_version": ARTIFACT_VERSION, "source_order_sha256": source_order_hash,
        "implementation_sha256": implementation,
        "protocol_version": PROTOCOL_VERSION, "instruction": INSTRUCTION,
        "backend": BACKEND, "response_format": "vllm_request_output_v1",
        "config": config, "versions": versions, "dataset": config["input"],
        "model_descriptions": descriptions,
        "snapshot_hash": snapshot_hash, "population_size": len(rows),
        "sample_size": len(sample), "exam_counts": counts,
        "tasks_per_model": len(tasks), "expected_requests": len(tasks) * len(config["models"]),
        "conditions": list(CONDITIONS), "models": {},
    }
    write_jsonl(run_dir / "sample.jsonl", sample)
    write_jsonl(run_dir / "tasks.jsonl", tasks)
    manifest["sample_sha256"] = file_hash(run_dir / "sample.jsonl")
    manifest["tasks_sha256"] = file_hash(run_dir / "tasks.jsonl")
    write_json(manifest_path, manifest)
    return run_dir, manifest, tasks


def pin_model(model_dir: Path, spec: dict, description: dict, run_identity: str, settings: dict) -> None:
    """Resolve a checkpoint once; model metadata is the authoritative revision pin."""
    from huggingface_hub import HfApi

    path = model_dir / "metadata.json"
    if path.exists():
        metadata = read_json(path)
        if metadata["run_identity"] != run_identity or metadata["spec"] != spec:
            raise ValueError(f"Incompatible model cache: {model_dir}")
        return
    info = HfApi().model_info(spec["repo"], revision=spec.get("revision"))
    checkpoint = {"repo": spec["repo"], "revision": info.sha, "tokenizer_revision": info.sha,
                  "total_parameters": getattr(info.safetensors, "total", None), **description,
                  "resolved_at": utc_now()}
    write_json(path, {"run_identity": run_identity, "spec": spec, "checkpoint": checkpoint,
                      "server": settings, "artifact_version": ARTIFACT_VERSION})


def check_offline_config(config: dict) -> None:
    settings = config["server"]
    for owner, values in [("server", settings), *[(spec["name"], spec) for spec in config["models"]]]:
        obsolete = HTTP_OPTIONS.intersection(values)
        if obsolete:
            raise ValueError(
                f"{owner}: HTTP-only options {sorted(obsolete)} are unsupported by offline inference. "
                "Remove them; use initialization_timeout_seconds for engine startup."
            )
    if settings.get("backend") != BACKEND:
        raise ValueError("server.backend must be 'offline'; this generator uses LLM.generate(), not HTTP.")
    if settings["vllm_version"] != VLLM_VERSION:
        raise ValueError(f"Offline inference requires vllm=={VLLM_VERSION} in the isolated runtime.")
    if settings["transformers_version"] != TRANSFORMERS_VERSION:
        raise ValueError(f"Offline inference requires transformers=={TRANSFORMERS_VERSION} for Gemma 4 compatibility.")
    for spec in config["models"]:
        model_server_config(spec, settings)
    for key in ("initialization_timeout_seconds", "batch_timeout_seconds", "shutdown_timeout_seconds"):
        if settings[key] <= 0:
            raise ValueError(f"server.{key} must be positive.")
    if type(settings["max_retries"]) is not int or not 0 <= settings["max_retries"] <= 3:
        raise ValueError("server.max_retries must be an integer between zero and three.")


def model_server_config(spec: dict, server: dict) -> dict:
    """Resolve per-model parallelism, batching and context limits."""
    settings = server.copy()
    for key in ("data_parallel_size", "tensor_parallel_size", "concurrency", "batch_size", "max_model_len"):
        settings[key] = spec.get(key, server[key])
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer for {spec['name']}")
    return settings


def runtime_environment(spec: dict, server: dict) -> dict:
    environment = os.environ.copy()
    gpu_count = server["data_parallel_size"] * server["tensor_parallel_size"]
    visible = spec.get("cuda_visible_devices", environment.get("CUDA_VISIBLE_DEVICES"))
    if visible is None:
        visible = ",".join(str(index) for index in range(gpu_count))
    devices = [device.strip() for device in visible.split(",") if device.strip()]
    if "-1" in devices or len(set(devices)) < gpu_count:
        raise ValueError(
            f"{spec['name']} requires {gpu_count} visible GPUs "
            f"(DP={server['data_parallel_size']}, TP={server['tensor_parallel_size']}); "
            f"CUDA_VISIBLE_DEVICES={visible!r}"
        )
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:gpu_count])
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return environment


def stop_runtime(process: subprocess.Popen, timeout: float) -> None:
    """Reap the isolated runtime and its own process group, including GPU workers."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    # Workers can outlive their coordinator; the group belongs to this launch only.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_model(run_dir: Path, spec: dict, config: dict, tasks: list[dict], manifest: dict) -> None:
    settings = model_server_config(spec, config["server"])
    model_dir = run_dir / "models" / spec["name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    pin_model(model_dir, spec, manifest["model_descriptions"][spec["name"]], manifest["identity"], settings)
    latest = latest_responses(model_dir)
    read_jsonl(model_dir / "operations.jsonl", repair=True)
    if all(latest.get(task["task_id"], {}).get("status") in TERMINAL_STATUSES for task in tasks):
        manifest["models"][spec["name"]] = response_summary(latest.values(), len(tasks))
        return
    before_ids = {key for key, row in latest.items() if row["status"] == "completed"}
    session = {"session_id": uuid.uuid4().hex, "started_at": utc_now(), "model": spec["name"],
               "backend": BACKEND, "status": "running", "attempts": 0,
               "gpu_count": settings["data_parallel_size"] * settings["tensor_parallel_size"]}
    started = time.perf_counter()
    try:
        environment = runtime_environment(spec, settings)
        with TemporaryDirectory(prefix=".runtime-", dir=model_dir) as temporary:
            for attempt in range(1, settings["max_retries"] + 2):
                session["attempts"] = attempt
                result_path = Path(temporary) / f"attempt-{attempt}.json"
                command = [*serving_command_prefix(settings), "python", str(WORKER_PATH),
                           "--run-dir", str(run_dir), "--model-dir", str(model_dir),
                           "--session-id", session["session_id"], "--attempt", str(attempt),
                           "--result-path", str(result_path)]
                record_operation(model_dir, "launch", session_id=session["session_id"], attempt=attempt,
                                 command=command, cuda_visible_devices=environment["CUDA_VISIBLE_DEVICES"])
                process = subprocess.Popen(command, env=environment, start_new_session=True)
                try:
                    return_code = process.wait()
                finally:
                    stop_runtime(process, settings["shutdown_timeout_seconds"])
                    outcome = read_json(result_path) if result_path.exists() else {
                        "status": "failed", "retryable": False,
                        "error": "Offline coordinator exited without a result; see rank logs.",
                    }
                if return_code == 0 and outcome["status"] == "finished":
                    session["status"] = "finished"
                    break
                if not outcome.get("retryable") or attempt > settings["max_retries"]:
                    raise RuntimeError(outcome.get("error", f"Offline runtime exited with code {return_code}"))
                print(f"{spec['name']}: transient failure; restarting all ranks (retry {attempt}).", flush=True)
                time.sleep(2 ** (attempt - 1))
    except BaseException as error:
        session.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        latest = latest_responses(model_dir)
        new_rows = [row for key, row in latest.items() if row["status"] == "completed" and key not in before_ids]
        session.update(response_summary(new_rows), wall_seconds=time.perf_counter() - started, ended_at=utc_now())
        record_operation(model_dir, "session", **session)
        manifest["models"][spec["name"]] = response_summary(latest.values(), len(tasks))


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=str, help="Explicit local annotated Dataset/Parquet/JSONL path.")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--models", nargs="+", help="Model names from config; omitted means all models.")
    parser.add_argument("--data-parallel-size", type=positive_int,
                        help="Local vLLM DP size; overrides all configured model values.")
    parser.add_argument("--tensor-parallel-size", type=positive_int,
                        help="GPUs per DP rank; overrides all configured model values.")
    parser.add_argument("--concurrency", type=positive_int,
                        help="Maximum active sequences per DP rank (vLLM max_num_seqs).")
    parser.add_argument("--batch-size", type=positive_int,
                        help="Tasks submitted together per DP rank; default is 128.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = read_json(args.config)
    if args.input:
        config["input"] = {"kind": "local", "path": str(resolve_path(args.input)), "split": config["input"]["split"]}
    if args.output_dir:
        config["output_dir"] = args.output_dir
    for key in ("data_parallel_size", "tensor_parallel_size", "concurrency", "batch_size"):
        value = getattr(args, key)
        if value is not None:
            config["server"][key] = value
            for spec in config["models"]:
                if key in spec:
                    spec[key] = value
    check_offline_config(config)
    selected = args.models or [spec["name"] for spec in config["models"]]
    unknown = set(selected) - {spec["name"] for spec in config["models"]}
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    # Changing code, data or configuration produces a different run directory.
    runtime_start = time.perf_counter()
    versions = {"controller": runtime_versions(), "serving": serving_runtime_versions(config["server"])}
    runtime_seconds = time.perf_counter() - runtime_start
    run_dir, manifest, tasks = prepare_run(config, versions)
    print(f"Run directory: {run_dir}\nSample: {manifest['sample_size']} questions; "
          f"{len(tasks)} tasks/model; {manifest['expected_requests']} tasks in the full panel.", flush=True)
    with (run_dir / ".generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        read_jsonl(run_dir / "operations.jsonl", repair=True)
        record_operation(run_dir, "runtime_preparation", **{
            "at": utc_now(), "elapsed_seconds": runtime_seconds, "versions": versions["serving"],
        })
        manifest["status"] = "running"
        write_json(run_dir / "manifest.json", manifest)
        try:
            for spec in config["models"]:
                if spec["name"] not in selected:
                    continue
                print(f"Starting/resuming {spec['name']}", flush=True)
                try:
                    run_model(run_dir, spec, config, tasks, manifest)
                except Exception as error:
                    record_operation(run_dir / "models" / spec["name"], "error", stage="model", error=str(error))
                    previous = manifest["models"].get(spec["name"], {})
                    manifest["models"][spec["name"]] = {**previous, "status": "model_error", "error": str(error), "updated_at": utc_now()}
                    print(f"{spec['name']}: {error}", file=sys.stderr, flush=True)
                write_json(run_dir / "manifest.json", manifest)
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            write_json(run_dir / "manifest.json", manifest)
            return 130
        manifest["status"] = "completed" if all(
            manifest["models"].get(spec["name"], {}).get("status") == "completed" for spec in config["models"]
        ) else "partial"
        manifest["updated_at"] = utc_now()
        write_json(run_dir / "manifest.json", manifest)
    print(f"Generation status: {manifest['status']}; artifacts: {run_dir}", flush=True)
    return 0 if all(manifest["models"].get(name, {}).get("status") == "completed" for name in selected) else 1


if __name__ == "__main__":
    def interrupt_on_term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_on_term)
    raise SystemExit(main())
