"""Bounded offline vLLM batches coordinated across local MoE DP ranks."""

import math
import multiprocessing as mp
import os
import queue
import signal
import socket
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import HfApi

from .config import RunConfig, timestamp, write_json
from .validation import FormatFailure, parse_annotation, response_schema

VLLM_VERSION = "0.25.0"
POLL_INTERVAL = 0.2


class InferenceError(RuntimeError):
    """Only sanitized engine, worker or protocol failure codes."""


@dataclass(frozen=True)
class OfflineConfig:
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    batch_size: int = 128
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 50
    max_num_batched_tokens: int = 4096
    enforce_eager: bool = False
    startup_timeout: float = 1800.0
    shutdown_timeout: float = 30.0
    batch_timeout: float = 0.0

    def __post_init__(self):
        for name in ("data_parallel_size", "tensor_parallel_size", "batch_size", "max_model_len",
                     "max_num_seqs", "max_num_batched_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        for name in ("startup_timeout", "shutdown_timeout", "batch_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != "batch_timeout" and value == 0):
                raise ValueError(f"Invalid {name}")

    def identity(self) -> dict:
        return {k: v for k, v in asdict(self).items()
                if k not in ("startup_timeout", "shutdown_timeout", "batch_timeout")}


def engine_arguments(config: RunConfig, settings: OfflineConfig, revision: str) -> dict:
    # Offline SPMD obtains DP rank/size from environment, not LLM constructor args.
    return {"model": config.model, "revision": revision, "tokenizer_revision": revision,
            "tensor_parallel_size": settings.tensor_parallel_size,
            "distributed_executor_backend": "mp", "enable_expert_parallel": False,
            "max_model_len": settings.max_model_len, "max_num_seqs": settings.max_num_seqs,
            "max_num_batched_tokens": settings.max_num_batched_tokens,
            "gpu_memory_utilization": settings.gpu_memory_utilization,
            "enforce_eager": settings.enforce_eager, "seed": config.seed,
            "language_model_only": True, "generation_config": "vllm",
            "enable_prefix_caching": False, "disable_log_stats": True,
            "structured_outputs_config": {"backend": "xgrammar", "reasoning_parser": "qwen3",
                                          "enable_in_reasoning": False}}


def prepare_inference(config: RunConfig, settings: OfflineConfig, previous: dict | None = None) -> dict:
    if version("vllm") != VLLM_VERSION:
        raise ValueError("Offline annotation requires vLLM 0.25.0: uv sync --locked --group annotation")
    if config.max_tokens >= settings.max_model_len:
        raise ValueError("max_tokens must leave room for the prompt within max_model_len")
    revision = ((previous or {}).get("inference_metadata") or {}).get("revision")
    if revision is None:
        revision = HfApi().model_info(config.model, revision=config.model_revision).sha
    if not isinstance(revision, str) or not revision:
        raise ValueError("Could not resolve an immutable model revision")
    return {"backend": "vllm_offline", "model": config.model, "revision": revision,
            "vllm_version": VLLM_VERSION, "settings": settings.identity(),
            "engine_arguments": engine_arguments(config, settings, revision)}


def check_hardware(settings: OfflineConfig) -> list[dict]:
    import torch

    required = settings.data_parallel_size * settings.tensor_parallel_size
    available = torch.cuda.device_count()
    if not torch.cuda.is_available() or available < required:
        raise ValueError(f"DP × TP requires {required} visible CUDA GPUs; available: {available}")
    return [{"device": i, "name": torch.cuda.get_device_properties(i).name,
             "total_bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(required)]


def sampling_arguments(config: RunConfig, item: dict) -> dict:
    values = config.generation() | {"seed": item["seed"]}
    if config.structured_output == "json_schema":
        from vllm.sampling_params import StructuredOutputsParams

        values["structured_outputs"] = StructuredOutputsParams(json=response_schema(item["candidates"]))
    return values


def generate_batch(llm, config: RunConfig, settings: OfflineConfig, items: list[dict]) -> list[dict]:
    """Return validated results and token counts, never completion text."""
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    inputs, sampling, active, results = [], [], [], []
    for item in items:
        token_ids = tokenizer.apply_chat_template(item["messages"], tokenize=True,
                                                 add_generation_prompt=True, enable_thinking=config.thinking,
                                                 truncation=False)
        if len(token_ids) + config.max_tokens > settings.max_model_len:
            results.append({"annotation_id": item["annotation_id"], "attempt": item["attempt"],
                            "result": None, "error": "context_overflow",
                            "metadata": {"prompt_tokens": len(token_ids)}})
            continue
        inputs.append({"prompt_token_ids": token_ids})
        sampling.append(SamplingParams(**sampling_arguments(config, item)))
        active.append(item)
    if not active:
        # Empty ranks must enter generate too: expert-layer collectives span all DP ranks.
        token_ids = tokenizer.apply_chat_template([{"role": "user", "content": "Reply briefly."}],
                                                 tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False, truncation=False)
        inputs = [{"prompt_token_ids": token_ids}]
        sampling = [SamplingParams(temperature=0, max_tokens=1, seed=config.seed)]
    outputs = llm.generate(inputs, sampling_params=sampling, use_tqdm=False)
    if len(outputs) != len(inputs):
        raise InferenceError("offline_output_count_mismatch")
    for item, output in zip(active, outputs):
        metadata = {"request_id": output.request_id, "finish_reason": None}
        result, error = None, None
        try:
            if len(output.outputs) != 1:
                raise FormatFailure("invalid_completion_count")
            completion = output.outputs[0]
            prompt_count = len(output.prompt_token_ids)
            completion_count = len(completion.token_ids)
            metadata.update({"finish_reason": completion.finish_reason,
                             "usage": {"prompt_tokens": prompt_count, "completion_tokens": completion_count,
                                       "total_tokens": prompt_count + completion_count}})
            if completion.finish_reason != "stop":
                raise FormatFailure("incomplete_generation")
            result = parse_annotation(completion.text, item["candidates"])
        except FormatFailure as exc:
            error = str(exc)
        results.append({"annotation_id": item["annotation_id"], "attempt": item["attempt"],
                        "result": result, "error": error, "metadata": metadata})
    return results


def offline_worker(rank: int, config: RunConfig, settings: OfflineConfig, metadata: dict,
                   port: int, incoming, events, log_path: str) -> None:
    os.setsid()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.environ.update({"VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
                       "VLLM_DP_SIZE": str(settings.data_parallel_size),
                       "VLLM_DP_MASTER_IP": "127.0.0.1", "VLLM_DP_MASTER_PORT": str(port),
                       "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false"})
    with open(log_path, "a", encoding="utf-8", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            from vllm import LLM

            llm = LLM(**metadata["engine_arguments"])
            events.put({"kind": "ready", "rank": rank, "pid": os.getpid(), "ready_at": timestamp()})
            while True:
                job = incoming.get()
                if job is None:
                    # Let collective-processing loops pause before all ranks tear down.
                    time.sleep(1)
                    llm.llm_engine.engine_core.shutdown(timeout=settings.shutdown_timeout)
                    return
                started = time.monotonic()
                results = generate_batch(llm, config, settings, job["items"])
                events.put({"kind": "batch", "rank": rank, "round": job["round"], "results": results,
                            "batch_duration_seconds": time.monotonic() - started,
                            "batch_completed_at": timestamp()})
        except BaseException as exc:
            # Third-party exception messages can echo prompts or completions.
            events.put({"kind": "error", "rank": rank, "error": f"offline_worker_failed:{type(exc).__name__}"})


def balanced_batches(items: list[dict], size: int) -> list[list[dict]]:
    quotient, remainder = divmod(len(items), size)
    boundaries = [rank * quotient + min(rank, remainder) for rank in range(size + 1)]
    return [items[boundaries[rank]:boundaries[rank + 1]] for rank in range(size)]


class OfflineWorkers:
    """Own spawned engines; the caller alone writes annotation checkpoints."""

    def __init__(self, config: RunConfig, settings: OfflineConfig, metadata: dict, run_dir: Path):
        self.config, self.settings, self.metadata, self.run_dir = config, settings, metadata, run_dir
        self.context = mp.get_context("spawn")
        self.incoming = [self.context.Queue(maxsize=1) for _ in range(settings.data_parallel_size)]
        self.events = self.context.Queue(maxsize=settings.data_parallel_size * 2)
        self.processes = []
        self.round = 0
        self.state = {"backend": "vllm_offline", "status": "starting", "started_at": timestamp(),
                      "settings": asdict(settings), "workers": [],
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def save_state(self) -> None:
        write_json(self.run_dir / "inference-session.json", self.state)

    def __enter__(self):
        try:
            self.state["devices"] = check_hardware(self.settings)
            self.save_state()
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            for rank, incoming in enumerate(self.incoming):
                process = self.context.Process(target=offline_worker, name=f"annotation-dp-{rank}",
                                               args=(rank, self.config, self.settings, self.metadata, port,
                                                     incoming, self.events,
                                                     str(self.run_dir / f"vllm-rank-{rank:04d}.log")))
                process.start()
                self.processes.append(process)
            waiting = set(range(self.settings.data_parallel_size))
            deadline = time.monotonic() + self.settings.startup_timeout
            while waiting:
                event = self.next_event(deadline, "offline_startup_timeout")
                if event["kind"] != "ready" or event["rank"] not in waiting:
                    raise InferenceError("offline_startup_protocol_error")
                waiting.remove(event["rank"])
                self.state["workers"].append(event)
                self.save_state()
            self.state.update({"status": "ready", "ready_at": timestamp()})
            self.save_state()
            return self
        except BaseException:
            self.close(graceful=False)
            raise

    def next_event(self, deadline: float | None, timeout_error: str) -> dict:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise InferenceError(timeout_error)
            try:
                event = self.events.get(timeout=POLL_INTERVAL)
            except queue.Empty:
                if any(p.exitcode is not None for p in self.processes):
                    raise InferenceError("offline_worker_exited")
                continue
            if event["kind"] == "error":
                raise InferenceError(event["error"])
            return event

    def generate(self, items: list[dict]):
        """Yield each rank's batch immediately for durable checkpoint writes."""
        if len(items) > self.settings.batch_size * self.settings.data_parallel_size:
            raise ValueError("Offline round exceeds configured batch capacity")
        self.round += 1
        batches = balanced_batches(items, self.settings.data_parallel_size)
        for incoming, batch in zip(self.incoming, batches, strict=True):
            incoming.put({"round": self.round, "items": batch}, timeout=POLL_INTERVAL)
        waiting = set(range(self.settings.data_parallel_size))
        deadline = time.monotonic() + self.settings.batch_timeout if self.settings.batch_timeout else None
        while waiting:
            event = self.next_event(deadline, "offline_batch_timeout")
            if event["kind"] != "batch" or event["round"] != self.round or event["rank"] not in waiting:
                raise InferenceError("offline_batch_protocol_error")
            waiting.remove(event["rank"])
            expected = {(x["annotation_id"], x["attempt"]) for x in batches[event["rank"]]}
            received = [(x["annotation_id"], x["attempt"]) for x in event["results"]]
            if len(received) != len(expected) or set(received) != expected:
                raise InferenceError("offline_batch_result_mismatch")
            yield event

    def close(self, graceful: bool = True) -> None:
        # A second interrupt must not leave descendants alive midway through cleanup.
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            self.stop_workers(graceful)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)

    def stop_workers(self, graceful: bool) -> None:
        if graceful:
            for incoming in self.incoming:
                try:
                    incoming.put_nowait(None)
                except queue.Full:
                    pass
            deadline = time.monotonic() + self.settings.shutdown_timeout
            for process in self.processes:
                process.join(timeout=max(0, deadline - time.monotonic()))
        # Also signal groups after a rank exits: its GPU workers may still be alive.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for process in self.processes:
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    if process.is_alive():
                        process.terminate() if sig == signal.SIGTERM else process.kill()
            deadline = time.monotonic() + (min(self.settings.shutdown_timeout, 2) if sig == signal.SIGTERM else 2)
            for process in self.processes:
                process.join(timeout=max(0, deadline - time.monotonic()))
        for channel in [*self.incoming, self.events]:
            channel.cancel_join_thread()
            channel.close()
        self.state.update({"status": "stopped" if graceful else "aborted", "stopped_at": timestamp(),
                           "exit_codes": [p.exitcode for p in self.processes]})
        self.save_state()

    def __exit__(self, exc_type, exc, traceback):
        self.close(graceful=exc_type is None)
