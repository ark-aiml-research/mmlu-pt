"""Run one model in the isolated vLLM environment, with a single journal writer."""

from __future__ import annotations

import argparse
import errno
import multiprocessing as mp
import os
import queue
import signal
import socket
import time
import traceback
from collections import Counter
from dataclasses import asdict, is_dataclass
from importlib.metadata import version
from pathlib import Path

from tqdm import tqdm

from artifacts import (
    BACKEND, TERMINAL_STATUSES, TRANSFORMERS_VERSION, VLLM_VERSION, append_rows,
    latest_responses, read_json, read_jsonl, record_operation, response_summary, utc_now, write_json,
)


def engine_kwargs(spec: dict, checkpoint: dict, settings: dict, generation: dict) -> dict:
    arguments = {
        "model": spec["repo"], "revision": checkpoint["revision"],
        "tokenizer_revision": checkpoint["tokenizer_revision"],
        "tensor_parallel_size": settings["tensor_parallel_size"],
        "distributed_executor_backend": "mp", "dtype": settings["dtype"],
        "max_model_len": settings["max_model_len"], "max_num_seqs": settings["concurrency"],
        "gpu_memory_utilization": settings["gpu_memory_utilization"],
        "generation_config": "vllm", "seed": generation["seed"],
        "language_model_only": spec.get("language_model_only", False),
        "enable_expert_parallel": False, "enable_prefix_caching": True,
        "disable_log_stats": False,
    }
    if spec.get("trust_remote_code"):
        arguments.update(trust_remote_code=True, code_revision=checkpoint["revision"])
    if spec.get("mamba_ssm_cache_dtype"):
        arguments["mamba_ssm_cache_dtype"] = spec["mamba_ssm_cache_dtype"]
    return arguments


def partition_tasks(tasks: list[dict], size: int) -> list[list[dict]]:
    quotient, remainder = divmod(len(tasks), size)
    boundaries = [rank * quotient + min(rank, remainder) for rank in range(size + 1)]
    return [tasks[boundaries[rank]:boundaries[rank + 1]] for rank in range(size)]


def retryable_error(error: BaseException) -> bool:
    """Retry known transport/timeouts only; unknown engine failures remain persistent."""
    chain, current = [], error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    message = " ".join(str(item).lower() for item in chain)
    if any(text in message for text in (
        "out of memory", "cuda error", "unauthorized", "forbidden", "gated repo",
        "not found", "unsupported", "not supported", "invalid", "permission denied",
    )):
        return False
    for item in chain:
        status = getattr(getattr(item, "response", None), "status_code", None)
        if status is not None:
            return status in {408, 429, 500, 502, 503, 504}
        if isinstance(item, (TimeoutError, ConnectionError)):
            return True
        if isinstance(item, OSError) and item.errno in {
            errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED, errno.EPIPE,
        }:
            return True
        if type(item).__name__ in {"ConnectTimeout", "ReadTimeout", "ConnectionError"}:
            return True
    return False


def prepare_tokens(llm, tokenizer, content_format: str, task: dict, spec: dict) -> list[int]:
    from vllm.entrypoints.chat_utils import parse_chat_messages
    from vllm.renderers.hf import safe_apply_chat_template

    messages = [{"role": "user", "content": task["prompt"]}]
    conversation, _, _ = parse_chat_messages(messages, llm.model_config, content_format)
    # Use vLLM's pinned resolver: some official templates live on the HF processor.
    tokens = safe_apply_chat_template(
        llm.model_config, tokenizer, conversation, tokenize=True,
        add_generation_prompt=True, truncation=False, **spec.get("chat_template_kwargs", {}),
    )
    return list(tokens)


def response_record(task: dict, output, tokens: list[int], spec: dict, batch: dict) -> dict:
    if len(output.outputs) != 1:
        raise ValueError(f"Expected one completion for task {task['task_id']}")
    completion = output.outputs[0]
    metrics = asdict(output.metrics) if is_dataclass(output.metrics) else None
    raw = {
        "format": "vllm_request_output_v1", "request_id": output.request_id,
        "finished": output.finished, "prompt_token_ids": list(output.prompt_token_ids or tokens),
        "num_cached_tokens": output.num_cached_tokens, "metrics": metrics,
        "outputs": [{
            "index": completion.index, "text": completion.text,
            "token_ids": list(completion.token_ids),
            "cumulative_logprob": completion.cumulative_logprob,
            "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
        }],
    }
    completed = output.finished and completion.finish_reason in {"stop", "length"}
    row = {
        **response_location(task, spec, batch), "status": "completed" if completed else "request_error",
        "text": completion.text or "", "finish_reason": completion.finish_reason,
        "prompt_tokens_checked": len(tokens),
        "usage": {"prompt_tokens": len(tokens), "completion_tokens": len(completion.token_ids),
                  "total_tokens": len(tokens) + len(completion.token_ids)},
        "raw_response": raw,
    }
    if not completed:
        row.update(stage="generation", error="vLLM returned an unfinished or aborted completion.")
    return row


def response_location(task: dict, spec: dict, batch: dict) -> dict:
    return {"task_id": task["task_id"], "model": spec["name"], "backend": BACKEND,
            "at": batch["started_at"], **{key: batch[key] for key in
                ("session_id", "attempt", "rank", "batch_id")}}


def generate_batch(llm, tokenizer, content_format, sampling, auxiliary_sampling, auxiliary_tokens,
                   tasks: list[dict], spec: dict, settings: dict, generation: dict,
                   batch: dict, barrier, events) -> dict:
    rows, valid = [], []
    stage = "tokenization"
    batch_started = started = time.perf_counter()
    try:
        events.put({"kind": "stage", "rank": batch["rank"], "stage": stage})
        for task in tasks:
            tokens = prepare_tokens(llm, tokenizer, content_format, task, spec)
            if len(tokens) + generation["max_tokens"] > settings["max_model_len"]:
                rows.append({**response_location(task, spec, batch), "status": "context_overflow",
                             "prompt_tokens_checked": len(tokens)})
            else:
                valid.append((task, tokens))
        batch["tokenization_seconds"] = time.perf_counter() - started
        stage = "synchronization"
        events.put({"kind": "stage", "rank": batch["rank"], "stage": stage})
        started = time.perf_counter()
        barrier.wait()
        prompts = [{"prompt_token_ids": tokens} for _, tokens in valid]
        auxiliary = auxiliary_tokens is not None and not prompts
        if auxiliary:
            prompts = [{"prompt_token_ids": auxiliary_tokens}]
        stage = "generation"
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=auxiliary_sampling if auxiliary else sampling,
                               use_tqdm=False) if prompts else []
        batch["generation_seconds"] = time.perf_counter() - started
        if len(outputs) != len(prompts):
            raise ValueError("vLLM returned a different number of outputs than submitted prompts.")
        stage = "serialization"
        started = time.perf_counter()
        for (task, tokens), output in zip(valid, [] if auxiliary else outputs, strict=True):
            rows.append(response_record(task, output, tokens, spec, batch))
        batch["serialization_seconds"] = time.perf_counter() - started
        event = {"kind": "batch"}
    except BaseException as error:
        batch.update(status="failed", **{f"{stage}_seconds": time.perf_counter() - started})
        completed_ids = {row["task_id"] for row in rows}
        rows.extend({**response_location(task, spec, batch), "status": "request_error",
                     "stage": stage, "error": str(error)}
                    for task in tasks if task["task_id"] not in completed_ids)
        event = {"kind": "error", "stage": stage, "error": traceback.format_exc(),
                 "retryable": retryable_error(error)}
        traceback.print_exc()
    batch.update(batch_wall_seconds=time.perf_counter() - batch_started, ended_at=utc_now())
    return {**event, "rank": batch["rank"], "rows": rows, "batch": batch}


def rank_worker(rank: int, tasks: list[dict], rounds: int, native_dp: bool, port: int,
                metadata: dict, generation: dict, arguments: dict, model_dir: str,
                session_id: str, attempt: int, barrier, acknowledgements, events) -> None:
    spec, settings = metadata["spec"], metadata["server"]
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.environ.update({
        "VLLM_DP_SIZE": str(settings["data_parallel_size"] if native_dp else 1),
        "VLLM_DP_RANK": str(rank if native_dp else 0),
        "VLLM_DP_RANK_LOCAL": str(rank if native_dp else 0),
        "VLLM_DP_MASTER_IP": "127.0.0.1", "VLLM_DP_MASTER_PORT": str(port),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false",
    })
    if not native_dp:
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        start = rank * settings["tensor_parallel_size"]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices[start:start + settings["tensor_parallel_size"]])
    stage = "initialization"
    try:
        with (Path(model_dir) / f"rank-{rank:04d}.log").open("a", buffering=1, encoding="utf-8") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            from vllm import LLM, SamplingParams
            from vllm.renderers.hf import resolve_chat_template_content_format

            started = time.perf_counter()
            llm = LLM(**arguments)
            tokenizer = llm.get_tokenizer()
            content_format = resolve_chat_template_content_format(
                None, None, "auto", tokenizer, model_config=llm.model_config)
            sampling = SamplingParams(**generation, n=1)
            auxiliary_sampling = SamplingParams(temperature=0, max_tokens=1, seed=generation["seed"])
            auxiliary_tokens = tokenizer.encode("Placeholder", add_special_tokens=False) if native_dp else None
            events.put({"kind": "ready", "rank": rank,
                        "initialization_seconds": time.perf_counter() - started,
                        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]})
            barrier.wait()
            stage = "processing"
            for round_index in range(rounds):
                start = round_index * settings["batch_size"]
                batch = {"session_id": session_id, "attempt": attempt, "rank": rank,
                         "round": round_index, "batch_id": f"{session_id}:{attempt}:{rank}:{round_index}",
                         "started_at": utc_now(), "status": "completed",
                         "tokenization_seconds": 0.0, "generation_seconds": 0.0}
                inputs = tasks[start:start + settings["batch_size"]]
                batch["task_count"] = len(inputs)
                event = generate_batch(llm, tokenizer, content_format, sampling, auxiliary_sampling,
                                       auxiliary_tokens, inputs, spec, settings, generation, batch, barrier, events)
                events.put(event)
                if event["kind"] == "error":
                    return
                acknowledgements[rank].wait()
                acknowledgements[rank].clear()
                barrier.wait()
            stage = "shutdown"
            if native_dp:
                time.sleep(1)
            llm.llm_engine.engine_core.shutdown(timeout=settings["shutdown_timeout_seconds"])
            events.put({"kind": "done", "rank": rank})
    except BaseException as error:
        events.put({"kind": "error", "rank": rank, "stage": stage, "rows": [],
                    "error": traceback.format_exc(), "retryable": retryable_error(error)})
        raise


def persist_batch(event: dict, model_dir: Path, latest: dict, counts: Counter, visited: set,
                  progress, outcome: dict) -> None:
    rows = [row for row in event.get("rows", [])
            if latest.get(row["task_id"], {}).get("status") not in TERMINAL_STATUSES]
    started = time.perf_counter()
    if rows:
        append_rows(model_dir / "responses.jsonl", rows, sync=True)
    advance = 0
    for row in rows:
        key = row["task_id"]
        counts[latest.get(key, {}).get("status", "pending")] -= 1
        counts[row["status"]] += 1
        latest[key] = row
        advance += key not in visited
        visited.add(key)
    duration = time.perf_counter() - started
    progress.set_postfix(responses=counts["completed"], errors=counts["request_error"],
                         context=counts["context_overflow"], refresh=False)
    progress.update(advance)
    if "batch" in event:
        batch = {**event["batch"], **response_summary(rows), "persistence_seconds": duration}
        record_operation(model_dir, "batch", **batch)
    if event["kind"] == "error":
        record_operation(model_dir, "error", **{key: event[key] for key in
                         ("rank", "stage", "error", "retryable")},
                         session_id=outcome["session_id"], attempt=outcome["attempt"])


def stop_ranks(processes: list, timeout: float) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
    # The launcher reaps the process group, including engine descendants.


def run_attempt(args) -> int:
    model_dir = args.model_dir
    metadata = read_json(model_dir / "metadata.json")
    spec, settings = metadata["spec"], metadata["server"]
    generation = read_json(args.run_dir / "manifest.json")["config"]["generation"]
    tasks = read_jsonl(args.run_dir / "tasks.jsonl")
    latest = latest_responses(model_dir)
    pending = [[task for task in part if latest.get(task["task_id"], {}).get("status") not in TERMINAL_STATUSES]
               for part in partition_tasks(tasks, settings["data_parallel_size"])]
    rounds = max((len(part) + settings["batch_size"] - 1) // settings["batch_size"] for part in pending)
    counts = Counter(latest.get(task["task_id"], {}).get("status", "pending") for task in tasks)
    visited = {key for key, row in latest.items() if row["status"] in TERMINAL_STATUSES}
    outcome = {"session_id": args.session_id, "attempt": args.attempt, "model": spec["name"],
               "backend": BACKEND, "status": "running", "retryable": False,
               "pending_by_rank": list(map(len, pending)), "startup_seconds": 0.0, "processing_seconds": 0.0}
    context = mp.get_context("spawn")
    barrier, events = context.Barrier(len(pending)), context.Queue()
    acknowledgements = [context.Event() for _ in pending]
    processes, ready, done, heartbeat = [], set(), set(), {}
    started = time.perf_counter()
    processing_started = last_batch_at = None
    progress = tqdm(total=len(tasks), initial=len(visited), desc=spec["name"], unit="task",
                    dynamic_ncols=True, mininterval=1, miniters=1,
                    postfix={"responses": counts["completed"], "errors": counts["request_error"],
                             "context": counts["context_overflow"]})
    try:
        if not rounds:
            outcome["status"] = "finished"
            return 0
        for package, expected in (("vllm", VLLM_VERSION), ("transformers", TRANSFORMERS_VERSION)):
            if version(package) != expected:
                raise ValueError(f"Expected isolated {package}=={expected}")
        import torch
        from vllm import EngineArgs

        required = settings["data_parallel_size"] * settings["tensor_parallel_size"]
        if not torch.cuda.is_available() or torch.cuda.device_count() < required:
            raise ValueError(f"DP × TP requires {required} visible CUDA GPUs; found {torch.cuda.device_count()}.")
        arguments = engine_kwargs(spec, metadata["checkpoint"], settings, generation)
        # vLLM 0.25.0 permits native offline SPMD DP for MoE only.
        native_dp = len(pending) > 1 and EngineArgs(**arguments).create_model_config().is_moe
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        record_operation(model_dir, "execution", session_id=args.session_id, attempt=args.attempt,
                         arguments=arguments, dp_mode="native_moe" if native_dp else "independent_replicas",
                         cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], gpu_count=required,
                         gpu_inventory=[{"index": i, "name": torch.cuda.get_device_properties(i).name,
                                         "total_bytes": torch.cuda.get_device_properties(i).total_memory}
                                        for i in range(required)])
        for rank, part in enumerate(pending):
            process = context.Process(target=rank_worker, name=f"ts-guessing-dp-{rank}",
                                      args=(rank, part, rounds, native_dp, port, metadata, generation,
                                            arguments, str(model_dir), args.session_id, args.attempt,
                                            barrier, acknowledgements, events))
            process.start()
            processes.append(process)
            heartbeat[rank] = time.monotonic()
        while len(done) < len(processes):
            try:
                event = events.get(timeout=0.5)
            except queue.Empty:
                failed = [p for p in processes if p.exitcode not in (None, 0)]
                if failed or all(p.exitcode is not None for p in processes):
                    raise RuntimeError("Ranks exited without confirming completion; see rank logs.")
                for rank, at in heartbeat.items():
                    limit = settings["initialization_timeout_seconds" if rank not in ready else "batch_timeout_seconds"]
                    if rank not in done and time.monotonic() - at > limit:
                        raise TimeoutError(f"Rank {rank} did not report progress for {limit} seconds.")
                continue
            rank = event["rank"]
            heartbeat[rank] = time.monotonic()
            if event["kind"] == "ready":
                ready.add(rank)
                record_operation(model_dir, "initialization", **{key: value for key, value in event.items() if key != "kind"},
                                 session_id=args.session_id, attempt=args.attempt)
                if len(ready) == len(processes):
                    processing_started = time.perf_counter()
                    outcome["startup_seconds"] = processing_started - started
                    progress.start_t = progress.last_print_t = time.time()
                    progress.last_print_n = progress.n
            elif event["kind"] in {"batch", "error"}:
                persist_batch(event, model_dir, latest, counts, visited, progress, outcome)
                if "batch" in event:
                    last_batch_at = time.perf_counter()
                if event["kind"] == "error":
                    outcome.update(error=event["error"], retryable=event["retryable"])
                    raise RuntimeError(event["error"])
                acknowledgements[rank].set()
            elif event["kind"] == "done":
                done.add(rank)
        for process in processes:
            process.join(timeout=settings["shutdown_timeout_seconds"])
            if process.exitcode != 0:
                raise RuntimeError(f"{process.name} did not exit cleanly after generation.")
        outcome["status"] = "finished"
        return 0
    except BaseException as error:
        outcome["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        if "error" not in outcome:
            outcome.update(error=traceback.format_exc(), retryable=retryable_error(error))
            record_operation(model_dir, "error", session_id=args.session_id, attempt=args.attempt,
                             stage="coordinator", error=outcome["error"], retryable=outcome["retryable"])
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        try:
            # Drain delivered batches without releasing another generation round.
            deadline = time.monotonic() + 1
            while processes and time.monotonic() < deadline:
                try:
                    event = events.get(timeout=0.1)
                except queue.Empty:
                    break
                if event["kind"] in {"batch", "error"}:
                    persist_batch(event, model_dir, latest, counts, visited, progress, outcome)
        finally:
            measured_at = time.perf_counter()
            outcome["startup_seconds"] = outcome["startup_seconds"] or (measured_at - started)
            if processing_started is not None:
                end = last_batch_at if outcome["status"] == "finished" else measured_at
                outcome["processing_seconds"] = max(0, (end or measured_at) - processing_started)
            barrier.abort()
            stop_ranks(processes, settings["shutdown_timeout_seconds"])
            events.close()
            events.join_thread()
            progress.close()
            outcome["ended_at"] = utc_now()
            record_operation(model_dir, "attempt", **outcome)
            write_json(args.result_path, outcome)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    return run_attempt(parser.parse_args())


if __name__ == "__main__":
    def interrupt_on_term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_on_term)
    raise SystemExit(main())
