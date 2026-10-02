"""Inferência offline com processos DP nativos do vLLM 0.19.1."""

import logging
import multiprocessing as mp
import os
import queue
import signal
import socket
import time
import traceback
from importlib.metadata import version
from pathlib import Path

from mmlu_pt.classification.config import MAX_OUTPUT_TOKENS, MODEL_ID, VLLM_VERSION, ClassificationConfig
from mmlu_pt.classification.persistence import append_predictions
from mmlu_pt.classification.taxonomy import LEVELS, allowed_subjects, macro_area, messages_for

LOGGER = logging.getLogger(__name__)


def check_hardware(config: ClassificationConfig) -> list[dict]:
    if version("vllm") != VLLM_VERSION:
        raise RuntimeError(f"Esta implementação requer vllm=={VLLM_VERSION}.")
    import torch

    required = config.data_parallel_size * config.tensor_parallel_size
    available = torch.cuda.device_count()
    if not torch.cuda.is_available() or available < required:
        raise RuntimeError(f"DP × TP requer {required} GPUs CUDA visíveis; disponíveis: {available}.")
    return [
        {"device": i, "name": torch.cuda.get_device_properties(i).name,
         "total_bytes": torch.cuda.get_device_properties(i).total_memory}
        for i in range(required)
    ]


def partition_records(records: list[dict], size: int) -> list[list[dict]]:
    quotient, remainder = divmod(len(records), size)
    boundaries = [rank * quotient + min(rank, remainder) for rank in range(size + 1)]
    return [records[boundaries[rank]:boundaries[rank + 1]] for rank in range(size)]


def prepare_prompt(tokenizer, record: dict, config: ClassificationConfig) -> list[int]:
    tokens = tokenizer.apply_chat_template(
        messages_for(record), tokenize=True, add_generation_prompt=True,
        enable_thinking=False, truncation=False,
    )
    if len(tokens) + MAX_OUTPUT_TOKENS > config.max_model_len:
        raise ValueError(
            f"Contexto excedido: split={record['split']!r}, index={record['index']}, "
            f"row_hash={record['row_hash']}; prompt={len(tokens)} + "
            f"resposta={MAX_OUTPUT_TOKENS} > max_model_len={config.max_model_len}."
        )
    return tokens


def decode_prediction(record: dict, output, prompt_tokens: int) -> dict:
    location = f"split={record['split']!r}, index={record['index']}"
    if len(output.outputs) != 1:
        raise ValueError(f"{location}: esperado exatamente um resultado.")
    completion = output.outputs[0]
    subject = completion.text.strip()
    area = macro_area(record["academic_level"], subject)
    if completion.finish_reason != "stop":
        raise ValueError(f"{location}: geração incompleta ({completion.finish_reason!r}).")
    return {
        "split": record["split"], "index": record["index"], "row_hash": record["row_hash"],
        "subject": subject, "macro_area": area, "prompt_tokens": prompt_tokens,
        "output_tokens": len(completion.token_ids),
    }


def _worker(rank: int, records: list[dict], placeholder: dict, rounds: int, port: int,
            model_revision: str, config: ClassificationConfig, checkpoint: str, barrier, events) -> None:
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.environ.update({
        "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
        "VLLM_DP_SIZE": str(config.data_parallel_size), "VLLM_DP_MASTER_IP": "127.0.0.1",
        "VLLM_DP_MASTER_PORT": str(port), "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TOKENIZERS_PARALLELISM": "false",
    })
    try:
        # O vLLM calcula CUDA_VISIBLE_DEVICES de cada engine a partir do rank local.
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        llm = LLM(
            model=MODEL_ID, revision=model_revision, tokenizer_revision=model_revision,
            tensor_parallel_size=config.tensor_parallel_size, distributed_executor_backend="mp",
            max_model_len=config.max_model_len, max_num_seqs=config.max_num_seqs,
            gpu_memory_utilization=config.gpu_memory_utilization, seed=config.seed,
            language_model_only=True, enable_expert_parallel=False, generation_config="vllm",
        )
        tokenizer = llm.get_tokenizer()
        sampling = {
            level: SamplingParams(
                temperature=0, max_tokens=MAX_OUTPUT_TOKENS, seed=config.seed,
                structured_outputs=StructuredOutputsParams(choice=allowed_subjects(level)),
            )
            for level in LEVELS
        }
        import torch

        memory = []
        for device in range(rank * config.tensor_parallel_size, (rank + 1) * config.tensor_parallel_size):
            free, total = torch.cuda.mem_get_info(device)
            memory.append({"device": device, "free_bytes_after_load": free, "total_bytes": total})
        events.put({"kind": "ready", "rank": rank, "memory": memory})
        barrier.wait()
        for round_index in range(rounds):
            start = round_index * config.batch_size
            batch = records[start:start + config.batch_size]
            # Até os ranks vazios participam das chamadas coletivas do modelo MoE.
            inputs = batch or [placeholder]
            tokens = [prepare_prompt(tokenizer, row, config) for row in inputs]
            barrier.wait()
            outputs = llm.generate(
                [{"prompt_token_ids": ids} for ids in tokens],
                sampling_params=[sampling[row["academic_level"]] for row in inputs], use_tqdm=False,
            )
            if len(outputs) != len(inputs):
                raise ValueError(f"Rank {rank}: quantidade de respostas divergente.")
            if batch:
                predictions = [
                    decode_prediction(row, output, len(ids))
                    for row, output, ids in zip(batch, outputs, tokens, strict=True)
                ]
                append_predictions(Path(checkpoint) / f"rank-{rank:04d}.jsonl", predictions)
                events.put({"kind": "progress", "rank": rank, "count": len(predictions)})
            barrier.wait()
        # Intervalo usado pelo exemplo oficial para pausar os loops internos DP.
        time.sleep(1)
        llm.llm_engine.engine_core.shutdown(timeout=30)
        events.put({"kind": "done", "rank": rank})
    except BaseException:
        events.put({"kind": "error", "rank": rank, "traceback": traceback.format_exc()})
        raise


def _stop_workers(processes: list) -> None:
    # Cada réplica tem uma sessão própria; os engines e workers TP herdam o grupo.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for process in processes:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                if process.is_alive():
                    process.terminate() if sig == signal.SIGTERM else process.kill()
        for process in processes:
            process.join(timeout=2)


def run_inference(records: list[dict], completed: set[tuple[str, int]], *, checkpoint: Path,
                  model_revision: str, config: ClassificationConfig) -> dict:
    partitions = partition_records(records, config.data_parallel_size)
    pending = [[row for row in part if (row["split"], row["index"]) not in completed] for part in partitions]
    rounds = max((len(part) + config.batch_size - 1) // config.batch_size for part in pending)
    if not rounds:
        return {"generated": 0, "rounds": 0}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = mp.get_context("spawn")
    barrier = context.Barrier(config.data_parallel_size)
    events = context.Queue()
    processes, memory, done = [], {}, set()
    generated = 0
    started = time.monotonic()
    try:
        for rank, part in enumerate(pending):
            process = context.Process(
                target=_worker, name=f"subject-dp-{rank}",
                args=(rank, part, records[0], rounds, port, model_revision, config, str(checkpoint), barrier, events),
            )
            process.start()
            processes.append(process)
        while len(done) < config.data_parallel_size:
            try:
                event = events.get(timeout=0.5)
            except queue.Empty:
                failed = [process for process in processes if process.exitcode is not None and process.exitcode != 0]
                if failed:
                    raise RuntimeError(f"Réplica {failed[0].name} encerrou com código {failed[0].exitcode}.")
                if all(process.exitcode is not None for process in processes):
                    raise RuntimeError("Réplicas encerraram sem confirmar conclusão de todos os ranks.")
                continue
            if event["kind"] == "error":
                raise RuntimeError(f"Falha na réplica DP {event['rank']}:\n{event['traceback']}")
            if event["kind"] == "ready":
                memory[str(event["rank"])] = event["memory"]
                LOGGER.info("Réplica DP %s pronta; memória: %s", event["rank"], event["memory"])
            elif event["kind"] == "progress":
                generated += event["count"]
                LOGGER.info("Classificadas %s/%s questões pendentes", generated, sum(map(len, pending)))
            elif event["kind"] == "done":
                done.add(event["rank"])
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError(f"Réplica {process.name} não encerrou corretamente.")
        if generated != sum(map(len, pending)):
            raise RuntimeError("Contagem de registros gerados divergente.")
        return {"generated": generated, "rounds": rounds, "pending_by_rank": list(map(len, pending)),
                "seconds": time.monotonic() - started, "gpu_memory_after_load": memory}
    finally:
        _stop_workers(processes)
        events.close()
        events.join_thread()
