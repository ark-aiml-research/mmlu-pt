"""Python-owned vLLM API process for a single scheduled annotation job."""

import asyncio
import json
import math
import os
import signal
import socket
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from huggingface_hub import HfApi

from .config import RunConfig, timestamp, write_json


@dataclass(frozen=True)
class ServerConfig:
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 50
    max_num_batched_tokens: int = 4096
    enforce_eager: bool = True
    startup_timeout: float = 1800.0
    shutdown_timeout: float = 30.0
    poll_interval: float = 1.0

    def __post_init__(self):
        for name in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        for name in ("startup_timeout", "shutdown_timeout", "poll_interval"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def server_address(endpoint: str) -> tuple[str, int]:
    url = urlsplit(endpoint)
    if (url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1", "::1")
            or url.path.rstrip("/") != "/v1" or url.query or url.fragment or url.username or url.password):
        raise ValueError("Managed vLLM requires a loopback HTTP endpoint ending in /v1; use --server-mode external otherwise")
    return ("127.0.0.1" if url.hostname == "localhost" else url.hostname), url.port or 80


def launch_command(config: RunConfig, settings: ServerConfig, revision: str) -> list[str]:
    host, port = server_address(config.endpoint)
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", config.model,
            "--served-model-name", config.model, "--host", host, "--port", str(port),
            "--tensor-parallel-size", "1", "--max-model-len", str(settings.max_model_len),
            "--gpu-memory-utilization", str(settings.gpu_memory_utilization),
            "--max-num-seqs", str(settings.max_num_seqs),
            "--max-num-batched-tokens", str(settings.max_num_batched_tokens),
            "--language-model-only", "--reasoning-parser", "qwen3", "--generation-config", "vllm",
            "--structured-outputs-config", '{"backend":"xgrammar","enable_in_reasoning":false}',
            "--no-enable-prefix-caching", "--no-enable-log-requests", "--no-enable-log-outputs",
            "--revision", revision, "--tokenizer-revision", revision,
            *(["--enforce-eager"] if settings.enforce_eager else [])]


def prepare_server(config: RunConfig, settings: ServerConfig, previous: dict | None = None) -> dict:
    if version("vllm") != "0.25.0":
        raise ValueError("Managed annotation requires vLLM 0.25.0: uv sync --group annotation")
    server_address(config.endpoint)
    if config.max_tokens >= settings.max_model_len:
        raise ValueError("max_tokens must leave room for the prompt within max_model_len")
    # A compatible resumed run reuses the weight pin and does not resolve main again.
    revision = ((previous or {}).get("server_metadata") or {}).get("revision")
    if revision is None:
        revision = HfApi().model_info(config.model, revision=config.model_revision).sha
    if not isinstance(revision, str) or not revision:
        raise ValueError("Could not resolve an immutable model revision")
    launch_settings = {k: v for k, v in asdict(settings).items()
                       if k not in ("startup_timeout", "shutdown_timeout", "poll_interval")}
    return {"mode": "managed", "model": config.model, "revision": revision, "vllm_version": "0.25.0",
            "launch_command": launch_command(config, settings, revision), "settings": launch_settings,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0"), "endpoint": config.endpoint}


def check_available_port(endpoint: str) -> None:
    host, port = server_address(endpoint)
    family = socket.AF_INET6 if host == "::1" else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            raise ValueError(f"Managed vLLM port {port} is unavailable; choose another --endpoint or --server-mode external") from None


async def wait_until_ready(process, metadata: dict, settings: ServerConfig, api_key: str,
                           transport=None) -> None:
    endpoint = metadata["endpoint"].rstrip("/")
    origin = endpoint.rsplit("/v1", 1)[0]
    deadline = asyncio.get_running_loop().time() + settings.startup_timeout
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=2.0, trust_env=False, headers=headers, transport=transport) as client:
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                raise ValueError(f"vLLM exited during startup (code {process.returncode}); inspect vllm.log")
            try:
                remaining = deadline - asyncio.get_running_loop().time()
                async with asyncio.timeout(remaining):
                    health = await client.get(origin + "/health")
                    if health.status_code in (401, 403):
                        raise ValueError("vLLM readiness authentication failed; inspect vllm.log")
                    if health.status_code == 200:
                        models = await client.get(endpoint + "/models")
                        if models.status_code in (401, 403):
                            raise ValueError("vLLM readiness authentication failed; inspect vllm.log")
                        if models.status_code == 200:
                            names = [model["id"] for model in models.json()["data"]]
                            if metadata["model"] not in names:
                                raise ValueError("vLLM readiness model mismatch; inspect vllm.log")
                            return
            except (httpx.TransportError, TimeoutError, json.JSONDecodeError, KeyError, TypeError):
                pass
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining > 0:
                await asyncio.sleep(min(settings.poll_interval, remaining))
    raise ValueError("vLLM startup timed out; inspect vllm.log")


def signal_process_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


async def stop_server(process, timeout: float) -> None:
    signal_process_group(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except TimeoutError:
        signal_process_group(process.pid, signal.SIGKILL)
        await process.wait()
    finally:
        # A dead API parent can leave engine/worker descendants in its process group.
        signal_process_group(process.pid, signal.SIGKILL)


@asynccontextmanager
async def managed_server(metadata: dict, settings: ServerConfig, run_dir: Path, api_key: str):
    check_available_port(metadata["endpoint"])
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": metadata["cuda_visible_devices"],
                        "VLLM_API_KEY": api_key, "VLLM_LOGGING_LEVEL": "INFO",
                        "VLLM_DEBUG_LOG_API_SERVER_RESPONSE": "0"})
    session = {"started_at": timestamp(), "status": "starting", "metadata": metadata,
               "operational_settings": asdict(settings)}
    process = None
    path = run_dir / "server-session.json"
    write_json(path, session)
    with (run_dir / "vllm.log").open("a", encoding="utf-8") as log:
        try:
            spawning = asyncio.create_task(asyncio.create_subprocess_exec(*metadata["launch_command"], env=environment,
                         stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True))
            try:
                process = await asyncio.shield(spawning)
            except asyncio.CancelledError:
                process = await spawning
                raise
            session["pid"] = process.pid
            write_json(path, session)
            print(f"Starting vLLM for this job (PID {process.pid}); waiting for {metadata['endpoint']}")
            await wait_until_ready(process, metadata, settings, api_key)
            session["status"], session["ready_at"] = "ready", timestamp()
            write_json(path, session)
            yield process
        finally:
            try:
                if process is not None:
                    cleanup = asyncio.create_task(stop_server(process, settings.shutdown_timeout))
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        await cleanup
                        raise
            finally:
                session["exit_code"] = process.returncode if process else None
                session["status"], session["stopped_at"] = "stopped", timestamp()
                write_json(path, session)


async def run_while_server_alive(process, work) -> None:
    async def watch():
        code = await process.wait()
        raise ValueError(f"vLLM exited during annotation (code {code}); inspect vllm.log and resume the job")

    async with asyncio.TaskGroup() as group:
        watcher = group.create_task(watch())
        task = group.create_task(work)
        await task
        watcher.cancel()
