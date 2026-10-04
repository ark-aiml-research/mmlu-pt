"""Shared artifact storage and response summaries for the TS-Guessing study."""

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BACKEND = "offline"
ARTIFACT_VERSION = 2
VLLM_VERSION = "0.25.0"
TRANSFORMERS_VERSION = "5.14.1"
TERMINAL_STATUSES = {"completed", "context_overflow"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_rows(path: Path, rows: list[dict], sync: bool = False) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
        if sync:
            stream.flush()
            os.fsync(stream.fileno())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    append_rows(temporary, rows)
    temporary.replace(path)


def read_jsonl(path: Path, repair: bool = False) -> list[dict]:
    """Ignore a torn final append; repair it only when resuming generation."""
    if not path.exists():
        return []
    rows = []
    with path.open("rb+" if repair else "rb") as stream:
        while line := stream.readline():
            start = stream.tell() - len(line)
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if stream.read(1):
                    raise ValueError(f"Corrupt journal before its final line: {path}")
                if repair:
                    stream.truncate(start)
                break
            if repair and not line.endswith(b"\n"):
                stream.write(b"\n")
    return rows


def latest_responses(model_dir: Path) -> dict:
    return {row["task_id"]: row for row in read_jsonl(model_dir / "responses.jsonl", repair=True)}


def response_summary(rows, expected: int | None = None) -> dict:
    counts = Counter()
    prompt_tokens = completion_tokens = 0
    for row in rows:
        counts[row["status"]] += 1
        if row["status"] == "completed":
            usage = row.get("usage", {})
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)
    result = {f"{status}_tasks": counts[status]
              for status in ("completed", "context_overflow", "request_error")}
    result.update(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    if expected is not None:
        result.update(expected_tasks=expected, pending_tasks=expected - sum(counts.values()),
                      status="completed" if counts["completed"] == expected else "partial")
    return result


def record_operation(model_dir: Path, kind: str, **fields) -> None:
    append_rows(model_dir / "operations.jsonl", [{"at": utc_now(), **fields, "kind": kind}])
