"""Explicit run configuration and small artifact helpers."""

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_TAXONOMY = PACKAGE_DIR / "mmlu_pt_taxonomy_v1_1.json"
DEFAULT_ALIASES = PACKAGE_DIR / "exam_aliases.json"
DEFAULT_MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"
DEFAULT_DATASET = "bench-temp/mmlu-pt-filtered"
DEFAULT_OUTPUT_DIR = Path("output/knowledge-annotation-work")
PROMPT_VERSION = "subject_annotation_v1"
UNCERTAIN = "UNCERTAIN"


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class RunConfig:
    dataset: str = DEFAULT_DATASET
    dataset_path: str | None = None
    dataset_revision: str | None = None
    dataset_config: str | None = None
    split: str = "train"
    model: str = DEFAULT_MODEL
    model_revision: str | None = None
    num_independent_passes: int = 2
    adjudicate_disagreements: bool = True
    thinking: bool = True
    include_exam_edition: bool = True
    structured_output: str = "json_schema"
    prompt_version: str = PROMPT_VERSION
    seed: int = 42
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    max_tokens: int = 8192
    max_attempts: int = 5
    format_retries: int = 2
    dry_run: bool = False
    samples_per_exam: int = 5

    def __post_init__(self):
        if self.num_independent_passes not in (1, 2):
            raise ValueError("num_independent_passes must be 1 or 2")
        if self.structured_output not in ("json_schema", "none"):
            raise ValueError("structured_output must be json_schema or none")
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError(f"Only {PROMPT_VERSION} is implemented")
        for name in ("max_tokens", "max_attempts", "samples_per_exam"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be positive or -1")
        for name in ("repetition_penalty",):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.format_retries < 0:
            raise ValueError("Invalid retry configuration")
        for name, low, high in (("temperature", 0, 2), ("top_p", 0, 1),
                                ("min_p", 0, 1), ("presence_penalty", -2, 2)):
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}]")

    def methodology(self) -> dict:
        operational = {"max_attempts", "format_retries"}
        return {k: v for k, v in asdict(self).items() if k not in operational}

    def generation(self) -> dict:
        names = ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                 "repetition_penalty", "max_tokens")
        return {name: getattr(self, name) for name in names}


def request_seed(base: int, annotation_id: str, kind: str, pass_number: int, attempt: int) -> int:
    return int(digest([base, annotation_id, kind, pass_number, attempt])[:8], 16)
