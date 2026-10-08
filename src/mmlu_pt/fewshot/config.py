"""Run configuration and small helpers for few-shot split construction."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from mmlu_pt.annotation.knowledge_area.config import canonical_json, digest, timestamp, write_json  # noqa: F401

DEFAULT_SOURCE = "bench-temp-2/mmlu-pt-knowledge-annotated"
DEFAULT_HIGH_SCHOOL_REPO = "bench-temp-2/mmlu-pt-high-school"
DEFAULT_UNDERGRADUATE_REPO = "bench-temp-2/mmlu-pt-undergraduate"
DEFAULT_OUTPUT_DIR = Path("output/fewshot-work")
DEFAULT_MODEL = "gpt-5.6-sol"
LEVELS = ("high_school", "undergraduate")
LETTERS = "ABCDE"
ID_COLUMN = "id"
RATIONALE_COLUMN = "rationale"
# Source columns kept in the published datasets; other knowledge-annotation columns are dropped.
REQUIRED_COLUMNS = ("exam", "exam_edition", "exam_url", "num", "question", "choices", "answer",
                    "academic_level", "subject", "macro_area")
OUTPUT_COLUMNS = (ID_COLUMN, *REQUIRED_COLUMNS, RATIONALE_COLUMN)
PROMPT_VERSION = "fewshot_rationale_v1"
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class RunConfig:
    source: str = DEFAULT_SOURCE
    source_revision: str | None = None
    split: str = "train"
    high_school_repo: str = DEFAULT_HIGH_SCHOOL_REPO
    undergraduate_repo: str = DEFAULT_UNDERGRADUATE_REPO
    shots: int = 5
    seed: int = 42
    model: str = DEFAULT_MODEL
    reasoning_effort: str = "low"
    max_output_tokens: int = 2000
    rationale_max_chars: int = 1200
    rationale_attempts: int = 2
    max_replacements: int = 10      # rationale rejections per stratum
    shared_prefix_chars: int = 100
    rationale_limit: int | None = None
    dry_run: bool = False
    push: bool = True

    def __post_init__(self):
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}")
        for name in ("shots", "max_output_tokens", "rationale_max_chars", "rationale_attempts",
                     "max_replacements", "shared_prefix_chars"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.rationale_limit is not None and self.rationale_limit < 0:
            raise ValueError("rationale_limit must be non-negative")

    def call_limit(self) -> int | None:
        """Maximum number of API calls; dry runs make none unless a limit is given."""
        if self.rationale_limit is not None:
            return self.rationale_limit
        return 0 if self.dry_run else None


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def letter(index: int) -> str:
    return LETTERS[index]


def level_repo(config: RunConfig, level: str) -> str:
    return config.high_school_repo if level == "high_school" else config.undergraduate_repo


def run_id(config: RunConfig, source: dict, prompt_digest: str) -> str:
    identity = {"source": source["identifier"], "revision": source["revision"], "split": config.split,
                "seed": config.seed, "shots": config.shots, "model": config.model,
                "reasoning_effort": config.reasoning_effort, "prompt_hash": prompt_digest,
                "shared_prefix_chars": config.shared_prefix_chars}
    return digest(identity)[:12]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(record) + "\n")
        handle.flush()
