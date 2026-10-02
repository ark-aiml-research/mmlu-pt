"""Parâmetros da classificação independente do pipeline de curadoria."""

from dataclasses import asdict, dataclass

MODEL_ID = "Qwen/Qwen3.5-122B-A10B-FP8"
VLLM_VERSION = "0.19.1"
MAX_OUTPUT_TOKENS = 32
LABEL_COLUMNS = ("subject", "macro_area")


@dataclass(frozen=True)
class ClassificationConfig:
    model_revision: str = "main"
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    max_num_seqs: int = 32
    batch_size: int = 128
    gpu_memory_utilization: float = 0.90
    seed: int = 42
    resume: bool = False
    limit: int | None = None
    dataset_source: dict | None = None

    def validate(self) -> None:
        for name in (
            "data_parallel_size", "tensor_parallel_size", "max_model_len",
            "max_num_seqs", "batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} deve ser um inteiro positivo.")
        if self.max_model_len <= MAX_OUTPUT_TOKENS:
            raise ValueError("max_model_len deve comportar o prompt e a resposta.")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization deve estar em (0, 1].")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed deve ser um inteiro não negativo.")
        if self.limit is not None and (type(self.limit) is not int or self.limit < 0):
            raise ValueError("limit deve ser um inteiro não negativo.")
        if not isinstance(self.model_revision, str) or not self.model_revision.strip():
            raise ValueError("model_revision não pode ser vazio.")

    def identity_parameters(self) -> dict:
        values = asdict(self)
        for name in ("resume", "dataset_source"):
            values.pop(name)
        return values
