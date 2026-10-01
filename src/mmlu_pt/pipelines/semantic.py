import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, List, Value

from mmlu_pt.mcqa_minimal import PUBLIC_FIELDS

DATASET_FEATURES = Features({
    field: List(Value("string")) if field == "choices" else Value(
        "int64" if field == "answer" else "string"
    )
    for field in PUBLIC_FIELDS
})

MODEL_ID = "Octen/Octen-Embedding-8B"
MODEL_REVISION = "5adcfa292e712091dfc30f0e97f0b2282e6cc66c"
EMBEDDING_DIMENSION = 4096
MAX_LENGTH = 32_768
BATCH_SIZE = 4
COMPARISON_BLOCK_SIZE = 1_024
COSINE_THRESHOLD = 0.9737051129341125
SEED = 42
PREFIX = "- "
PARAMETERS = {
    "model": MODEL_ID,
    "revision": MODEL_REVISION,
    "representation": "question_choices",
    "serialization": "nfkc_whitespace_case_preserved_labeled_choices_v1",
    "prefix": PREFIX,
    "dimension": EMBEDDING_DIMENSION,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "dtype": "bfloat16",
    "attention": "sdpa",
    "padding_side": "left",
    "normalization": "float32_l2",
    "seed": SEED,
    "cosine_threshold": COSINE_THRESHOLD,
    "comparison_block_size": COMPARISON_BLOCK_SIZE,
    "representative_policy": "filled_fields_then_semantic_text_length_then_stable_id_v1",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialize_semantic_text(question: str, choices: list[str]) -> str:
    """Serializa o item com o mesmo protocolo do benchmark pareado."""
    labeled = "\n".join(f"[{chr(ord('A') + index)}] {choice}" for index, choice in enumerate(choices))
    return " ".join(unicodedata.normalize("NFKC", f"{question}\n{labeled}").split())


def read_records(input_dir: Path) -> tuple[list[dict], list[dict]]:
    """Lê os campos públicos e identifica cada ocorrência deterministicamente."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Diretório de entrada semântica ausente: {input_dir}")
    records, manifest = [], []
    occurrences = Counter()
    for path in sorted(input_dir.glob("*.jsonl")):
        original_hash = file_sha256(path)
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise ValueError("O registro deve ser um objeto JSON")
                    question, choices = raw.get("question"), raw.get("choices")
                    if not isinstance(question, str):
                        raise ValueError("question deve ser uma string")
                    if not isinstance(choices, list) or not all(isinstance(choice, str) for choice in choices):
                        raise ValueError("choices deve ser uma lista de strings")
                    public = {field: raw.get(field) for field in PUBLIC_FIELDS}
                    payload = canonical_json(public)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Registro inválido em {path}:{line_number}: {error}") from error
                digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                occurrence = occurrences[digest]
                occurrences[digest] += 1
                filled = sum(value is not None and value != "" and value != [] for value in public.values())
                records.append({
                    "id": f"{digest}:{occurrence:08d}",
                    "public": public,
                    "text": serialize_semantic_text(question, choices),
                    "filled_fields": filled,
                    "source": path.name,
                    "line": line_number,
                })
                count += 1
        if file_sha256(path) != original_hash:
            raise RuntimeError(f"A entrada mudou durante a leitura: {path}")
        manifest.append({"file": path.name, "sha256": original_hash, "records": count})
    records.sort(key=lambda row: (-row["filled_fields"], -len(row["text"]), row["id"]))
    return records, manifest


def validate_embeddings(embeddings: np.ndarray, count: int) -> None:
    if embeddings.shape != (count, EMBEDDING_DIMENSION) or embeddings.dtype != np.float32:
        raise ValueError("Matriz de embeddings incompatível com os registros")
    for start in range(0, count, COMPARISON_BLOCK_SIZE):
        block = embeddings[start:start + COMPARISON_BLOCK_SIZE]
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-5):
            raise ValueError("Embeddings devem ser finitos e normalizados por L2")


def compute_embeddings(texts: list[str]) -> tuple[np.ndarray, dict]:
    """Reproduz a inferência Octen do notebook em uma GPU."""
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32), {"gpu": None, "tokens": []}
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("A geração de embeddings Octen exige uma GPU NVIDIA/CUDA")
    torch.manual_seed(SEED)
    model = SentenceTransformer(
        MODEL_ID, revision=MODEL_REVISION, device="cuda", trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    try:
        model.max_seq_length = MAX_LENGTH
        model.eval()
        lengths = []
        for start in range(0, len(texts), 64):
            encoded = model.tokenizer(
                [PREFIX + text for text in texts[start:start + 64]],
                truncation=False, padding=False, verbose=False,
            )
            lengths.extend(map(len, encoded["input_ids"]))
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        embeddings = model.encode(
            texts, prompt=PREFIX, batch_size=BATCH_SIZE,
            normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=True,
        ).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms == 0).any():
            raise ValueError("Octen produziu embeddings inválidos")
        embeddings /= norms
        torch.cuda.synchronize()
        validate_embeddings(embeddings, len(texts))
        return embeddings, {
            "gpu": torch.cuda.get_device_name(0), "tokens": lengths,
            "seconds": time.perf_counter() - started,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def load_or_compute_embeddings(records: list[dict], cache_dir: Path, identity: dict) -> tuple[np.ndarray, dict]:
    """Reutiliza somente embeddings íntegros da mesma entrada e configuração."""
    embedding_path, info_path = cache_dir / "embeddings.npy", cache_dir / "info.json"
    if embedding_path.is_file() and info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if info["identity"] != identity or info["sha256"] != file_sha256(embedding_path):
                raise ValueError("Identidade ou checksum do cache divergente")
            if len(info["tokens"]) != len(records) or any(type(n) is not int or n < 0 for n in info["tokens"]):
                raise ValueError("Auditoria de tokens incompatível")
            embeddings = np.load(embedding_path, allow_pickle=False)
            validate_embeddings(embeddings, len(records))
            return embeddings, {**info, "cache_reused": True, "cache_dir": str(cache_dir)}
        except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
            print(f"Cache de embeddings inválido; recalculando: {error}", flush=True)
    embeddings, info = compute_embeddings([row["text"] for row in records])
    validate_embeddings(embeddings, len(records))
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".embeddings-", dir=cache_dir) as temporary:
        staged = Path(temporary)
        np.save(staged / "embeddings.npy", embeddings, allow_pickle=False)
        info = {
            **info, "identity": identity, "sha256": file_sha256(staged / "embeddings.npy"),
            "truncated_items": sum(n > MAX_LENGTH for n in info["tokens"]),
        }
        (staged / "info.json").write_text(canonical_json(info) + "\n", encoding="utf-8")
        os.replace(staged / "embeddings.npy", embedding_path)
        os.replace(staged / "info.json", info_path)
    return embeddings, {**info, "cache_reused": False, "cache_dir": str(cache_dir)}


def select_representatives(records: list[dict], embeddings: np.ndarray) -> tuple[list[int], list[dict], dict]:
    """Compara blocos globalmente e remove somente por representantes mantidos."""
    validate_embeddings(embeddings, len(records))
    if not records:
        return [], [], {"comparison_device": None}
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kept, removals = [], []
    # TF32 pode alterar decisões próximas ao limiar ajustado em float32.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in range(0, len(records), COMPARISON_BLOCK_SIZE):
            stop = min(start + COMPARISON_BLOCK_SIZE, len(records))
            block = torch.as_tensor(embeddings[start:stop], device=device)
            representatives = np.full(stop - start, -1, dtype=np.int64)
            similarities = np.zeros(stop - start, dtype=np.float32)
            for offset in range(0, len(kept), COMPARISON_BLOCK_SIZE):
                candidates = kept[offset:offset + COMPARISON_BLOCK_SIZE]
                reference = torch.as_tensor(embeddings[candidates], device=device)
                scores = (block @ reference.T).clamp(-1, 1).cpu().numpy()
                matches = scores.astype(np.float64) >= COSINE_THRESHOLD
                rows = np.flatnonzero((representatives < 0) & matches.any(axis=1))
                columns = matches[rows].argmax(axis=1)
                representatives[rows] = np.asarray(candidates)[columns]
                similarities[rows] = scores[rows, columns]
                if (representatives >= 0).all():
                    break
            scores = (block @ block.T).clamp(-1, 1).cpu().numpy()
            local_kept = []
            for local, index in enumerate(range(start, stop)):
                representative = int(representatives[local])
                similarity = float(similarities[local])
                if representative < 0:
                    matches = np.flatnonzero(scores[local, local_kept].astype(np.float64) >= COSINE_THRESHOLD)
                    if len(matches):
                        candidate = local_kept[int(matches[0])]
                        representative, similarity = start + candidate, float(scores[local, candidate])
                if representative < 0:
                    kept.append(index)
                    local_kept.append(local)
                    continue
                removals.append({
                    "removed_id": records[index]["id"],
                    "representative_id": records[representative]["id"],
                    "cosine": similarity,
                })
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    kept_ids = {records[index]["id"] for index in kept}
    removed_ids = {row["removed_id"] for row in removals}
    if kept_ids & removed_ids or len(kept_ids) + len(removed_ids) != len(records):
        raise AssertionError("Partição semântica inválida")
    for row in removals:
        if row["representative_id"] not in kept_ids or row["cosine"] < COSINE_THRESHOLD:
            raise AssertionError("Remoção sem correspondência direta a um representante mantido")
    return kept, removals, {"comparison_device": device}


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def save_local_dataset(path: Path, rows) -> None:
    """Salva os campos públicos com um schema estável, inclusive sem registros."""
    columns = {field: [] for field in PUBLIC_FIELDS}
    for row in rows:
        for field in PUBLIC_FIELDS:
            value = row[field]
            if field not in {"answer", "choices"} and value is not None:
                value = str(value)
            columns[field].append(value)
    dataset = Dataset.from_dict(columns, features=DATASET_FEATURES)
    # Um shard explícito permite recarregar datasets vazios no datasets 5.0.1.
    dataset.save_to_disk(str(path), num_shards=1 if not len(dataset) else None)


def run_semantic_deduplication(input_dir: Path, output_dir: Path, work_dir: Path) -> dict:
    """Executa a etapa semântica e publica o dataset somente após validá-lo."""
    started = time.perf_counter()
    input_dir, output_dir, work_dir = input_dir.resolve(), output_dir.resolve(), work_dir.resolve()
    for left, right in ((input_dir, output_dir), (input_dir, work_dir), (output_dir, work_dir)):
        if left == right or left in right.parents or right in left.parents:
            raise ValueError("As pastas de entrada, saída e trabalho devem ser separadas")
    records, manifest = read_records(input_dir)
    versions = {name: importlib.metadata.version(name) for name in (
        "numpy", "torch", "transformers", "sentence-transformers",
    )}
    identity = {
        "inputs": manifest, "parameters": PARAMETERS, "versions": versions,
        "ids": [row["id"] for row in records],
        "text_sha256": hashlib.sha256(canonical_json([row["text"] for row in records]).encode()).hexdigest(),
    }
    fingerprint = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    run_parent = work_dir / fingerprint
    print(f"\nSemantic: {len(records)} registros; embeddings Octen e representantes diretos.", flush=True)
    embeddings, embedding_info = load_or_compute_embeddings(records, run_parent / "embeddings", identity)
    kept, removals, diagnostics = select_representatives(records, embeddings)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=run_parent))
    write_jsonl(run_dir / "removals.jsonl", removals)
    kept_set = set(kept)
    write_jsonl(run_dir / "records.jsonl", (
        {"id": row["id"], "source": row["source"], "line": row["line"], "kept": index in kept_set}
        for index, row in enumerate(records)
    ))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".semantic-output-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "dataset"
        staged.mkdir()
        dataset_path = staged / "data.jsonl"
        write_jsonl(dataset_path, (records[index]["public"] for index in kept))
        with dataset_path.open(encoding="utf-8") as stream:
            output_count = sum(1 for line in stream if set(json.loads(line)) == set(PUBLIC_FIELDS))
        if output_count != len(kept):
            raise AssertionError("Contagem ou schema do output semântico inválido")
        save_local_dataset(staged / "huggingface", (records[index]["public"] for index in kept))
        if sorted(path.name for path in input_dir.glob("*.jsonl")) != [row["file"] for row in manifest]:
            raise RuntimeError("A lista de arquivos de entrada mudou durante a deduplicação")
        for row in manifest:
            if file_sha256(input_dir / row["file"]) != row["sha256"]:
                raise RuntimeError("A entrada mudou durante a deduplicação")
        summary = {
            "fingerprint": fingerprint, "parameters": PARAMETERS, "inputs": manifest,
            "input_dir": str(input_dir), "output_dir": str(output_dir), "run_dir": str(run_dir),
            "input_records": len(records), "output_records": len(kept), "removed_records": len(removals),
            **diagnostics, "embeddings": embedding_info, "versions": versions,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "nemo_curator_version": importlib.metadata.version("nemo-curator"),
            "implementation_sha256": file_sha256(Path(__file__)),
            "audit_sha256": {
                name: file_sha256(run_dir / name)
                for name in ("records.jsonl", "removals.jsonl")
            },
            "output_sha256": file_sha256(dataset_path),
            "elapsed_seconds": time.perf_counter() - started,
        }
        (run_dir / "manifest.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
        backup = None
        if output_dir.exists():
            backup_parent = Path(tempfile.mkdtemp(prefix=".semantic-backup-", dir=output_dir.parent))
            backup = backup_parent / output_dir.name
            output_dir.rename(backup)
        try:
            staged.rename(output_dir)
        except OSError:
            if backup is not None:
                backup.rename(output_dir)
            raise
        if backup is not None:
            print(f"Output semântico anterior preservado em: {backup}")
    print(
        f"Semantic: {len(records)} entradas; {len(kept)} representantes; "
        f"{len(removals)} removidos; {summary['elapsed_seconds']:.2f}s.\nAuditoria: {run_dir}\n"
        f"Dataset Hugging Face local: {output_dir / 'huggingface'}"
    )
    return summary
