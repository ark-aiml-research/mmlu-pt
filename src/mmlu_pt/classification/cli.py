"""CLI para datasets do Hub ou salvos com save_to_disk."""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import HfApi

from mmlu_pt.classification.config import ClassificationConfig
from mmlu_pt.classification.runner import classify_dataset

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    defaults = ClassificationConfig()
    parser = argparse.ArgumentParser(description="Classifica disciplinas com Qwen3.5 FP8 e calcula macroáreas.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="ID do dataset no Hugging Face Hub.")
    source.add_argument("--dataset-path", type=Path, help="Diretório produzido por save_to_disk.")
    parser.add_argument("--dataset-config", help="Nome da configuração do dataset no Hub.")
    parser.add_argument("--dataset-revision", help="Revisão do dataset no Hub (padrão: main).")
    parser.add_argument("--split", action="append", help="Split a processar; pode ser repetido. Padrão: todos.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-revision", default=defaults.model_revision, help="Revisão dos pesos e tokenizer do Qwen.")
    parser.add_argument("--data-parallel-size", type=int, default=defaults.data_parallel_size)
    parser.add_argument("--tensor-parallel-size", type=int, default=defaults.tensor_parallel_size)
    parser.add_argument("--max-model-len", type=int, default=defaults.max_model_len)
    parser.add_argument("--max-num-seqs", type=int, default=defaults.max_num_seqs, help="Máximo de sequências concorrentes por réplica.")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="Registros por lote de cada réplica.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=defaults.gpu_memory_utilization)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Exporta e classifica somente as primeiras N linhas de cada split.")
    return parser


def select_splits(dataset: Dataset | DatasetDict, names: list[str] | None) -> Dataset | DatasetDict:
    if names is None:
        return dataset
    if not isinstance(dataset, DatasetDict):
        raise ValueError("--split requer um DatasetDict com splits nomeados.")
    if len(names) != len(set(names)):
        raise ValueError("--split contém nomes duplicados.")
    missing = set(names) - set(dataset)
    if missing:
        raise ValueError(f"Splits inexistentes: {sorted(missing)}; disponíveis: {list(dataset)}.")
    return DatasetDict({name: data for name, data in dataset.items() if name in names})


def load_input(args: argparse.Namespace) -> tuple[Dataset | DatasetDict, dict]:
    if args.dataset:
        requested = args.dataset_revision or "main"
        resolved = HfApi().dataset_info(args.dataset, revision=requested).sha
        if not resolved:
            raise RuntimeError("Não foi possível resolver a revisão do dataset.")
        dataset = load_dataset(args.dataset, name=args.dataset_config, revision=resolved)
        source = {"kind": "hub", "id": args.dataset, "config": args.dataset_config,
                  "requested_revision": requested, "resolved_revision": resolved}
    else:
        if args.dataset_config or args.dataset_revision:
            raise ValueError("--dataset-config e --dataset-revision são exclusivos de --dataset.")
        path = args.dataset_path.expanduser().resolve()
        dataset = load_from_disk(str(path))
        source = {"kind": "disk", "path": str(path)}
    return select_splits(dataset, args.split), source


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = ClassificationConfig(
        model_revision=args.model_revision, data_parallel_size=args.data_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs, batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed, resume=args.resume,
        limit=args.limit,
    )
    try:
        config.validate()
        dataset, source = load_input(args)
        classify_dataset(dataset, output_dir=args.output_dir, config=replace(config, dataset_source=source))
    except KeyboardInterrupt:
        LOGGER.error("Interrompido; use o mesmo comando com --resume para retomar os checkpoints disponíveis.")
        return 130
    except Exception as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
