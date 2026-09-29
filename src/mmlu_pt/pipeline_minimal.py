import os
import shutil

# Evita que o Ray recrie o ambiente gerenciado pelo uv em /tmp para cada execução.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse  # noqa: I001
from pathlib import Path

import ray
from nemo_curator.core.client import RayClient

from mmlu_pt.pipelines.definition import (
    DEDUPLICATED_DIR,
    DEDUPLICATION_WORK_DIR,
    FILTERED_DIR,
    ORIGINAL_DIR,
    PRE_WORD_FILTER_DIR,
    READ_DIR,
    SEMANTIC_DEDUPLICATED_DIR,
    SEMANTIC_WORK_DIR,
    PipelineStep,
    create_deduplication_input_pipeline,
    create_duplicate_removal_workflow,
    create_exact_deduplication_workflow,
    create_preprocessing_pipeline,
)
from mmlu_pt.pipelines.semantic import run_semantic_deduplication
from mmlu_pt.utils.manifest import read_manifest_file
from mmlu_pt.utils.pipeline_utils import (
    get_stage_record_counts,
    prepare_sources_for_processing,
    print_duplicate_removal_summary,
    print_exact_deduplication_summary,
    print_stage_record_counts,
)

DEFAULT_MANIFEST_FILE = Path("config/sources.yaml")
STEP_INPUT_DIRS = {
    PipelineStep.NORMALIZE: ORIGINAL_DIR,
    PipelineStep.STRUCTURAL_FILTER: READ_DIR,
    PipelineStep.WORD_FILTER: PRE_WORD_FILTER_DIR,
    PipelineStep.DEDUPLICATE: FILTERED_DIR,
    PipelineStep.SEMANTIC_DEDUPLICATE: DEDUPLICATED_DIR,
}


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser dos parâmetros do pipeline."""
    parser = argparse.ArgumentParser(description="Executa o pipeline mínimo do MMLU-PT.")
    parser.add_argument(
        "--config",
        "--manifest-file",
        dest="manifest_file",
        type=Path,
        default=DEFAULT_MANIFEST_FILE,
        help="Caminho do manifest YAML (padrão: config/sources.yaml).",
    )
    parser.add_argument(
        "--resume-from-step",
        type=int,
        choices=tuple(step.value for step in PipelineStep),
        default=PipelineStep.PREPARE_SOURCES.value,
        help="Etapa inicial da execução; reutiliza o output da etapa anterior.",
    )
    return parser


def _validate_resume_input(start_step: PipelineStep) -> None:
    """Verifica se o output exigido para retomar o pipeline está disponível."""
    if start_step == PipelineStep.PREPARE_SOURCES:
        return

    input_dir = STEP_INPUT_DIRS[start_step]
    if start_step == PipelineStep.SEMANTIC_DEDUPLICATE and input_dir.is_dir():
        return
    if input_dir.is_dir() and any(input_dir.glob("*.jsonl")):
        return

    raise FileNotFoundError(
        f"Não foi encontrado nenhum arquivo JSONL para iniciar a etapa "
        f"{start_step.value} em: {input_dir}"
    )


def _run_deduplication():
    """Executa os pipelines que produzem o output deduplicado."""
    if DEDUPLICATION_WORK_DIR.exists():
        shutil.rmtree(DEDUPLICATION_WORK_DIR)

    input_pipeline = create_deduplication_input_pipeline()
    input_results = input_pipeline.run()

    exact_workflow = create_exact_deduplication_workflow()
    exact_results = exact_workflow.run()

    removal_workflow = create_duplicate_removal_workflow()
    removal_results = removal_workflow.run()

    return (
        input_pipeline,
        input_results,
        exact_results,
        removal_results,
    )


def main(
    manifest_file: Path = DEFAULT_MANIFEST_FILE,
    resume_from_step: int = PipelineStep.PREPARE_SOURCES.value,
) -> int:
    try:
        start_step = PipelineStep(resume_from_step)
    except ValueError as error:
        valid_steps = ", ".join(str(step.value) for step in PipelineStep)
        raise ValueError(f"resume_from_step deve ser um de: {valid_steps}") from error

    _validate_resume_input(start_step)

    if start_step == PipelineStep.SEMANTIC_DEDUPLICATE:
        run_semantic_deduplication(DEDUPLICATED_DIR, SEMANTIC_DEDUPLICATED_DIR, SEMANTIC_WORK_DIR)
        print("Pipeline completed successfully!")
        return 0

    if start_step == PipelineStep.PREPARE_SOURCES:
        manifest_file = manifest_file.resolve()
        manifest = read_manifest_file(manifest_file)
        prepare_sources_for_processing(ORIGINAL_DIR, clean=True, manifest=manifest)

    with RayClient(include_dashboard=False):
        pipeline_error = None
        try:
            pipeline = None
            pipeline_results = None
            if start_step <= PipelineStep.WORD_FILTER:
                preprocessing_start = max(start_step, PipelineStep.NORMALIZE)
                pipeline = create_preprocessing_pipeline(preprocessing_start)
                pipeline_results = pipeline.run()

            (
                deduplication_input_pipeline,
                deduplication_input_results,
                exact_deduplication_results,
                duplicate_removal_results,
            ) = _run_deduplication()
        except BaseException as error:
            pipeline_error = error
            raise
        finally:
            try:
                ray.shutdown()
            except Exception as shutdown_error:
                if pipeline_error is None:
                    raise
                pipeline_error.add_note(
                    f"Falha adicional ao encerrar o Ray: {shutdown_error}"
                )

    deduplication_input_counts = get_stage_record_counts(
        deduplication_input_pipeline,
        deduplication_input_results,
    )
    deduplication_input_records = (
        deduplication_input_counts[-1][2] if deduplication_input_counts else 0
    )

    if pipeline is not None:
        print_stage_record_counts(pipeline, pipeline_results)
    print_stage_record_counts(
        deduplication_input_pipeline,
        deduplication_input_results,
    )
    print_exact_deduplication_summary(
        exact_deduplication_results,
        deduplication_input_records,
    )
    print_duplicate_removal_summary(
        duplicate_removal_results,
        deduplication_input_records,
    )

    run_semantic_deduplication(DEDUPLICATED_DIR, SEMANTIC_DEDUPLICATED_DIR, SEMANTIC_WORK_DIR)

    print(
        "Pipeline completed successfully! "
        f"Processed {len(deduplication_input_results or [])} tasks."
    )
    return 0


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(
        main(
            manifest_file=args.manifest_file,
            resume_from_step=args.resume_from_step,
        )
    )
