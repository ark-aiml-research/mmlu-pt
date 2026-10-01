from enum import IntEnum
from pathlib import Path

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
from nemo_curator.stages.text.deduplication.removal_workflow import (
    TextDuplicatesRemovalWorkflow,
)
from nemo_curator.stages.text.filters import Filter, ScoreFilter
from nemo_curator.stages.text.filters.heuristic import WordCountFilter
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modifiers import Modify

from mmlu_pt.mcqa_minimal import (
    PUBLIC_FIELDS,
    extract_choices,
    group_answer_and_choices,
    group_question_and_choices,
    has_answer_in_bounds,
    has_described_choices,
    has_matching_alternative_lengths,
    has_no_images,
    has_supported_choice_count,
    keep_question,
    normalize_answer,
    normalize_question_for_dedup,
    parse_alternatives,
    serialize_choices,
    serialize_question_and_choices,
)
from mmlu_pt.utils.pipeline_utils import IntermediateJsonlReader

OUTPUT_DIR = Path("output").resolve()
ORIGINAL_DIR = OUTPUT_DIR / "01 - original"
READ_DIR = OUTPUT_DIR / "02 - read"
PRE_WORD_FILTER_DIR = OUTPUT_DIR / "03 - pre-word-filter"
FILTERED_DIR = OUTPUT_DIR / "04 - filtered"
DEDUPLICATION_WORK_DIR = OUTPUT_DIR / "exact-deduplication-work"
DEDUPLICATION_INPUT_DIR = DEDUPLICATION_WORK_DIR / "input"
DEDUPLICATION_RESULTS_DIR = DEDUPLICATION_WORK_DIR / "results"
EXACT_DUPLICATE_IDS_DIR = DEDUPLICATION_RESULTS_DIR / "ExactDuplicateIds"
EXACT_ID_GENERATOR_PATH = DEDUPLICATION_RESULTS_DIR / "exact_id_generator.json"
DEDUPLICATED_DIR = OUTPUT_DIR / "05 - deduplicated"
SEMANTIC_DEDUPLICATED_DIR = OUTPUT_DIR / "06 - semantic-deduplicated"
SEMANTIC_WORK_DIR = OUTPUT_DIR / "semantic-deduplication-work"
NORMALIZED_QUESTION_FIELD = "question_normalized"
ANSWER_AND_CHOICES_FIELD = "answer_and_choices"
QUESTION_AND_CHOICES_FIELD = "question_and_choices"
SERIALIZED_CHOICES_FIELD = "choices_serialized"
SERIALIZED_QUESTION_AND_CHOICES_FIELD = "question_and_choices_serialized"
DEDUPLICATION_FIELDS = [*PUBLIC_FIELDS, NORMALIZED_QUESTION_FIELD]
DEDUPLICATION_INPUT_BLOCKSIZE = "256MiB"


class PipelineStep(IntEnum):
    PREPARE_SOURCES = 1
    NORMALIZE = 2
    STRUCTURAL_FILTER = 3
    WORD_FILTER = 4
    DEDUPLICATE = 5
    SEMANTIC_DEDUPLICATE = 6


def _normalization_stages() -> list:
    return [
        Modify(keep_question, input_fields="statement", output_fields="question"),
        Modify(
            parse_alternatives,
            input_fields="alternatives",
            output_fields="parsed",
        ),
        Modify(normalize_answer, input_fields="answer", output_fields="answer"),
        Modify(extract_choices, input_fields="parsed", output_fields="choices"),
        JsonlWriter(
            path=str(READ_DIR),
            write_kwargs={"index": False},
            mode="overwrite",
        ),
    ]


def _structural_filter_stages() -> list:
    return [
        Filter(
            has_matching_alternative_lengths,
            filter_field="parsed",
        ),
        Filter(
            has_described_choices,
            filter_field="choices",
        ),
        Filter(
            has_supported_choice_count,
            filter_field="choices",
        ),
        Modify(
            group_answer_and_choices,
            input_fields=[["answer", "choices"]],
            output_fields=ANSWER_AND_CHOICES_FIELD,
        ),
        Filter(
            has_answer_in_bounds,
            filter_field=ANSWER_AND_CHOICES_FIELD,
        ),
        Modify(
            group_question_and_choices,
            input_fields=[["question", "choices"]],
            output_fields=QUESTION_AND_CHOICES_FIELD,
        ),
        Filter(
            has_no_images,
            filter_field=QUESTION_AND_CHOICES_FIELD,
        ),
        JsonlWriter(
            path=str(PRE_WORD_FILTER_DIR),
            fields=PUBLIC_FIELDS,
            write_kwargs={"index": False},
            mode="overwrite",
        ),
    ]


def _word_filter_stages() -> list:
    return [
        Modify(
            serialize_choices,
            input_fields="choices",
            output_fields=SERIALIZED_CHOICES_FIELD,
        ),
        ScoreFilter(
            filter_obj=WordCountFilter(
                min_words=4,
                max_words=1_000,
                lang="pt",
            ),
            text_field="question",
        ),
        ScoreFilter(
            filter_obj=WordCountFilter(
                min_words=0,
                max_words=300,
                lang="pt",
            ),
            text_field=SERIALIZED_CHOICES_FIELD,
        ),
        Modify(
            serialize_question_and_choices,
            input_fields=[["question", "choices"]],
            output_fields=SERIALIZED_QUESTION_AND_CHOICES_FIELD,
        ),
        ScoreFilter(
            filter_obj=WordCountFilter(
                min_words=0,
                max_words=1_000,
                lang="pt",
            ),
            text_field=SERIALIZED_QUESTION_AND_CHOICES_FIELD,
        ),
        JsonlWriter(
            path=str(FILTERED_DIR),
            fields=PUBLIC_FIELDS,
            write_kwargs={"index": False},
            mode="overwrite",
        ),
    ]


def create_preprocessing_pipeline(start_step: PipelineStep) -> Pipeline:
    """Cria o pipeline de pré-processamento a partir de uma etapa materializada."""
    if start_step not in {
        PipelineStep.NORMALIZE,
        PipelineStep.STRUCTURAL_FILTER,
        PipelineStep.WORD_FILTER,
    }:
        raise ValueError("O pré-processamento deve começar nas etapas 2, 3 ou 4")

    stages = []
    if start_step == PipelineStep.NORMALIZE:
        stages.append(JsonlReader(file_paths=str(ORIGINAL_DIR)))
        stages.extend(_normalization_stages())

    if start_step <= PipelineStep.STRUCTURAL_FILTER:
        reader = (
            IntermediateJsonlReader()
            if start_step == PipelineStep.NORMALIZE
            else JsonlReader(file_paths=str(READ_DIR))
        )
        stages.append(reader)
        stages.extend(_structural_filter_stages())

    if start_step <= PipelineStep.WORD_FILTER:
        reader = (
            IntermediateJsonlReader()
            if start_step <= PipelineStep.STRUCTURAL_FILTER
            else JsonlReader(file_paths=str(PRE_WORD_FILTER_DIR))
        )
        stages.append(reader)
        stages.extend(_word_filter_stages())

    return Pipeline(
        name="mmlu_pt",
        description="Read, normalize and filter the MMLU-PT dataset",
        stages=stages,
    )


def create_deduplication_input_pipeline() -> Pipeline:
    """Cria o pipeline que materializa a pergunta normalizada."""
    return Pipeline(
        name="mmlu_pt_exact_deduplication_input",
        description="Normalize questions for exact deduplication",
        stages=[
            JsonlReader(file_paths=str(FILTERED_DIR)),
            Modify(
                normalize_question_for_dedup,
                input_fields="question",
                output_fields=NORMALIZED_QUESTION_FIELD,
            ),
            JsonlWriter(
                path=str(DEDUPLICATION_INPUT_DIR),
                fields=DEDUPLICATION_FIELDS,
                write_kwargs={"index": False},
                mode="overwrite",
            ),
        ],
    )


def create_exact_deduplication_workflow() -> ExactDeduplicationWorkflow:
    """Cria o workflow que identifica perguntas normalizadas duplicadas."""
    return ExactDeduplicationWorkflow(
        input_path=str(DEDUPLICATION_INPUT_DIR),
        output_path=str(DEDUPLICATION_RESULTS_DIR),
        text_field=NORMALIZED_QUESTION_FIELD,
        input_filetype="jsonl",
        input_blocksize=DEDUPLICATION_INPUT_BLOCKSIZE,
        perform_removal=False,
        assign_id=True,
    )


def create_duplicate_removal_workflow() -> TextDuplicatesRemovalWorkflow:
    """Cria o workflow que remove os IDs de perguntas duplicadas."""
    return TextDuplicatesRemovalWorkflow(
        input_path=str(DEDUPLICATION_INPUT_DIR),
        ids_to_remove_path=str(EXACT_DUPLICATE_IDS_DIR),
        output_path=str(DEDUPLICATED_DIR),
        input_filetype="jsonl",
        input_blocksize=DEDUPLICATION_INPUT_BLOCKSIZE,
        id_field="_curator_dedup_id",
        duplicate_id_field="_curator_dedup_id",
        id_generator_path=str(EXACT_ID_GENERATOR_PATH),
        output_filetype="jsonl",
        output_kwargs={"index": False},
        output_fields=PUBLIC_FIELDS,
        output_mode="overwrite",
    )
