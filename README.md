# Portuguese-MMLU

Data curation pipeline for Portuguese-MMLU, a multiple-choice question
answering (MCQA) benchmark built from Brazilian exams. The repository turns
tabular sources into a consistent JSONL dataset, applies structural and length
validation, and removes duplicate questions with NVIDIA NeMo Curator and Octen embeddings.

[Project website](https://golabai.github.io/mmlu-pt/) ·
[Paper](https://openreview.net/pdf?id=2FpqhqcNnu) ·
[Datasets](https://huggingface.co/collections/mmlu-pt/mmlu-pt-dataset)

> The current scope of this repository is data curation. The source CSV files
> and model evaluation code are not included in this repository.

## Pipeline

```mermaid
flowchart LR
    A[YAML manifest] --> B[CSV or JSONL]
    B --> C[Conversion and metadata]
    C --> D[MCQA normalization]
    D --> E[Structural validation and image filtering]
    E --> F[Question: 4 to 1,000 words<br/>Choices: up to 300 words<br/>Combined: up to 1,000 words]
    F --> G[Question normalization]
    G --> H[Exact deduplication]
    H --> I[Question + labeled choices<br/>Octen embeddings + cosine similarity]
    I --> J[Direct representative matching]
    J --> K[Final JSONL + local Hugging Face Dataset]
```

The pipeline performs the following operations:

1. Reads the sources declared in the manifest and adds `exam` and
   `academic_level` to each record.
2. Renames `statement` to `question`, converts `alternatives` into `choices`,
   and maps answers `A`–`E` to zero-based integer indices `0`–`4`.
3. Discards records with invalid or empty alternatives, mismatched choice and
   label counts, or anything other than four or five choices. Also discards
   records containing images in the question or any choice: Markdown images
   (`![...](...)` or `![...][...]`), HTML `<img>` tags, or `data:image/` content,
   matched case-insensitively. Fenced and inline Markdown code is ignored;
   plain image URLs and textual mentions of images or figures are preserved.
4. Keeps questions containing between 4 and 1,000 words, whose combined
   choices contain at most 300 words, and whose question plus choices contain
   at most 1,000 words. All limits are inclusive and applied in that order.
5. Normalizes question text with Unicode NFKC, `casefold`, and whitespace
   collapsing, then uses the result as the exact-deduplication key.
6. Applies semantic deduplication to question plus ordered, labeled choices,
   then writes the final dataset with only the public fields.

Semantic deduplication uses `Octen/Octen-Embedding-8B` at revision
`5adcfa292e712091dfc30f0e97f0b2282e6cc66c` and cosine similarity ≥
`0.9737051129341125`, selected in the
[semantic ablation](ablations/deduplication/semantic/ablation.ipynb).
Text is normalized with Unicode NFKC and whitespace collapsing while preserving
case; alternatives retain their order and receive `[A]`, `[B]`, etc. labels.
The encoder receives the document prefix `- `. The answer is excluded.
Inference uses SentenceTransformers on CUDA, BF16, SDPA, left padding, batches
of 4, and a 32,768-token limit. Embeddings have 4,096 dimensions and are
converted to float32 before L2 normalization. Token counts and truncations are audited.

Representatives are processed by descending number of filled public fields,
descending semantic text length, and ascending stable ID (SHA-256 of canonical
public fields plus occurrence index). An item is removed only when its cosine
meets the threshold against an already preserved representative; the first
matching representative in that order wins. Comparisons are global in blocks
of at most 1,024 items, without clustering or a full corpus similarity matrix.
Transitive connections alone cannot cause removal. This differs from the NeMo
SemDeDup workflow explored in the notebook, whose references may also be removed.

Records that fail a filter are removed. At the end of a run, the pipeline
prints the input, output, and removal counts for every stage.

## Requirements

- Python 3.12 (`>=3.12,<3.13`);
- [uv](https://docs.astral.sh/uv/);
- an environment compatible with `nemo-curator[text_cuda12]==1.3.0`;
- an NVIDIA/CUDA GPU with enough memory for the 8B Octen encoder in BF16.

The `nemo-curator` dependency group includes CUDA 12 NeMo Curator and
`vllm==0.19.1`. The mutually exclusive `annotation` group uses `vllm==0.25.0`.
Both use the same `.venv`; select the group when synchronizing and running commands.

Install the dependencies from the repository root:

```bash
uv sync --group nemo-curator
```

## Configuring sources

The [`config/sources.yaml`](config/sources.yaml) manifest is validated with
Pydantic. It must define a root directory, at least one source, and metadata for
every source:

```yaml
source_root: /path/to/source/files

sources:
  - path: enem/enem_questions.csv
    exam: ENEM
    academic_level: high_school

  - path: enade/enade_questions.jsonl
    exam: ENADE
    academic_level: undergraduate
```

Paths under `sources` are resolved relative to `source_root`. The only accepted
values for `academic_level` are `high_school` and `undergraduate`. CSV files
are read as UTF-8 with BOM support (`utf-8-sig`); both `.csv` and `.jsonl`
sources are supported.

The versioned manifest lists 19 sources, but its `source_root` is an absolute
path from the development environment. Update it before running the pipeline
on another machine.

### Input record contract

| Field | Expected format | Purpose |
| --- | --- | --- |
| `statement` | Text | Prompt that becomes `question` |
| `alternatives` | Dictionary or serialized Python literal | Must contain `text` and `label` lists |
| `answer` | An uppercase letter from `A` to `E` | Converted to a zero-based index |
| `exam_edition` | Text | Exam edition identifier |
| `exam_url` | Text | URL of the source exam |
| `num` | Text or integer | Question number in the source exam |

Example `alternatives` value in a CSV file:

```text
{'text': ['Choice A', 'Choice B', 'Choice C', 'Choice D'], 'label': ['A', 'B', 'C', 'D']}
```

Manifest values override existing `exam` and `academic_level` fields in a
source.

## Running the pipeline

Run the pipeline from the repository root:

```bash
uv run --group nemo-curator python -m mmlu_pt.pipeline_minimal \
  --config config/sources.yaml
```

`--manifest-file` is an alias for `--config`. Starting from step 1 recreates
`output/01 - original/` before preparing the sources, preventing stale files
from being mixed with the sources declared in the current manifest.

The output location is fixed to `output/`, relative to the working directory.
The exact-deduplication workspace is recreated whenever step 5 runs.
Semantic audits use a fingerprint of the input files and encoder configuration, with
separate run directories. GPU selection defaults to `CUDA_VISIBLE_DEVICES=0`;
an existing environment setting is respected.

Use `--resume-from-step` to reuse a materialized output and rerun that step and
the following ones:

```bash
uv run --group nemo-curator python -m mmlu_pt.pipeline_minimal --resume-from-step 5
```

| Starting step | Reused input | First operation executed |
| --- | --- | --- |
| `1` | None | Prepare the sources |
| `2` | `output/01 - original/` | Normalize the records |
| `3` | `output/02 - read/` | Apply structural filters |
| `4` | `output/03 - pre-word-filter/` | Apply the question, choice, and combined word-count filters |
| `5` | `output/04 - filtered/` | Run exact deduplication |
| `6` | `output/05 - deduplicated/` | Run semantic deduplication |

The selected input directory must contain at least one JSONL file for steps
2–5. Step 6 also accepts an empty directory as an empty dataset. The source
manifest is only used when starting from step 1. Starting from step 2 reuses
`output/01 - original/` without preparing or cleaning it. Outputs from the
selected step onward are recreated normally.

To apply image filtering to existing normalized records, restart with
`--resume-from-step 3`. Starting from steps 4–6 reuses earlier outputs and does
not retroactively apply this filter.

To rerun only semantic deduplication on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group nemo-curator python -m mmlu_pt.pipeline_minimal --resume-from-step 6
```

Step 6 runs after the earlier Ray context has closed. It records every removal,
its preserved representative, and their cosine similarity. Embeddings are cached
by input file hashes, ordered record IDs, serialized texts, model revision,
parameters, and library versions. Only complete caches with matching identities,
checksums, dimensions, and normalized finite vectors are reused. Incomplete or
invalid caches are recomputed. Empty inputs publish an empty dataset without
loading the model.

The final JSONL is staged and validated before publication; previous semantic
outputs are retained in a `.semantic-backup-*` directory whose path is printed
after replacement. Failures preserve the previous published dataset. Historical
fuzzy outputs are not modified.

## Knowledge classification

An independent CLI assigns exam-restricted `subject` labels with
`Qwen/Qwen3.5-122B-A10B-FP8` through a local vLLM OpenAI-compatible API, then
derives `macro_area` from taxonomy 1.1. It uses two independent passes,
adjudication, 50 concurrent requests, SQLite checkpoints, audit manifests and
Hugging Face/Parquet/JSONL exports. The answer key never enters model prompts.
The Python job starts vLLM, waits for readiness and stops it on completion or interruption.

```bash
uv sync --group annotation
uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli inspect
# One job on the B200 inference host:
CUDA_VISIBLE_DEVICES=0 uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --run-dir output/knowledge-annotation-work/full
```

See [the annotation guide](src/mmlu_pt/annotation/knowledge_area/README.md)
for methodology, inference settings, dry run, resume, reports and manual review.
The Sauron job is defined by [knowledge_annotation.yaml](knowledge_annotation.yaml)
and [knowledge_annotation.sauron](knowledge_annotation.sauron).
Switch back to curating with `uv sync --group nemo-curator`.

## Outputs

```text
output/
├── 01 - original/                  # sources converted to JSONL
├── 02 - read/                      # normalized and auxiliary fields
├── 03 - pre-word-filter/           # after structural validation
├── 04 - filtered/                  # after all word-count filters
├── exact-deduplication-work/
│   ├── input/                      # materialized question_normalized field
│   └── results/
│       ├── ExactDuplicateIds/      # IDs identified as duplicates
│       └── exact_id_generator.json
├── 05 - deduplicated/              # exact-deduplicated dataset
├── semantic-deduplication-work/
│   └── <fingerprint>/
│       ├── embeddings/             # embeddings.npy and info.json (identity and token audit)
│       └── run-*/                  # manifest.json, records.jsonl, removals.jsonl
├── 06 - semantic-deduplicated/
│   ├── data.jsonl                  # final JSONL
│   └── huggingface/                # local Hugging Face Dataset
└── knowledge-annotation-work/
    ├── inspection/                # schema and taxonomy validation
    └── <run>/                     # source snapshot, checkpoints, server logs, exports and reports
```

NeMo Curator partitions intermediate data and may produce hash-based file names.
The final file `output/06 - semantic-deduplicated/data.jsonl` uses the following schema:

| Field | Description |
| --- | --- |
| `exam` | Exam name provided by the manifest |
| `exam_edition` | Exam edition |
| `exam_url` | URL of the source exam |
| `num` | Original question number |
| `question` | Question prompt |
| `choices` | List containing four or five choices |
| `answer` | Zero-based integer index of the correct answer |
| `academic_level` | `high_school` or `undergraduate` |

The pipeline also saves a single Hugging Face `Dataset`, without splits, in
`output/06 - semantic-deduplicated/huggingface/`. It contains the same records
in the same order as the final JSONL, with only the eight public fields.
`answer` is an integer, `choices` is a list of strings, and the remaining
fields are strings; numeric identifiers are converted to text and null values
are preserved. Empty results retain the same schema.

This export runs both in the full pipeline and with `--resume-from-step 6`.
It uses `save_to_disk` locally, without authentication or upload to the Hub.
The JSONL and Hugging Face dataset are staged together before replacement,
so an export failure preserves the previous published output.

Load the saved dataset with:

```python
from datasets import load_from_disk

dataset = load_from_disk("output/06 - semantic-deduplicated/huggingface")
```

The intermediate directories make each transformation inspectable. They and
the final output directory are ignored by Git.

## Question-length analysis

The
[`ablations/filtering/question_length_ablation.ipynb`](ablations/filtering/question_length_ablation.ipynb)
notebook evaluates retention by exam and academic level across different
word-count thresholds. It reads `output/03 - pre-word-filter/` and selects
inclusive limits of 4 to 1,000 words for the question text.

```bash
uv sync --group notebook
uv run jupyter lab ablations/filtering/question_length_ablation.ipynb
```

## Choice-length analysis

The
[`ablations/filtering/choices_length_ablation.ipynb`](ablations/filtering/choices_length_ablation.ipynb)
notebook evaluates maximum thresholds for the total number of words across all
choices. It reads `output/03 - pre-word-filter/`, reports the incremental effect
after the current question-length filter, and recommends an inclusive limit of
300 words.

```bash
uv sync --group notebook
uv run jupyter lab ablations/filtering/choices_length_ablation.ipynb
```

## Question-plus-choices length analysis

The
[`ablations/filtering/question_choices_length_ablation.ipynb`](ablations/filtering/question_choices_length_ablation.ipynb)
notebook evaluates an additional maximum threshold on the combined word count
of each question and its choices. It applies the existing inclusive question
and choice limits first, then reports the incremental effect of candidate total
limits and selects an inclusive combined limit of 1,000 words.

```bash
uv sync --group notebook
uv run jupyter lab ablations/filtering/question_choices_length_ablation.ipynb
```

## Semantic-deduplication exploration

The [semantic-deduplication notebook](ablations/deduplication/semantic/ablation.ipynb)
compares BGE-M3, GTE-multilingual-base, Octen, Qwen3, and NVIDIA Nemotron embeddings on the 466 pairs in
`ablations/deduplication/assets/annotations/manual_labels_llm_annotated.csv`. It contrasts question-only
and question-plus-choices embeddings, audits truncation, and selects a model and
cosine threshold using nested grouped cross-validation with MCC. Explanations
and literature references are in English; original exam questions remain in Portuguese.

```bash
uv sync --group notebook
CUDA_VISIBLE_DEVICES=0 uv run --group notebook jupyter lab ablations/deduplication/semantic/ablation.ipynb
```

`RUN_COMPUTE=True` downloads the models and computes missing caches on one NVIDIA
GPU. Nemotron uses a temporary Transformers 4.51.0/tokenizers 0.21.4 runtime
through `ablations/deduplication/semantic/nemotron_embeddings.py`, preserving the project environment.
The full experiment targets a B200; batch sizes are configurable. Set
`RUN_COMPUTE=False` to reuse complete caches without initializing CUDA. Results,
model revisions, folds, figures, and manifests are saved under
`ablations/deduplication/semantic/output/ablation/`, keyed by the input and configuration.

The notebook also runs NeMo Curator 1.3.0 semantic deduplication with 1, 8, and 32
clusters on the annotated items. It distinguishes pair classification from item
removal, and audits references to items that are themselves marked for removal.
The labels and production corpus are never modified. Results describe a
lexically enriched, LLM-labeled sample rather than corpus-wide performance.

## TS-Guessing contamination study

[`ablations/contamination/ts_guessing/`](ablations/contamination/ts_guessing/) samples
10% of each exam, rounded up, without replacement, using seed 42 and stable hashes.
The default input is the knowledge-annotated dataset
[`bench-temp/mmlu-pt-knowledge-annotated`](https://huggingface.co/datasets/bench-temp/mmlu-pt-knowledge-annotated),
configuration `default`, split `train`, revision `9a578d25fb50b093b71591a19ad67ec3e8c1b37c`.
Existing fields are consumed directly, without MCQA or classification prevalidation.
Every incorrect alternative is masked separately. Portuguese prompts request its literal
text, with and without exam/edition/question-number hints. Area annotations are retained
for analysis and excluded from prompts. All models receive the same sample and tasks.

[`config.json`](ablations/contamination/ts_guessing/config.json) contains the 19 open-weight
checkpoints in the Direct matrix of [`docs/experimental_evaluation.md`](docs/experimental_evaluation.md#21-matriz-direct-e-elegibilidade-para-estudos-posteriores),
all below 40B: Qwen3.5 (0.8B, 2B, 4B, 9B, 27B, 35B-A3B), Llama (1B, 3B, 8B),
Tucano2 (0.5B, 1.5B, 3.7B), Gemma 4 (E2B, E4B, 12B, 26B-A4B, 31B),
and Nemotron 3 Nano (4B BF16, 30B-A3B BF16). Each entry has identifiers and necessary
runtime overrides. Family/nominal/active/effective parameter metadata is snapshotted from
the Direct table and its PLE footnote. Stored parameter totals come from Hugging Face.
Each model's `metadata.json` fixes its weight, tokenizer and remote-code revisions on first
use; subsequent attempts reuse those pins. Gated checkpoints require existing access.

Download the configured checkpoints before generation:

```bash
uv run --no-project --with 'huggingface-hub>=0.36,<2' python \
  ablations/contamination/ts_guessing/download_models.py
```

The downloader reads `models` from the adjacent `config.json` and calls
`snapshot_download` for each repository, using the default Hugging Face cache.
The Sauron job in `download_models.sauron` sets `HF_HUB_CACHE` to
`$PWD/output/huggingface/hub` to avoid permission conflicts in the shared cache.
Set the same variable when running generation to reuse the downloaded weights.

The generator uses **vLLM 0.25.0 / Transformers 5.14.1** in an isolated `uv` environment.
The Transformers pin avoids the incompatible Gemma 4 per-layer configuration access;
see the [upstream issue](https://github.com/vllm-project/vllm/issues/51744).
The classification environment remains separate. Models run sequentially through
`LLM.generate()` with official templates and token IDs, tokenized once per task attempt.
There are no HTTP inference calls. Each launch's processes are cleaned up before the next
model, including on failures and interruptions.

Run from the repository root:

```bash
uv sync
CUDA_VISIBLE_DEVICES=0 uv run python ablations/contamination/ts_guessing/generate_results.py

# Four replicas, 128 tasks per batch and at most 50 active sequences per replica.
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python ablations/contamination/ts_guessing/generate_results.py \
  --data-parallel-size 4 --batch-size 128 --concurrency 50

# Two replicas, each using two GPUs with tensor parallelism.
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python ablations/contamination/ts_guessing/generate_results.py \
  --data-parallel-size 2 --tensor-parallel-size 2

# Explicit local annotated input and model subset.
uv run python ablations/contamination/ts_guessing/generate_results.py \
  --input output/knowledge-classification/dataset --models tucano2-0.5b qwen3.5-2b
```

Local input accepts a saved Hugging Face Dataset/DatasetDict, Parquet or JSONL.
`--config` selects a configuration and `--output-dir` selects the artifact root;
relative paths resolve from the repository root. Selecting models runs a subset of the
recorded full panel, which can be completed with later invocations.

| Setting / CLI flag | Default | Meaning |
|---|---:|---|
| `data_parallel_size` / `--data-parallel-size` | 1 | Local replicas/ranks |
| `tensor_parallel_size` / `--tensor-parallel-size` | 1 | GPUs per replica/rank |
| `concurrency` / `--concurrency` | 50 | Engine `max_num_seqs` per replica |
| `batch_size` / `--batch-size` | 128 | Tasks submitted together per replica |

The historical configuration key `server` contains these offline settings. The four
settings and `max_model_len` accept per-model overrides; CLI flags override their
corresponding global and model values. `DP × TP` GPUs must be visible through
`CUDA_VISIBLE_DEVICES` or a per-model `cuda_visible_devices` string. Dense models use
independent replicas; MoE models use synchronized native DP without expert parallelism.
Empty MoE ranks participate through one-token auxiliary requests, excluded from response
and token counts and all reconstruction metrics. Their overhead remains in runtime cost.
The partitioning, rounds and scheduler settings are preserved by the simplification.

Defaults are BF16 without quantization, greedy decoding, 8,192 context tokens and up to
512 output tokens. Tucano uses its native 4,096 context limit, reserving 512 for output.
Oversized prompts are recorded as `context_overflow` and never truncated. Qwen/Gemma
use text-only mode and `enable_thinking=false`; Nemotron uses that thinking control,
remote code pinned to the checkpoint revision and an FP32 Mamba SSM cache with BF16 weights.
HTTP-only configuration options produce an explanatory error.

The `tqdm` bar advances after the sole response writer persists each rank's batch and
confirms it to the rank. It starts at previously completed tasks and context overflows,
shows response/error/context counts, and excludes engine startup from its processing ETA.
Failed tasks remain pending for future attempts. A 100% bar reports tasks processed in
that pass and can include operational errors. Known transient failures get up to three
retries; OOM, access, configuration and incompatibility errors are not automatically retried.
Initialization and batch inactivity timeouts default to 1,800 seconds.

Version-2 artifacts are stored under `ablations/contamination/ts_guessing/output/<fingerprint>/`:

```text
manifest.json                 # data source, counts, versions, code hashes and panel
sample.jsonl                  # original annotated questions, IDs and source row positions
tasks.jsonl                   # prompts, masks, targets and predictability diagnostics
operations.jsonl              # isolated runtime preparation
models/<model>/
    metadata.json             # authoritative checkpoint pins and resolved engine settings
    responses.jsonl           # append-only responses, tokens, native output and errors
    operations.jsonl          # launch/execution/initialization/batch/error/attempt/session
    rank-0000.log             # vLLM log per rank
```

Responses are flushed and synced before a batch is acknowledged. A temporary file
communicates each attempt's outcome to the controller and is removed afterwards.
Repeating the same command resumes compatible pending tasks. Data, source order,
configuration, library or generator/worker/artifact-helper changes create a new identity.
Question and task IDs retain their original hash formulas. Source row positions are
zero-based before sampling; for local files they refer to the concatenated input in sorted
file order. The position and record hash must be interpreted against the recorded snapshot.

After generation, open [`analysis.ipynb`](ablations/contamination/ts_guessing/analysis.ipynb):

```bash
uv sync --group notebook
uv run --group notebook jupyter lab ablations/contamination/ts_guessing/analysis.ipynb
```

Set `RUN_DIR` to the printed directory; it is selected automatically only when there is
exactly one version-2 run. `ANALYSIS_MODELS=None` uses the recorded full panel; an explicit
list selects the analysis panel. The notebook accepts only the new format. It performs
local analysis and exports tables/JSONL and PNG/PDF figures under `<fingerprint>/analysis/`.

It retains strict/normalized/label-tolerant EM, token F1, output diagnostics, question-first
aggregation, exam/subject/macro-area coverage, macro and population weighting, paired
contrasts, 2,000 exam-stratified bootstrap draws, length/predictability diagnostics and costs.
Failures and missing generations are not assigned incorrect-answer scores.

Question review exports are:

- `panel.json`: exact selected models, families and scoring rule.
- `question_catalog.jsonl`: original questions, alternatives, annotations, source/revision,
  snapshot/order/record hashes, row positions and question IDs.
- `question_evidence.csv/.jsonl`: each mask's target, raw response, status, metrics and
  diagnostics, linked to its question and model/condition.
- `question_model_scores.csv` and `question_family_scores.csv`: component scores and coverage.
- `question_review_coverage.csv/.jsonl`: every sampled question, eligibility and family scores.
- `question_ranking.csv/.jsonl`: eligible questions ordered by equal-family reconstruction score.

The consensus averages distractor normalized EMs per question/model/prompt, the two prompts
per model, checkpoints within each family, then families with equal weights. Ranking requires
all masks in both prompts for **every selected model**. Incomplete items retain coverage but
no consensus score/rank; incomplete families also have no score. Ties use `question_id`.
Predictable short/numeric/context-visible targets are reported alongside the ranking.
The files provide traceability for later review and removal tooling. No automatic threshold,
contamination verdict or dataset deletion is implemented.

Costs use processing wall time per attempt for tasks/s and tokens/s. Rank phase durations
are derived from batch records; their sums can exceed wall time with DP. Session wall time
includes startup, shutdown and retries. Batch duration is not individual task latency.
GPU-hours are estimates from session wall time times DP × TP, not measured utilization.

This refactor is checked statically only. Generation, model loading and notebook execution
are left to the user; actual throughput equivalence requires a GPU run under the same
conditions. Existing classification code and jobs are not modified or interrupted.

## LLM-as-a-Judge annotation

The annotation CLI classifies candidate question pairs as `duplicate`,
`related_but_distinct`, `distinct`, or `unsure`. It supports immediate,
row-by-row requests through the Responses API and asynchronous processing
through the Batch API. Install its dependencies and configure the API key:

```bash
uv sync --group annotation
export OPENAI_API_KEY="..."
```

Use synchronous mode for small tests or when results are needed immediately.
`--limit` selects the first eligible rows after already labeled rows are
discarded:

```bash
uv run --group annotation python ablations/deduplication/annotation/llm_judge.py run \
  ablations/deduplication/assets/annotations/manual_labels_blind.csv \
  --mode sync \
  --limit 10
```

Synchronous responses are persisted as they arrive. An interrupted execution
can continue from its checkpoint with the same arguments plus `--resume`.

Use Batch mode for larger offline runs:

```bash
uv run --group annotation python ablations/deduplication/annotation/llm_judge.py run \
  ablations/deduplication/assets/annotations/manual_labels_blind.csv \
  --mode batch
```

The Batch workflow can also be controlled step by step:

```bash
uv run --group annotation python ablations/deduplication/annotation/llm_judge.py prepare \
  ablations/deduplication/assets/annotations/manual_labels_blind.csv

uv run --group annotation python ablations/deduplication/annotation/llm_judge.py submit \
  ablations/deduplication/assets/annotations/manual_labels_blind.judge-requests.jsonl

uv run --group annotation python ablations/deduplication/annotation/llm_judge.py status \
  ablations/deduplication/assets/annotations/manual_labels_blind.batch-state.json

uv run --group annotation python ablations/deduplication/annotation/llm_judge.py collect \
  ablations/deduplication/assets/annotations/manual_labels_blind.csv
```

By default, the CLI writes the applicable files next to the input CSV:

```text
<stem>.judge-requests.jsonl
<stem>.judge-requests.meta.json
<stem>.batch-state.json
<stem>.sync-state.json
<stem>.responses.jsonl
<stem>.errors.jsonl
<stem>.judged.csv
```

Output paths, model, and reasoning effort can be changed through command-line
options. Existing labels are skipped unless `--include-labeled` is used. The
judge instructions live in
[`ablations/deduplication/assets/prompts/question_pair_deduplication.md`](ablations/deduplication/assets/prompts/question_pair_deduplication.md),
and their SHA-256 hash is recorded with each execution.

## Code structure

```text
src/mmlu_pt/
├── classification/                # subject inference, taxonomy, checkpoints and CLI
├── pipeline_minimal.py             # CLI arguments and orchestration
├── pipelines/
│   ├── definition.py               # NeMo Curator stages and workflows
│   ├── semantic.py                 # Octen embeddings and direct representatives
│   └── fuzzy.py                    # historical fuzzy baseline for ablations
├── mcqa_minimal.py                 # parsing, normalization, and predicates
└── utils/
    ├── csv.py                      # CSV/JSONL source preparation
    ├── manifest.py                 # manifest schema and loading
    └── pipeline_utils.py           # metrics and stage integration
```

Notebooks are grouped by experiment, with shared annotation tools and assets:

```text
ablations/
├── filtering/
│   ├── question_length_ablation.ipynb
│   ├── choices_length_ablation.ipynb
│   └── question_choices_length_ablation.ipynb
└── deduplication/
    ├── fuzzy/
    │   ├── ablation.ipynb
    │   ├── parameter_search.ipynb
    │   ├── pareto_audit.ipynb
    │   └── output/
    │       ├── .gitignore
    │       ├── ablation/
    │       ├── parameter_search/
    │       └── pairwise_benchmark/
    ├── semantic/
    │   ├── ablation.ipynb
    │   ├── nemotron_embeddings.py
    │   └── output/
    │       ├── .gitignore
    │       └── ablation/
    ├── annotation/
    │   ├── human_llm_agreement.ipynb
    │   ├── llm_judge.py
    │   └── llm_as_a_judge/         # CLI, shared logic, sync and Batch transports
    └── assets/                    # annotations, prompts, and source snapshots
```

## Current limitations

- Raw source files must be obtained separately.
- The `output/` directory cannot yet be configured through the command line.
- Exact deduplication still uses the question alone, so identical prompts with
  different choices can be removed before the semantic stage.
- The semantic threshold was fitted on a lexically enriched, LLM-labeled panel;
  it does not guarantee a minimum removal precision across the full corpus.
- Global semantic comparisons have quadratic worst-case computation, although
  similarity matrices are bounded by the block size.
- Answers are validated as `A`–`E`, but their indices are not checked against
  the number of choices in each record.
- The repository does not yet include an automated test suite.

## License

The code is distributed under the [Apache License 2.0](LICENSE). Review the
terms of the original sources and published datasets before redistributing the
data.
