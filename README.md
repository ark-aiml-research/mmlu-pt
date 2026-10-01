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

The lockfile includes the CUDA 12 variant of NeMo Curator and pins
`vllm==0.15.1` on Linux x86-64. This is therefore the project's primary target
environment.

Install the dependencies from the repository root:

```bash
uv sync
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
uv run python -m mmlu_pt.pipeline_minimal \
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
uv run python -m mmlu_pt.pipeline_minimal --resume-from-step 5
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
CUDA_VISIBLE_DEVICES=0 uv run python -m mmlu_pt.pipeline_minimal --resume-from-step 6
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
└── 06 - semantic-deduplicated/
    ├── data.jsonl                  # final JSONL
    └── huggingface/                # local Hugging Face Dataset
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
