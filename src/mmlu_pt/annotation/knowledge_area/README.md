# Reproducible academic subject annotation

This CLI annotates `bench-temp/mmlu-pt-filtered` through a local OpenAI-compatible
server. The source `answer` is preserved in exports but never passed to the model.
Questions and **all choices** determine the required discipline. Candidates and
their definitions come directly from the authoritative taxonomy. `macro_area`
is derived from `subject`, never generated.

## Installation: one environment, two exclusive groups

Python 3.12 and uv are required. Run commands from the repository root.
All default artifacts are written inside `output/knowledge-annotation-work/`:
inspection uses `inspection/`, and annotation uses a separate `run-<hash>/`
or `dry-run-<hash>/` directory. The examples below name these run directories
explicitly. Reports, review exports, checkpoints and server logs stay within
the selected run directory.

```bash
# Curating: NeMo Curator 1.3.0 + vLLM 0.19.1
uv sync --locked --group nemo-curator

# Switch the SAME .venv to annotation: vLLM 0.25.0
uv sync --locked --group annotation
```

`default-groups=[]` makes group selection explicit. Do not select both groups.
`uv sync` removes dependencies exclusive to the previous selection. Use
`uv run --group annotation ...` for annotation, and `--group nemo-curator` for
curating. The lockfile resolves each group's transitives independently.

## Verified source and taxonomy

Inspection on 2026-10-06 loaded revision
`bef2dcfb3e3ca6d0add3cf374a4c0421927af680` with `datasets.load_dataset`:
41,653 rows in `train`; strings `exam`, `exam_edition`, `exam_url`, `num`,
`question`, `academic_level`; list-of-string `choices`; integer `answer`.
There were no null fields or duplicate identifying-content hashes.

| Exam | Rows | Allowed subjects |
|---|---:|---:|
| AFA | 1,085 | 4 |
| BACEN | 1,178 | 18 |
| BLUEX | 663 | 9 |
| BNDES | 4,048 | 29 |
| CFCES | 707 | 11 |
| CNU | 491 | 50 |
| COMVEST | 452 | 11 |
| ENADE | 10,785 | 70 |
| ENAM | 314 | 11 |
| ENEM | 2,688 | 15 |
| FUVEST | 1,659 | 13 |
| IME | 958 | 5 |
| OAB | 10,251 | 16 |
| OBI | 2,301 | 3 |
| POSCOMP | 1,304 | 9 |
| RESIDENCIA_USP_UNICAMP | 1,574 | 5 |
| REVALIDA | 1,195 | 5 |

These are observations, not hard-coded validation assumptions. Every invocation
discovers the schema and validates the entire selected split before inference.

The default `mmlu_pt_taxonomy_v1_1.json` contains 70 subjects in 11 macro areas.
Version 1.0 lacks ENEM and IME and therefore cannot cover the dataset. The only
alias is explicitly recorded in `exam_aliases.json`:
`RESIDENCIA_USP_UNICAMP` → `USP_UNICAMP_MEDICAL_RESIDENCY`.
Unknown exams fail preflight and are all listed; there is no fuzzy matching.

```bash
uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli inspect

# Pin the inspected revision when reproducing it:
uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli inspect \
  --dataset-revision bef2dcfb3e3ca6d0add3cf374a4c0421927af680
```

Inspection writes `output/knowledge-annotation-work/inspection/inspection.json` and
never contacts an inference endpoint. `--dataset-path PATH` accepts a local
`save_to_disk` Dataset or DatasetDict. `--dataset`, `--dataset-config`,
`--dataset-revision` and `--split` configure Hub loading. `train` is the default
split, validated against the splits actually present. All original fields,
including additional columns in later revisions, are preserved.

## One Python job on the B200 inference host

Run this job on the host with one B200:

```bash
uv sync --locked --group annotation
CUDA_VISIBLE_DEVICES=0 uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --run-dir output/knowledge-annotation-work/full \
  --concurrency 50 --max-num-seqs 50 \
  --max-model-len 16384 --gpu-memory-utilization 0.90
```

`annotate` defaults to `--server-mode managed`. After dataset/taxonomy validation,
Python resolves an immutable model revision, records the launch configuration in
`manifest.json`, and starts vLLM as a supervised subprocess using the same Python
interpreter and `.venv`. No launch script or second terminal is required. The
annotation client remains an independent `AsyncOpenAI` client; vLLM owns batching.

The job waits for `/health` and `/v1/models` to confirm the expected model before
sending annotations. Startup has a configurable 1,800-second deadline. Server
stdout/stderr go to `vllm.log`; lifecycle status goes to `server-session.json`.
The job monitors unexpected server exits, cancels in-flight work and leaves the
checkpoint resumable. Completion, startup failure and SIGINT/SIGTERM stop the
owned process group; shutdown escalates to SIGKILL after 30 seconds. The server
is stopped before exports and reports are built. A fully completed resume skips
model startup entirely.

Managed launch uses one GPU, `--language-model-only`, the `qwen3` reasoning parser,
`--generation-config vllm`, xgrammar for final structured output, and eager
execution. Prefix caching and request/output logging are disabled. Both model
and tokenizer use the resolved revision. `CUDA_VISIBLE_DEVICES` defaults to `0`.

The Python entrypoint was checked against the [vLLM 0.25.0 API server source](https://github.com/vllm-project/vllm/blob/v0.25.0/vllm/entrypoints/openai/api_server.py).
Flags were checked against the [vLLM 0.25.0 CLI reference](https://docs.vllm.ai/en/v0.25.0/cli/serve/)
and [structured-output documentation](https://docs.vllm.ai/en/v0.25.0/features/structured_outputs/).
The Qwen parser separates thinking from final content; grammar enforcement is
reserved for final output. Client requests set `enable_thinking` explicitly.

The settings reduce context, prefill work and CUDA-graph overhead while keeping
50 active sequences available. GPU memory fit and performance have not been
measured on this development host. vLLM schedules tokens and may preempt
sequences according to available cache; 50 pending client requests does not
reserve 50 full contexts in advance. All limits are configurable in the same job:
`--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`,
`--max-num-batched-tokens` (default 4096), `--no-enforce-eager`,
`--server-startup-timeout`, `--server-shutdown-timeout` and
`--server-poll-interval` (default one second). Environment equivalents use
`MMLU_ANNOTATION_SERVER_<FIELD>`, for example
`MMLU_ANNOTATION_SERVER_MAX_MODEL_LEN` or `MMLU_ANNOTATION_SERVER_STARTUP_TIMEOUT`.
Use `--model-revision` to request a specific Hub revision.

The default endpoint is `http://localhost:8000/v1`. Managed mode requires a
loopback HTTP address with an unused port; use `--endpoint http://127.0.0.1:8001/v1`
for another local port. It bypasses HTTP proxy environment settings. For an
already running server, explicitly select `--server-mode external`; that mode
never starts or stops a server. Its optional `--server-manifest PATH` records
externally supplied model/revision metadata. The Python `annotate()` API accepts
`server_config=ServerConfig()` for managed operation and defaults to external
operation for embedding.

## Dry run, full annotation and resume

Run a stratified dry run first; it manages its own server:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --dry-run --samples-per-exam 5 \
  --run-dir output/knowledge-annotation-work/dry-run
```

The entire split is validated, then up to five questions per exam are chosen by
a stable seeded hash. Original row positions remain intact. A few safe prompts
and validated results are printed. The sample gets the same checkpointing,
adjudication, exports and QC as a full run. Dry and full runs cannot share a run
directory or be pushed interchangeably.

```bash
# Full annotation
CUDA_VISIBLE_DEVICES=0 uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --run-dir output/knowledge-annotation-work/full

# Resume after interruption: repeat exactly the same command
CUDA_VISIBLE_DEVICES=0 uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --run-dir output/knowledge-annotation-work/full
```

Completed requests are skipped, interrupted work is recovered, and failed tasks
are retried with a bounded budget in the next invocation. Resume uses the verified
local source snapshot without reloading the dataset from Hugging Face. Managed
resume also reuses the immutable model revision instead of resolving a moving
Hub branch again; vLLM loads those weights from its cache or downloads them if
needed. Both source and model revisions remain recorded in the manifest.

SQLite uses WAL, `synchronous=FULL`, one event-loop writer, transaction-per-attempt
durability and an exclusive process lock. Every attempt and seed is committed
before its HTTP request; interrupted attempts remain auditable. Use a local filesystem, not a network
filesystem. The queue is limited to twice concurrency, with fixed workers and
a semaphore. Progress advances only after results or terminal errors persist.
SIGINT/SIGTERM cancels workers and closes resources. A lost in-flight response
may be repeated, but persisted successful work is not duplicated.

Changing source, taxonomy, aliases, methodology, server launch settings or annotation
code requires a new run directory. Operational settings such as concurrency,
timeout, retry budgets and server readiness/shutdown deadlines can change and
are recorded per session.
`--force` reruns the current methodology in a new checkpoint generation,
preserving prior attempts and invalidating dependent adjudications.

## Scientific defaults and ablations

| Setting | Default |
|---|---|
| Model / endpoint | Qwen/Qwen3.5-122B-A10B-FP8 / http://localhost:8000/v1 |
| Independent passes | 2 |
| Adjudication | Disagreement or either pass UNCERTAIN |
| Thinking / exam edition | Enabled / included |
| Concurrency / timeout | 50 / 600 seconds |
| Server lifecycle | Managed by Python; startup 1,800 seconds / shutdown 30 seconds |
| Server context / GPU memory / active sequences | 16,384 / 0.90 / 50 |
| Seed | 42 |
| Temperature / top-p / top-k | 1.0 / 0.95 / 20 |
| Min-p / presence penalty / repetition penalty | 0.0 / 1.5 / 1.0 |
| Generation budget | 8,192 tokens including thinking |
| Structured output | JSON Schema required |
| Attempts / formatting retries | At most 5 / at most 2 |
| Exponential backoff | 1 second initially, capped at 30 seconds |
| Automatic confidence cutoff | None |

Sampling defaults follow the [model's general thinking guidance](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8).
These are methodological choices, not accuracy guarantees. Separate requests
with distinct pass seeds do not make one model statistically independent of itself.
Seeds are derived from the base seed, annotation ID, stage, pass and local attempt.
On a resumed retry session the local attempt sequence restarts; all actual seeds
are recorded. vLLM scheduling/hardware can affect bitwise reproducibility.

Pass 2 never receives pass 1 results. Adjudication receives only each prior
label and concise validated justification. It may choose another allowed label.
For an agreed subject, final confidence is the minimum pass confidence and the
first pass provides the short justification/alternative. Each complete parsed
pass result remains in SQLite for analysis.

Examples (use a NEW run directory for every methodological ablation):

```bash
uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --num-independent-passes 1 --no-thinking --temperature 0.7 --top-p 0.8 \
  --run-dir output/knowledge-annotation-work/one-pass-no-thinking

uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --no-adjudicate-disagreements --no-include-exam-edition \
  --run-dir output/knowledge-annotation-work/no-adjudication-no-edition
```

Every `RunConfig` option has a CLI flag and a `MMLU_ANNOTATION_<UPPERCASE_NAME>`
environment variable. Precedence is CLI, environment, default. Boolean environment
values are `true`, `false`, `1`, `0`. API authentication uses
`MMLU_ANNOTATION_API_KEY`, default `EMPTY`, never recorded. Taxonomy/alias overrides
use `--taxonomy`, `--exam-aliases`. Use `annotate --help` for all parameters.
Non-thinking does not silently change sampling values. `--structured-output none`
is an explicit compatibility ablation; strict local validation remains mandatory.

Transient HTTP/connection failures use bounded backoff. Invalid final schemas
get bounded independent reformulations using the same answer-free prompt and a
new attempt seed, without replaying raw output. Authentication, missing model,
model-name mismatch and server configuration errors stop the run. Context
overflow is a terminal row error; inputs are never silently truncated.

## Outputs and statuses

```text
output/knowledge-annotation-work/full/
├── manifest.json
├── server-session.json          # last managed server lifecycle
├── vllm.log                     # appended server diagnostics; response logging disabled
├── taxonomy.json
├── exam_aliases.json
├── validation.json
├── source/                      # immutable selected original Dataset
├── checkpoint.sqlite3           # tasks and audit attempts, all generations
├── exports/
│   ├── latest.json              # points to last complete validated export
│   └── generation-0001-<id>/
│       ├── dataset/             # load_from_disk
│       ├── data.parquet
│       └── data.jsonl
├── reports/
│   ├── report.json
│   ├── report.md
│   ├── by_exam.csv
│   ├── subjects_by_exam.csv
│   └── disagreement_pairs.csv
└── review/                      # created by export-review
    ├── manual_review.csv
    └── manual_review.jsonl
```

Exports preserve every original field. Added columns:

| Column | Meaning |
|---|---|
| annotation_id | SHA-256 of canonical exam, edition, number, question and ordered choices |
| annotation_row_index | Zero-based position before sampling |
| subject / macro_area | Allowed global subject / deterministic area, or null area |
| alternative_subject | Distinct allowed alternative or null |
| subject_confidence | Model-reported, uncalibrated confidence; null for operational failure/unresolved disagreement without adjudication |
| subject_justification | Concise final explanation, at most 400 characters |
| annotation_status | accepted, adjudicated, uncertain, error |
| annotation_agreement | agreement, disagreement, uncertain, single_pass, incomplete |
| annotation_pass_1_subject / annotation_pass_2_subject | Individual pass labels; second null for one-pass runs |
| annotation_model / annotation_prompt_version / taxonomy_version | Provenance |
| annotation_run_id | Manifest identity |

`accepted`: a valid single-pass subject or agreed subject. `adjudicated`: a valid
adjudicator subject. `uncertain`: abstention, including unresolved disagreement
when adjudication is disabled. `error`: a required request failed or remains
incomplete. Failed rows have null subject/area. UNCERTAIN has null area.
Two UNCERTAIN results are never accepted as agreement. A failed independent pass
is not bypassed by accepting the other pass.

Stable identities exclude both answer and row position. Duplicate occurrences
share annotation work while retaining separate source rows. Source order and
values are checked after reloading the staged Hugging Face export; Parquet count
is also checked before the latest-export pointer changes. Prior exports are retained.

The manifest captures dataset revision/fingerprint/schema, model endpoint and
immutable revision for managed jobs (when supplied for external servers), prompt/version, taxonomy snapshots/checksums,
generation/strategy/thinking settings, actual seeds in SQLite, dependencies,
timestamps, commit/dirty state and annotation-code hashes. All successful parsed
results, attempts and token usage are audit-ready. Raw model output and reasoning
fields are discarded, including on failed parses.

## QC report and human review

```bash
uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli report \
  --run-dir output/knowledge-annotation-work/full

uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli export-review \
  --run-dir output/knowledge-annotation-work/full \
  --confidence-below 0.7 --include-disagreements
```

Reports/review are rebuilt offline from the source snapshot and SQLite, including
interrupted runs. They do not call the model. Review includes uncertain/error rows
and optional low-confidence/disagreement rows. The original answer is excluded;
`--include-answer` explicitly opts in. CSV alternatives are JSON arrays.

QC includes totals, accepted subjects, errors, uncertainty, independent agreement,
adjudication requests, subject/macro counts, subject-by-exam distributions,
uncertainty/disagreement rates per exam, confidence histogram and disagreement pairs.
Agreement and disagreement rates use rows with two successful structured results;
agreement additionally excludes UNCERTAIN. Adjudication counts include attempted,
failed and successful adjudications. All-row rates use selected rows; null rates
have no eligible denominator. Duplicate occurrences count as source rows.

To push explicitly to a new repository after annotation:

```bash
uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --run-dir output/knowledge-annotation-work/full \
  --push-to-hub YOUR_NAMESPACE/mmlu-pt-subject-annotated
```

The source repository is rejected as a destination. Dry runs cannot publish.

## Future edition/course restrictions

Candidate resolution supports explicit `context_rules` inside an exam entry:

```json
{
  "allowed_subjects": ["Mathematics", "Logic"],
  "context_rules": [
    {
      "match": {"exam_edition": "EXACT EDITION VALUE"},
      "allowed_subjects": ["Mathematics"]
    }
  ]
}
```

This is an interface example, not a change to the authoritative taxonomy.
Selectors can use `exam_edition`, `course`, `specialization`, `block`; never answer.
All selector fields must match exactly; no match uses the exam list, multiple
matches fail preflight. Lists must be nonempty subsets of the exam list.
ENADE has 471 course-bearing editions, but taxonomy 1.1 does not yet define
course lists. The pipeline therefore uses its authoritative 70 candidates, and
does not infer narrower lists from edition names. BNDES/BACEN/CNU work the same way.
