# Reproducible academic subject annotation

This CLI annotates `bench-temp/mmlu-pt-filtered` with offline `vllm.LLM.generate`. The source `answer` is preserved in exports but never passed to the model.
Questions and **all choices** determine the required discipline. Candidates and
their definitions come directly from the authoritative taxonomy. `macro_area`
is derived from `subject`, never generated.

## Installation: one environment, two exclusive groups

Python 3.12 and uv are required. Run commands from the repository root.
All default artifacts are written inside `output/knowledge-annotation-work/`:
inspection uses `inspection/`, and annotation uses a separate `run-offline-<hash>/`
or `dry-run-offline-<hash>/` directory. The examples below name these run directories
explicitly. Reports, review exports, checkpoints and worker logs stay within
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
| BLUEX | 663 | 10 |
| BNDES | 4,048 | 30 |
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

Dataset counts were observed on 2026-10-06; candidate counts reflect the current
taxonomy snapshot, including Literature for BLUEX and Public Policy and Public
Administration for BNDES. These are observations, not hard-coded validation assumptions. Every invocation
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
never loads an inference engine. `--dataset-path PATH` accepts a local
`save_to_disk` Dataset or DatasetDict. `--dataset`, `--dataset-config`,
`--dataset-revision` and `--split` configure Hub loading. `train` is the default
split, validated against the splits actually present. All original fields,
including additional columns in later revisions, are preserved.

## Offline inference: one Python job on the B200 host

No HTTP server, endpoint, API key or second terminal is required. Python creates
one spawned worker per DP rank; each worker loads `vllm.LLM` once and reuses it
across independent passes, formatting retries and adjudication. Imports of vLLM
and CUDA happen only when annotation has pending tasks. Inspection, reports,
review exports and completed resumes do not load the model.

Defaults are DP=1, TP=1, 128 questions per batch per rank, and at most 50 active
sequences per rank. Each question has its own prompt and `SamplingParams`;
`LLM.generate` performs batching. A round contains at most `DP × batch_size`
questions, distributed evenly in deterministic order. Only one round is in flight.
The next round starts after all ranks complete the current round. With thinking,
one long generation can prolong a batch; offline mode does not eliminate that cost.

DP follows the [vLLM 0.25.0 offline MoE example](https://github.com/vllm-project/vllm/blob/v0.25.0/examples/features/data_parallel/data_parallel_offline.py).
Ranks obtain their DP identity from environment variables. The engines allocate
TP workers from the scheduler-visible devices. `CUDA_VISIBLE_DEVICES` is preserved,
including GPU UUIDs; when absent, all CUDA-visible GPUs remain available. The job
checks for at least `DP × TP` visible GPUs. Expert parallel is disabled. For this
MoE, ranks participate in shared expert-layer collectives; an empty rank supplies
one short auxiliary request whose output is discarded and never checkpointed.

Engine configuration uses pinned model/tokenizer revisions, text-only inference,
`generation_config="vllm"`, no prefix caching, and xgrammar with the `qwen3`
reasoning parser. Grammar constraints apply to the final JSON, including in
non-thinking mode. CUDA Graphs/compilation are allowed by default
(`enforce_eager=False`); use `--enforce-eager` for an explicit ablation.

| Offline option | Default |
|---|---:|
| `--data-parallel-size` / `--tensor-parallel-size` | 1 / 1 |
| `--batch-size` | 128 per rank |
| `--max-num-seqs` | 50 per rank |
| `--max-model-len` | 16384 |
| `--max-num-batched-tokens` | 4096 |
| `--gpu-memory-utilization` | 0.90 |
| `--enforce-eager` | false |
| `--startup-timeout` / `--shutdown-timeout` | 1800 / 30 seconds |
| `--batch-timeout` | 0: disabled |

`--batch-timeout`, when positive, applies to the entire DP round. Exceeding it
stops every worker and leaves unfinished tasks resumable. Engine/CUDA errors and
worker deaths also stop the job. They do not trigger automatic full-batch retries.
No per-request HTTP timeout or exponential network backoff remains. Polling detects
unexpected worker exits even when the batch deadline is disabled. SIGINT/SIGTERM
stop worker groups and their descendants; graceful shutdown escalates if necessary.

## Dry run, full annotation and resume

Run these commands on the inference host, from the repository root:

```bash
# Stratified dry run, with thinking
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --dry-run --samples-per-exam 5 \
  --batch-size 128 --max-num-seqs 50 \
  --run-dir output/knowledge-annotation-work/dry-run-offline

# Separate no-thinking ablation; other sampling parameters remain explicit
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --dry-run --samples-per-exam 5 --no-thinking --max-tokens 512 \
  --run-dir output/knowledge-annotation-work/dry-run-offline-no-thinking

# Full annotation
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --run-dir output/knowledge-annotation-work/full-offline

# Resume: repeat the same command, without --force
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --run-dir output/knowledge-annotation-work/full-offline

# DP=2, TP=1: requires two GPUs on the same machine
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --dry-run --samples-per-exam 5 \
  --data-parallel-size 2 --tensor-parallel-size 1 \
  --batch-size 128 --max-num-seqs 50 \
  --run-dir output/knowledge-annotation-work/dry-run-offline-dp2
```

DP=2 with `max_num_seqs=50` permits up to 100 active sequences across both ranks;
DP=2/TP=2 requires four GPUs. There is no separate client concurrency setting.
GPU memory fit, multi-GPU communication and throughput require measurement on the
inference machine. Passing simulated checks is not a GPU performance guarantee.

`knowledge_annotation.sauron` runs the thinking dry-run above using the same
`.venv`. `knowledge_annotation.yaml` keeps one GPU, eight CPUs and a 24-hour limit.
For DP=2/TP=1 change `gpus: 1` to `gpus: 2` and the command's DP size to 2.
In general request `gpus = DP × TP`; the job duration must be assessed from measurements.

The whole split is validated before selecting up to five questions per exam by a
stable seeded hash. Dry-run prints at most two safe prompt/result pairs per stage
and uses the same validation, checkpoint, export and report pipeline. It cannot
publish to the Hub or share a directory with a full run.

## Durability, timing and HTTP-era migration

SQLite uses WAL, `synchronous=FULL`, an exclusive run lock and one writer: the
coordinator. Every attempt and seed is committed before dispatch. Each rank returns
its validated results when its batch completes; the coordinator saves them while
other ranks may still be running. Unreturned results can be lost on interruption
and regenerated. Successful persisted tasks are always skipped on resume.
An interrupted run reuses its source snapshot and immutable model revision.

Formatting failures alone receive up to `format_retries` independent retries,
bounded also by `max_attempts`. New seeds are used without replaying malformed
responses. Only failed rows are regenerated. Context overflow is a terminal row
error: prompts are tokenized and checked against `max_model_len` before generation;
no truncation occurs. A resumed invocation gives failed tasks a new bounded budget,
retaining the previous audit attempts. `--force` creates a new checkpoint generation.

Offline attempt `duration_seconds` is null because `LLM.generate` returns completed
batches, not individual response timings. Attempt metadata records `dp_rank`, round, a session-unique `batch_id`,
`batch_completed_at` and `batch_duration_seconds` with `duration_scope="batch_only"`.
The batch duration is shared metadata, not an individual latency or a quantity to
sum across rows. Auxiliary outputs do not contribute to annotation token counts.
Startup, GPU information, exit codes and worker lifecycle are separately recorded.

Existing HTTP runs require a **new run directory** for offline annotation. Changing
code, taxonomy, source, scientific parameters, batch size or DP/TP/engine settings
also requires a new directory. Operational startup/shutdown/batch deadlines and
retry budgets may change on resume; their values remain recorded per session.

`report` and `export-review` still understand old manifests and SQLite checkpoints;
they do not import the removed HTTP client or load CUDA. Old `--endpoint`,
`--api-key`, `--server-mode`, `--server-manifest`, `--concurrency`, `--timeout`,
backoff and server-lifecycle flags fail with migration guidance. New environment
variables use `MMLU_ANNOTATION_<FIELD>`, for example
`MMLU_ANNOTATION_DATA_PARALLEL_SIZE` and `MMLU_ANNOTATION_STARTUP_TIMEOUT`.
Old HTTP/server environment variables are no longer used.

## Scientific defaults and ablations

| Setting | Default |
|---|---|
| Model | Qwen/Qwen3.5-122B-A10B-FP8 |
| Independent passes / adjudication | 2 / disagreement or either pass UNCERTAIN |
| Thinking / exam edition | enabled / included |
| Seed | 42 |
| Temperature / top-p / top-k | 1.0 / 0.95 / 20 |
| Min-p / presence penalty / repetition penalty | 0.0 / 1.5 / 1.0 |
| Generation budget | 8192 tokens including thinking |
| Structured output | JSON Schema plus strict local validation |
| Attempts / additional formatting retries | at most 5 / at most 2 |
| Automatic confidence cutoff | none |

Sampling defaults follow the [model's thinking guidance](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8).
Changing `--thinking` does not silently change temperature or sampling settings.
No separate thinking-token budget is added by this migration. `--max-tokens`
limits the combined reasoning and final generation. Model confidence is not
calibrated, and agreement is not a substitute for human validity checks.

Pass 2 never receives pass 1 results. Adjudication receives only prior labels and
concise validated justifications and can select another allowed subject. For an
agreed subject, confidence is the minimum pass confidence and the first pass
supplies the final justification/alternative. Complete parsed pass results remain
in SQLite. Seeds derive from base seed, annotation ID, stage, pass and local attempt.
The local attempt sequence restarts on resumed retries, with all actual seeds
recorded. Batch composition, scheduling, DP/TP and hardware may affect reproducibility.

All options have CLI/environment equivalents, with precedence CLI, environment,
default. Boolean environment values are `true`, `false`, `1`, `0`. Taxonomy/alias
paths remain configurable. `--structured-output none` disables grammar enforcement
as an explicit ablation; strict final validation still applies.

## Outputs and statuses

```text
output/knowledge-annotation-work/full-offline/
├── manifest.json
├── inference-session.json       # last offline worker lifecycle and GPU information
├── vllm-rank-0000.log            # appended rank diagnostics; no completion printing
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

The manifest captures dataset revision/fingerprint/schema, offline backend,
immutable model/tokenizer revision, DP/TP and engine settings, prompt/version, taxonomy snapshots/checksums,
generation/strategy/thinking settings, actual seeds in SQLite, dependencies,
timestamps, commit/dirty state and annotation-code hashes. All successful parsed
results, attempts and token usage are audit-ready. Raw model output and reasoning
fields are discarded, including on failed parses.

## QC report and human review

```bash
uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli report \
  --run-dir output/knowledge-annotation-work/full-offline

uv run --group annotation python -m mmlu_pt.annotation.knowledge_area.cli export-review \
  --run-dir output/knowledge-annotation-work/full-offline \
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
  --run-dir output/knowledge-annotation-work/full-offline \
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
