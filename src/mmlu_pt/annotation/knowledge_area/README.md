# Reproducible academic subject annotation

This CLI annotates `bench-temp-2/mmlu-pt-revised` (stage 04 after the [dataset revision](../../../../docs/dataset_revision.md); earlier runs used `bench-temp/mmlu-pt-filtered`) with offline `vllm.LLM.generate`. The source `answer` is preserved in exports but never passed to the model.
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
| BACEN | 1,178 | 26 |
| BLUEX | 663 | 11 |
| BNDES | 4,048 | 36 |
| CFCES | 707 | 11 |
| CNU | 491 | 50 |
| COMVEST | 452 | 11 |
| ENADE | 10,785 | 78 |
| ENAM | 314 | 11 |
| ENEM | 2,688 | 15 |
| FUVEST | 1,659 | 13 |
| IME | 958 | 5 |
| OAB | 10,251 | 16 |
| OBI | 2,301 | 3 |
| POSCOMP | 1,304 | 12 |
| RESIDENCIA_USP_UNICAMP | 1,574 | 11 |
| REVALIDA | 1,195 | 5 |

Dataset counts were observed on 2026-10-06; candidate counts reflect taxonomy 1.2.
These are observations, not hard-coded validation assumptions. Every invocation
discovers the schema and validates the entire selected split before inference.

The default `mmlu_pt_taxonomy_v1_2.json` contains 78 subjects in nine macro areas.
The preserved `mmlu_pt_taxonomy_v1_1.json` contains 70 subjects in 11 macro areas.
Version 1.0 lacks ENEM and IME and therefore cannot cover the dataset. The only
alias is explicitly recorded in `exam_aliases.json`:
`RESIDENCIA_USP_UNICAMP` → `USP_UNICAMP_MEDICAL_RESIDENCY`.
Unknown exams fail preflight and are all listed; there is no fuzzy matching.

See [the 1.2 change history](TAXONOMY_CHANGELOG.md) for the added disciplines,
candidate restrictions, evidence and projected macro-area counts. The full-run
audit is in `output/knowledge-annotation-work/taxonomy-v1.2-audit/`. Its assessed
cases are engineering reviews, not human gold labels; remaining cases are
explicitly pending. No existing annotation was relabelled or regenerated.

Select an older taxonomy explicitly with
`--taxonomy src/mmlu_pt/annotation/knowledge_area/mmlu_pt_taxonomy_v1_1.json`.
Reports and review exports use the taxonomy snapshot stored in their run, not
the current CLI default. Annotation resume still requires the original run
identity, including code hashes; selecting an old taxonomy alone does not bypass
that check. Use a new run directory for 1.2.

**Context size for 1.2:** the added definitions increase prompt length. With the
existing output budget of 8,192 tokens, use `--max-model-len 20480` for this dataset.
The default engine context remains 16,384; it is insufficient for the longest
ENADE questions with taxonomy 1.2. `context_check.json` in the audit directory
records tokenizer-only checks; no model performance or semantic accuracy has
been validated for 1.2. Runtime context checks still apply to every request.

```bash
# On the inference host; new directory, same generation budget as the baseline:
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --taxonomy src/mmlu_pt/annotation/knowledge_area/mmlu_pt_taxonomy_v1_2.json \
  --dry-run --samples-per-exam 5 --no-thinking --no-enforce-eager \
  --max-model-len 20480 \
  --run-dir output/knowledge-annotation-work/dry-run-offline-taxonomy-v1.2
```

For a full run, replace `--dry-run --samples-per-exam 5` with `--no-dry-run`
and use a new directory such as `full-run-taxonomy-v1.2`. Macro-area counts in
the change history project **old subject labels**; revalidate the 2,000-row
minimum after the new annotation. Do not change subject labels merely to meet
that threshold. All exams receive changed annotation policies in their prompts,
so a uniformly annotated 1.2 dataset requires reannotation of all rows.

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
  --max-model-len 20480 \
  --batch-size 128 --max-num-seqs 50 \
  --run-dir output/knowledge-annotation-work/dry-run-offline-thinking-taxonomy-v1.2

# Separate no-thinking ablation; other sampling parameters remain explicit
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --dry-run --samples-per-exam 5 --no-thinking --max-tokens 512 \
  --run-dir output/knowledge-annotation-work/dry-run-offline-no-thinking-taxonomy-v1.2

# Full annotation
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --max-model-len 20480 \
  --run-dir output/knowledge-annotation-work/full-offline-taxonomy-v1.2

# Resume: repeat the same command, without --force
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --max-model-len 20480 \
  --run-dir output/knowledge-annotation-work/full-offline-taxonomy-v1.2

# DP=2, TP=1: requires two GPUs on the same machine
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli annotate \
  --dry-run --samples-per-exam 5 \
  --max-model-len 20480 \
  --data-parallel-size 2 --tensor-parallel-size 1 \
  --batch-size 128 --max-num-seqs 50 \
  --run-dir output/knowledge-annotation-work/dry-run-offline-dp2-taxonomy-v1.2
```

DP=2 with `max_num_seqs=50` permits up to 100 active sequences across both ranks;
DP=2/TP=2 requires four GPUs. There is no separate client concurrency setting.
GPU memory fit, multi-GPU communication and throughput require measurement on the
inference machine. Passing simulated checks is not a GPU performance guarantee.

`knowledge_annotation.sauron` is the job template using the same `.venv`.
Before submitting it for 1.2, select the new taxonomy and a new run directory,
and set `--max-model-len 20480` when retaining the 8,192-token generation budget.
`knowledge_annotation.yaml` keeps one GPU, eight CPUs and a 24-hour limit.
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
Worker failures include the rank, execution phase and stack locations in the rank
log and `inference-session.json`. Exception messages, source lines, local values
and model completions are excluded from these diagnostics. Tokenization explicitly
uses `return_dict=False` and `return_tensors=None` so Transformers 5 returns the
flat token ID list expected by vLLM, including for auxiliary requests.

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
  --max-model-len 20480 \
  --run-dir output/knowledge-annotation-work/full-offline-taxonomy-v1.2 \
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
ENADE has 471 course-bearing editions, but taxonomy 1.2 does not yet define
audited course lists. The pipeline therefore uses its authoritative 78 candidates, and
does not infer narrower lists from edition names. BNDES/BACEN/CNU work the same way.

## Taxonomia 1.3: experimento com os 23 UNCERTAIN da full-run-2

A versão 1.3 foi experimental e selecionada explicitamente por `--taxonomy`; o padrão da
CLI era a 1.2 e hoje é a 1.4. Ela mantém 78 disciplinas e nove macroáreas; as alterações,
IDs de evidência e pendências estão no [histórico](TAXONOMY_CHANGELOG.md) e no
[registro de evidências](mmlu_pt_taxonomy_v1_3_evidence.json). Esses registros não entram
nos prompts e não constituem um padrão-ouro humano.

Prepare o dataset local (sem inferência):

```bash
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli prepare-subset \
  --source-run output/knowledge-annotation-work/full-run-2 \
  --output-dir output/knowledge-annotation-work/subsets/full-run-2-uncertain
```

O comando lê a geração publicada em `exports/latest.json` e recupera os registros
no `source` salvo. Seleciona somente `subject == "UNCERTAIN"`, que corresponde a
23 registros na segunda full run. Recusa pasta existente e não modifica o SQLite.
A pasta contém `dataset/`, carregável com `datasets.load_from_disk`, e `subset.json`,
com hashes, revisão da fonte, geração de origem e mapeamento de IDs/índices.

O dataset contém os campos originais, incluindo `answer`, mas nenhuma anotação
anterior. `source_row_index` preserva o índice original, base zero, e `source_run_id`
identifica a execução de origem. A nova `annotation_row_index` será a posição local
no subset; `annotation_id` permanece igual. Apenas exame, edição, pergunta e
alternativas são enviados ao worker/modelo; o gabarito e os campos de proveniência
não participam dos prompts.

Copie a pasta inteira do subset para o mesmo caminho na máquina de inferência.
Não é necessário copiar a full run de origem para executar a anotação.

Inspecione a seleção e a taxonomia sem carregar vLLM:

```bash
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli inspect \
  --dataset-path output/knowledge-annotation-work/subsets/full-run-2-uncertain/dataset \
  --taxonomy src/mmlu_pt/annotation/knowledge_area/mmlu_pt_taxonomy_v1_3.json \
  --output-dir output/knowledge-annotation-work/inspection-uncertain-v1.3
```

Na máquina com B200, execute o job `knowledge_annotation_uncertain.sauron`, usando
`knowledge_annotation_uncertain.yaml` no scheduler, ou diretamente:

```bash
bash knowledge_annotation_uncertain.sauron
```

O job usa os parâmetros científicos da full-run-2: pesos na mesma revisão imutável,
thinking desligado, duas passagens e adjudicação, seed 42, amostragem igual,
CUDA Graphs habilitados, DP=TP=1, lote 128, 50 sequências e contexto 20.480.
**Não acrescente `--dry-run`: o dataset já contém todos os casos do experimento**;
a amostragem por exame excluiria parte das questões. O job aponta sempre para o subset e a
taxonomia mais recentes e para a pasta de saída do experimento corrente; para outro experimento
(outra taxonomia, thinking ligado), edite esses três valores no próprio arquivo e use uma pasta nova.
O experimento 1.3 foi executado com o subset original de 23 casos e a taxonomia 1.3.

Para retomar, repita exatamente o comando da execução interrompida, com os mesmos
arquivos e parâmetros. Não use `--force`: os resultados bem-sucedidos serão reutilizados.
O contexto é verificado em tempo de execução, sem truncamento silencioso.

Depois de trazer a run experimental para a máquina que possui a baseline, compare:

```bash
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli compare-subset \
  --source-run output/knowledge-annotation-work/full-run-2 \
  --run-dir output/knowledge-annotation-work/uncertain-taxonomy-v1.3 \
  --output-dir output/knowledge-annotation-work/comparison-uncertain-v1.3
```

A comparação usa IDs e valida o conteúdo e a proveniência. Gera `comparison.json`,
`comparison.md`, `cases.csv` e `cases.jsonl`, sem gabaritos, com resultados anteriores
e novos, passagens, status de adjudicação, confiança e justificativa final. IDs ausentes,
inesperados, conteúdo alterado, rótulos não permitidos e macroáreas inconsistentes
invalidam a comparação; o CLI retorna código 2. Erros finais de anotação também retornam 2.
Os comandos de preparação/comparação recusam sobrescrever uma pasta existente;
use um novo destino ao comparar uma nova geração.

`--review-notes` permite selecionar outro registro de avaliações no mesmo formato do
JSON de evidências fornecido. As notas de problemas de fonte são avaliações de engenharia,
não rótulos esperados, e não afetam a inferência. “Recebeu disciplina” não significa
“foi corrigido”: revisar adequação semântica, sobretudo onde a fonte está incompleta.
Os 23 casos não medem acurácia global nem validam as mudanças do CNU, ausente do subset.

## Taxonomia 1.4: experimento com os 15 casos restantes e full run sobre o dataset revisado

O experimento 1.3 (run `9f4f7174e7631904`) deu disciplina a 14 dos 23 casos e manteve 9 UNCERTAIN.
Oito desses nove não têm enunciado recuperável e foram removidos do dataset, junto com sete linhas
com o mesmo defeito ([docs/dataset_revision.md](../../../../docs/dataset_revision.md)). A versão 1.4
trata o caso restante (AFA 2019 item 45) e a fragilidade dos dois IME, que só receberam English
Language na adjudicação: enunciado e alternativas em inglês identificam English Language quando
permitido, mesmo sem a passagem. Candidatos e aliases são os da 1.3; mudanças e evidências estão no
[histórico](TAXONOMY_CHANGELOG.md) e em [mmlu_pt_taxonomy_v1_4_evidence.json](mmlu_pt_taxonomy_v1_4_evidence.json).

O subset revisado, com os 15 casos restantes, já está em
`output/knowledge-annotation-work/subsets/full-run-2-uncertain-revised/huggingface` (gerado por
`python -m mmlu_pt.revision revise` a partir do subset original). O job
`knowledge_annotation_uncertain.sauron` já aponta para esse subset, para a taxonomia 1.4 e para
`output/knowledge-annotation-work/uncertain-taxonomy-v1.4`. Copie a pasta do subset para a máquina
de inferência e execute:

```bash
bash knowledge_annotation_uncertain.sauron
```

Compare com a baseline usando as notas da 1.4, que já marcam os oito itens removidos:

```bash
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli compare-subset \
  --source-run output/knowledge-annotation-work/full-run-2 \
  --run-dir output/knowledge-annotation-work/uncertain-taxonomy-v1.4 \
  --output-dir output/knowledge-annotation-work/comparison-uncertain-v1.4 \
  --review-notes src/mmlu_pt/annotation/knowledge_area/mmlu_pt_taxonomy_v1_4_evidence.json
```

A comparação parte dos 23 UNCERTAIN da baseline. Os oito itens marcados como
`removed_from_dataset` nas notas contam como `removed`, não como ausentes, e não invalidam a
comparação; qualquer outro ID ausente continua sendo `missing`. Avalie os 15 casos presentes em
`cases.csv`.

Após avaliar o experimento, execute a full run sobre o dataset revisado completo
(`bench-temp-2/mmlu-pt-revised`, 41.636 linhas, publicado por `python -m mmlu_pt.revision publish`),
sem `--dataset-path`, em uma nova pasta. O job `knowledge_annotation.sauron` (com
`knowledge_annotation.yaml` no scheduler) faz isso usando os padrões da CLI: dataset
`bench-temp-2/mmlu-pt-revised` na revisão mais recente do Hub (o sha resolvido fica no manifesto),
taxonomia 1.4, duas passagens com adjudicação, seed 42 e a mesma amostragem da full-run-2;
o script fixa apenas thinking desligado, contexto 20.480, DP=TP=1, lote 128 e 50 sequências:

```bash
bash knowledge_annotation.sauron
```

Para retomar, repita o comando. Para fixar uma revisão específica do dataset, acrescente
`--dataset-revision 0e8b69a1d3ba429903b199cb48f40b813b874f43` ao script.

## Pós-anotação: atribuições manuais e remoções sobre o export da full run

Depois da full run, os `UNCERTAIN` restantes são revisados um a um. Disciplinas atribuídas
manualmente ficam em [config/manual_annotations.json](../../../../config/manual_annotations.json)
(sempre dentro dos candidatos do exame, com justificativa) e questões sem material de apoio entram em
[config/removed_questions.json](../../../../config/removed_questions.json). O comando abaixo aplica
as duas listas ao export publicado e grava `post-annotation/` na pasta da run, sem tocar no
checkpoint nem nos exports:

```bash
uv run --locked --group annotation python -m mmlu_pt.annotation.knowledge_area.cli finalize \
  --run-dir output/knowledge-annotation-work/full-run-taxonomy-v1.4
```

As linhas manuais recebem `annotation_status = "manual"` e confiança nula; as passagens do modelo
são preservadas. Cada decisão da full run 1.4 está documentada em
[docs/post_annotation.md](../../../../docs/post_annotation.md). O dataset final fica em
`post-annotation/huggingface` e é o que deve ser publicado.

O limite de 2.000 registros por macroárea deverá ser conferido nessa full run;
ele não se aplica ao subset. Nenhum resultado experimental é mesclado automaticamente
com a execução anterior. Se houver novas mudanças após o piloto, versionar a taxonomia
e usar outra pasta em vez de substituir silenciosamente a versão já executada.
