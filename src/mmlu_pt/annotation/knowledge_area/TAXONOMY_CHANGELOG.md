# Taxonomy 1.2 — 2026-10-07

Version 1.2 preserves all 70 subject names from 1.1, adds eight subjects, and
reorganizes 11 macro areas into nine. The CLI defaults to 1.2; version 1.1 and
existing run snapshots remain unchanged. Definitions and annotation policies
are included in prompts directly; no inference backend or generation setting
was changed for this release. The prompt template version stays
`subject_annotation_v1`; taxonomy hashes identify the changed prompt content.

This is an evidence-based taxonomy revision, not a claim that new annotations
have been generated or validated. No model inference was run.

## Evidence and review limits

The baseline is full run `a777aaac987d0966`, dataset revision
`bef2dcfb3e3ca6d0add3cf374a4c0421927af680`, with 41,653 rows. The audit selects
88 final UNCERTAIN results, 2,901 results with confidence below 0.90, and
additional targeted cases or textual signals. Confidence is uncalibrated and
is only a screening criterion.

The audit contains 3,011 distinct rows. There are 140 explicit engineering
assessments, including all 88 UNCERTAIN cases; 2,871 rows remain pending semantic
review. An assessment by Codex is not an independent human gold label. Model
justifications are retained for inspection but are not accepted as ground truth.
No suggested label is applied to the source or annotated dataset.

Read the answer-free audit in
`output/knowledge-annotation-work/taxonomy-v1.2-audit/README.md`.
`cases.jsonl` and `cases.csv` preserve questions, choices, pass/adjudication
results, review status and notes. `changes.json` links each changed definition,
rule and candidate addition to evidence with annotation ID, original row index,
exam and edition. `reviewed_cases.json`, `evidence.json` and `build_audit.py`
make those decisions inspectable and the exports reproducible. These artifacts
are local outputs under the repository's existing ignored `output/` directory;
archive them with the run when sharing the research artifacts.

No final label violated the old allowed-subject lists. The audit also checks
83,306 independent task results, 3,482 adjudications and 86,788 parsed attempt
results against the old candidates. The concern is semantic: an allowed label
can be an incorrect substitute for a missing discipline. The single malformed
attempt was recovered in the baseline and contains no parsed result to validate.

## Eight new subjects

| Subject | Macro area | Evidence from the baseline |
|---|---|---|
| Occupational Therapy | Health Sciences | ENADE 2004 Occupational Therapy, Q23–25; 2007, Q15 and Q19 |
| Radiology and Medical Imaging | Health Sciences | ENADE 2010 Radiology, Q19/Q23; USP Neuroradiology 2024, Q28 |
| Aesthetics and Cosmetology | Health Sciences | ENADE aesthetic procedures, original JSONL lines 20221 and 24542 |
| Theology and Religious Studies | Humanities | ENADE Theology doctrinal/exegetical questions; BNDES 2008_2 Q49 |
| Gastronomy and Culinary Arts | Social and Applied Social Sciences | ENADE Gastronomy wine pairing, pastry and restaurant service |
| Public Safety and Physical Security | Social and Applied Social Sciences | BACEN Técnico 2006, Q7/Q8/Q28 |
| Computer Graphics | Computing, Engineering, Architecture, and Design | POSCOMP graphics pipelines, original JSONL lines 37837 and 38323 |
| Human-Computer Interaction | Computing, Engineering, Architecture, and Design | POSCOMP perception/affordance in HCI, original JSONL line 37604 |

Gastronomy is grouped with tourism and applied professional practices, not
with clinical nutrition. The descriptions explicitly separate culinary practice
from dietetics, chemistry and industrial food engineering. Physical security is
separate from cybersecurity. The new health subjects do not absorb every
question from the corresponding professional course: clinical diagnosis,
fundamental physics and professional ethics retain their own subjects.

## Candidate additions

Existing candidates are preserved. All other exam lists remain unchanged.

| Exam | Added subjects | Count before → after |
|---|---|---:|
| BACEN | Civil Law and Civil Procedure; Criminal Law and Criminal Procedure; Labor Law and Labor Procedure; International Law; Consumer Law; Social Security Law; Psychology; Public Safety and Physical Security | 18 → 26 |
| BNDES | History; Geography; Sociology; International Relations; Criminal Law and Criminal Procedure; Theology and Religious Studies | 30 → 36 |
| BLUEX | Sociology | 10 → 11 |
| POSCOMP | Information Security; Computer Graphics; Human-Computer Interaction | 9 → 12 |
| USP_UNICAMP_MEDICAL_RESIDENCY | Psychology; Physics; Civil Law and Civil Procedure; Criminal Law and Criminal Procedure; Labor Law and Labor Procedure; Radiology and Medical Imaging | 5 → 11 |
| ENADE | All eight new global subjects | 70 → 78 |

ENADE retains its all-global-subjects policy. Adding a candidate there does not
assert that it occurs in every course, nor even that this baseline contains an
ENADE item for every new subject. No course-specific lists are inferred from
edition metadata. The existing explicit alias for the residency exam is unchanged.

## Definition and policy changes

Twenty-nine existing definitions are refined; original subject names remain
stable. The main distinctions are:

- No proxy labels for missing subjects, and no claim that a subject is absent
  without reading every candidate's complete name and definition.
- Combined legal labels include their procedural components. Industrial
  property, patents and business-innovation licensing are explicit in Business Law.
- Mathematics covers calculation with fully supplied rules, including legal or
  clinical narratives; selecting or interpreting specialist rules still requires
  the corresponding domain.
- Language comprehension follows the competence assessed, not an exam edition
  title. The Spanish clarification applies the same rule as the observed English
  cases; it is not based on an alleged Spanish-specific failure.
- Adult psychiatric diagnosis/treatment, psychological theory and forensic legal
  questions are distinguished. Nursing and Internal Medicine are not proxies for
  any health profession.
- Graphics rendering, image interpretation, interaction models and hardware
  pipelines have explicit boundaries.
- Internal professional-body governance and elections are distinguished from
  public electoral law. Professional Ethics does not include all biographical
  or institutional trivia about professionals.
- Missing context is a separate source-quality issue. Assign a subject only if
  the available question and choices identify the required knowledge; never
  invent details of missing passages, figures or prior cases.
- Formatting instructions embedded in source questions are untrusted data.
  A recoverable original question may still be classified, but a formatting
  example must not replace the actual academic task.

Control cases include ENEM 2018 Q163 (fractions in a sentencing story), BACEN
Procurador 2002 Q87 (criminal knowledge forced into Business Law), BLUEX
intersectionality (Sociology absent), POSCOMP RSA (Information Security absent),
and ENADE packaging regulation whose justification falsely claimed that
Administrative Law and Consumer Law were unavailable. These cases distinguish
taxonomy gaps from model mistakes. Seven BLUEX questions containing LaTeX
instructions and the explanatory text replacing the Stella do Patrocínio item
remain source-correction cases, not taxonomy fixes.

## Macro-area migration and projected counts

Computer Science is merged with Engineering, Architecture, and Design. Natural
and Earth Sciences is merged with Agricultural and Veterinary Sciences. Social
Sciences and Public Policy is renamed to include its applied professional fields.
Other existing subject-to-area assignments are unchanged.

| Macro area in 1.2 | Baseline subjects projected into the area |
|---|---:|
| Law | 11,124 |
| Mathematics, Statistics, and Logic | 4,796 |
| Languages, Literature, Arts, and Communication | 4,264 |
| Health Sciences | 4,213 |
| Computing, Engineering, Architecture, and Design | 3,743 |
| Natural, Earth, Agricultural and Veterinary Sciences | 3,691 |
| Humanities | 3,689 |
| Economics, Business, and Accounting | 3,581 |
| Social and Applied Social Sciences | 2,464 |

All nine areas exceed 2,000 projected records. These counts total 41,565 old
subject assignments, exclude the 88 UNCERTAIN rows, and allocate no predictions
to the eight new subjects. They are not a simulated or completed 1.2 annotation.
The deterministic remapping changes the macro-area name of 9,898 baseline rows.
No subject was changed to meet a frequency threshold. Check the minimum again
after reannotation and report any violation rather than changing labels to fill
an area.

## Reannotation and context budget

`impact.jsonl`/`impact.csv` distinguish candidate changes, definition changes,
policy changes and deterministic macro-area migration for every source row.
The annotation policies are part of every prompt, so all 41,653 rows need new
inference for a uniformly annotated 1.2 collection. A selective study must
preserve its mixed provenance. Recomputing macro areas alone does not make
old subjects predictions from taxonomy 1.2.

Use a new run directory and explicit taxonomy selection. Reports and review
exports continue to read the taxonomy stored in each old run. Annotation
resume retains the original code/configuration identity requirements; changing
the default taxonomy does not bypass them.

Tokenizer-only checks used revision
`a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9`, without weights or inference, on all
41,653 classification prompts and 3,482 adjudications using the old proposals.
The longest new classification prompt has 9,449 tokens. With an 8,192-token
output budget, the old 16,384-token context would reject two classification
inputs and one baseline adjudication. Use **`--max-model-len 20480`** for the
next run with that output budget; the engine default and generation budget
remain unchanged. All checked inputs fit 20,480 tokens, but future adjudication
proposals may differ and runtime length validation remains necessary.
