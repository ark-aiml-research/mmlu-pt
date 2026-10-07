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

## Version 1.3 — experimento dirigido após full-run-2

Experimental: 78 disciplinas e nove macroáreas, sem renomear ou reagrupar. A versão padrão permanece 1.2. Nenhuma inferência 1.3 foi executada no host de desenvolvimento.

As adições de candidatos são English Language no CNU; Professional Ethics no REVALIDA; Professional Ethics e Social Security Law na residência USP/UNICAMP. Todos os candidatos anteriores permanecem.

O teste preparado contém apenas os 23 UNCERTAIN finais. Alterações motivadas por outros casos, como inglês do CNU, só serão avaliadas em uma execução que os inclua. As evidências e pendências completas estão em [mmlu_pt_taxonomy_v1_3_evidence.json](mmlu_pt_taxonomy_v1_3_evidence.json); não são gabaritos humanos.

| Tipo / alvo | Motivação e impacto esperado | Evidências (índice original base zero; ID) |
|---|---|---|
| candidates: CNU: English Language | Allow English tasks instead of forcing Portuguese. Not exercised by the 23-row subset. | 4392: `13c6b98a7c693d65643696ebdc79e908708de5be5660b6a9e31dfdf259085b9b` (CNU, CNU 2024 - bloco_6 - tarde, item 41); 4393: `892b2abd2b0d9308db3e19f3180cde574fb3142fafa7ded4d36f2895ac50e305` (CNU, CNU 2024 - bloco_6 - tarde, item 42); 4394: `17ea0cbc8520cd940c94c4203b6ec1a28541d6f71788f7099c995664a2ee324d` (CNU, CNU 2024 - bloco_6 - tarde, item 43); 4395: `d352e9f80fc6c9a79db184180b2effd7b1be237eda584ad4e6d32539ae558a72` (CNU, CNU 2024 - bloco_6 - tarde, item 44); 4396: `fdf29b6d4346d4f8e156439cc2798563b088f4d8bf7a81816324a71974e4df88` (CNU, CNU 2024 - bloco_6 - tarde, item 45); 4397: `779f986adf8679cf8bb032e7a5c6c62fcd43511ce5a321c9fa38436e7b30488c` (CNU, CNU 2024 - bloco_6 - tarde, item 46); 4398: `069cae7f25b33055c3d9fc579b26fded67a6189148b99c70266555d683b5aa16` (CNU, CNU 2024 - bloco_6 - tarde, item 47); 4399: `1dd7811fc56b4ba218bb35adf71a8c85a90d2dc7e23484674c7854fdeeb1a6f5` (CNU, CNU 2024 - bloco_6 - tarde, item 48); 4400: `d25fa830278ac946fc55abf4f6c748033aba0530cc000da1c70a1d3c0345891c` (CNU, CNU 2024 - bloco_6 - tarde, item 49); 4401: `135b3eb9346273d97b5e75a91937f9f644edb2dec08aed58e8b4c42b8d53273a` (CNU, CNU 2024 - bloco_6 - tarde, item 50) |
| candidates: REVALIDA: Professional Ethics | Allow profession-specific ethical knowledge instead of clinical proxy labels. | 40510: `97545148e8a6c16654d28759d30ad8c8d1ae2bdd4cdf308a1fd78f35628151a1` (REVALIDA, 2011, item 57); 40910: `aaada860474ea2aa5e774c083cb031e65ee1ef35ef8e1617be35f5d400281c67` (REVALIDA, 2015, item 24) |
| candidates: USP_UNICAMP_MEDICAL_RESIDENCY: Professional Ethics; Social Security Law | Cover medical codes and legal eligibility for BPC/LOAS. | 36658: `6b67ee9f2b917d4fe27b3b19a07d788cda0fdcc4f3d1d7654e8550e4e60a54d0` (RESIDENCIA_USP_UNICAMP, Residência Médica USP - Medicina Paliativa - 2024, item 10); 36985: `beb3e86bd1018fdee835cdd7f12343349ad61192efdb1cb19a8cac98a1a41429` (RESIDENCIA_USP_UNICAMP, Residência Médica USP - Psiquiatria Forense - 2024, item 28) |
| definition: Architecture and Urban Planning | Clarify CAD drawing operations without equating software use with development. | 1933: `9ecd327fd87aa949436a29544d934ea42b36878a1786da184f856041f982ceb4` (BNDES, BNDES 2008_2 - Arquiteto, item 60) |
| definition: Internal Medicine | Clarify non-surgical ophthalmology and clinical toxicology; preserve Physics/Chemistry boundaries. | 37061: `853fe1fd6a71ff0b325e1d6d4579dc4196ada2e84c84419efcc63c689bbc94eb` (RESIDENCIA_USP_UNICAMP, Residência Médica USP - Psiquiatria- 2024, item 34); 37076: `75b863c0b2a7d07ad9b03021958473733a6679007696b42a52e8cf4dcfa3fdc5` (RESIDENCIA_USP_UNICAMP, Residência Médica USP - Transplante de Córnea - Oftalmologia - 2024, item 10); 37078: `ecc247d095f0fe7f6c202c08f014a6f9cf28c643ab25a234a9beb9b6a56a0a24` (RESIDENCIA_USP_UNICAMP, Residência Médica USP - Transplante de Córnea - Oftalmologia - 2024, item 12) |
| definition: Family and Community Medicine and Public Health | Clarify health-system allocation and access, without absorbing specialist legal interpretation. | 36419: `14d3a48c22cba84effa7e2df6079d378b146db35ec8352a2ce841c47fd65a4b8` (RESIDENCIA_USP_UNICAMP, Residência Médica USP - Cirurgia Torácica - 2024, item 4); 36877: `27c52625bc36ebf4820b8a0040b7f7da63950e8b6bf29867f02e4ee8bc78141b` (RESIDENCIA_USP_UNICAMP, Residência Médica USP - Pneumologia - 2024, item 26) |
| definition: Labor Law and Labor Procedure | Separate legal occupational duties from engineering and population health. | 18217: `3b7ab188017bc3aa0d2b461cb3d8c92bd4b110071e9bb913d7fdcef14f4475a5` (ENADE, ENADE 2013 - nutricao, item 21); 16770: `8ca856789705dfaa666b8b160a3a3aec0ad5abea3a5435ee17171ee4ae88d5cb` (ENADE, ENADE 2010 - tecnologia_em_gestao_hospitalar, item 29); 17580: `f0fbb52f5b3ea33a26667622378a729e65ef57a74670ef74ee77ea08a9f60ce4` (ENADE, ENADE 2011 - tecnologia_em_saneamento_basico, item 33); 22561: `56c76cdcab0e06c0f3816f73dfbb189f7a2d752c844100c1f9293ae129f71050` (ENADE, ENADE 2019 - tecnologia_em_seguranca_do_trabalho, item 10) |
| definition: Social Security Law | Make non-contributory statutory assistance and BPC/LOAS explicit. | 36985: `beb3e86bd1018fdee835cdd7f12343349ad61192efdb1cb19a8cac98a1a41429` (RESIDENCIA_USP_UNICAMP, Residência Médica USP - Psiquiatria Forense - 2024, item 28) |
| definition: Civil Law and Civil Procedure | Make civil capacity and curatela explicit, distinct from criminal imputability. | 36992: `e4c517262c72b2da362f792d14af08d96a58038adb5203629560a0e05afec5c3` (RESIDENCIA_USP_UNICAMP, Residência Médica USP - Psiquiatria Forense - 2024, item 36) |
| definition: Professional Ethics | Explicit coverage of medical codes and professional duties. | 36658: `6b67ee9f2b917d4fe27b3b19a07d788cda0fdcc4f3d1d7654e8550e4e60a54d0` (RESIDENCIA_USP_UNICAMP, Residência Médica USP - Medicina Paliativa - 2024, item 10); 40510: `97545148e8a6c16654d28759d30ad8c8d1ae2bdd4cdf308a1fd78f35628151a1` (REVALIDA, 2011, item 57); 40910: `aaada860474ea2aa5e774c083cb031e65ee1ef35ef8e1617be35f5d400281c67` (REVALIDA, 2015, item 24) |
| definition: Administrative Law; Environmental Law | Separate sanitary administrative regulation from environmental legal obligations. | 22449: `2eefb51524d246eb3a5d827c3ec30453815700fcd27f9d992d47bc3106581eb0` (ENADE, ENADE 2019 - tecnologia_em_agronegocio, item 26) |
| policy: missing_context_rule | Identify a recognizable domain without inventing missing passages or solving the question. | 7234: `e0c57dfc854608f6eead84138fc69ff471272ddf6901fd04ba3e14e934ed4d32` (AFA, AFA_2019, item 45); 11758: `a74eac0818f2b9373ae39707e628bbdefa5e72729d56980eb39e4cd43356d6ce` (IME, IME 2010 - Portugues, item 36); 11820: `2495ad0959037a7a9d3428b8f280bcd51c0e47ab797ca32dbb964438f436b12b` (IME, IME 2011 - Portugues, item 36); 24928: `470869dad67ac45eb148c112834c489662aa3cb51c1e37f7bfac965ec0d63ca7` (COMVEST, COMVEST 2015, item 38) |
| policy: source_integrity_rule | Do not classify formatting demonstrations as the original question. | 12549: `ffa1528bf8148169364106ef0f6d0a6c5e3c77b48f14890a56e5421c62bc2b2a` (BLUEX, UNICAMP_2018, item 2); 12872: `6a8315ca1d35b655cc6897b5a3f7efcacad5a2a50bec177a8aff4a41cc5004bf` (BLUEX, UNICAMP_2024, item 12) |
| policy: label_justification_consistency_rule | Require the chosen label to match the decisive knowledge described; use UNCERTAIN instead of an unrelated proxy. | 22374: `e4cb969f883f9f33b28116a1e108749e9a38e281bebc650b311c2a141ec15656` (ENADE, ENADE 2019 - medicina_veterinaria, item 25); 4266: `9bd2c23e2d46606fa3d3e290dc04eb2996351c7656ddf4639fe87409d6bbcbef` (CNU, CNU 2024 - bloco_1 - tarde, item 43); 40707: `a9ec533b647b68b83922c151dd9e684cf4f48c36cd03f2d5a7f05446d802f32e` (REVALIDA, 2013, item 35); 36992: `e4c517262c72b2da362f792d14af08d96a58038adb5203629560a0e05afec5c3` (RESIDENCIA_USP_UNICAMP, Residência Médica USP - Psiquiatria Forense - 2024, item 36) |

As oito fontes insuficientes/contaminadas continuam pendentes de recuperação. Não reconstruir textos nem imagens neste experimento. AutoCAD, priorização de transplantes, inalantes e lentes permanecem casos de fronteira a avaliar: a definição esclarecida não obriga um rótulo nem garante acerto.

O mínimo de 2.000 registros por macroárea não é aplicável ao subset. O mapa de macroáreas não mudou, portanto a projeção da full-run-2 permanece acima do mínimo; conferir novamente após a próxima full run.
