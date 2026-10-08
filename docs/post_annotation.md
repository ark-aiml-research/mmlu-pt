# Pós-anotação da full run 1.4: atribuições manuais e remoções

Data: 8 de outubro de 2026. Run de origem: `1640098255bb0bed`
(`output/knowledge-annotation-work/full-run-taxonomy-v1.4`, taxonomia 1.4, dataset
`bench-temp-2/mmlu-pt-revised` na revisão `0e8b69a1…`). Listas versionadas:
[config/manual_annotations.json](../config/manual_annotations.json) e
[config/removed_questions.json](../config/removed_questions.json).
Ferramenta: `python -m mmlu_pt.annotation.knowledge_area.cli finalize`
([post_annotation.py](../src/mmlu_pt/annotation/knowledge_area/post_annotation.py)).

## Contexto

A full run 1.4 anotou 41.636 questões sem erros e terminou com 8 abstenções (`UNCERTAIN`),
nenhuma delas repetida da full run anterior. Cada uma foi revisada manualmente e tratada de uma
de duas formas: atribuição manual de disciplina, sempre dentro dos candidatos do exame, ou remoção
da questão por material de apoio ausente. O resultado é o dataset anotado final, sem abstenções.

A ferramenta lê o export publicado da run (`exports/latest.json`), aplica as duas listas por
`annotation_id`, valida que cada disciplina manual pertence aos candidatos do exame e deriva a
macroárea da taxonomia da run. Grava `post-annotation/` dentro da pasta da run com `huggingface/`,
`data.jsonl`, `manual_assignments.jsonl`, `removals.jsonl` e `manifest.json`. O checkpoint e os
exports da run não são modificados.

## Atribuições manuais

As seis linhas abaixo têm `annotation_status = "manual"` e `subject_confidence` nulo no dataset
final. As passagens do modelo e o estado de concordância ficam preservados nas colunas
`annotation_pass_1_subject`, `annotation_pass_2_subject` e `annotation_agreement` para auditoria.
Toda justificativa no dataset começa com "Atribuição manual".

| Exame / edição / item | Passagens do modelo | Disciplina atribuída | Motivo |
|---|---|---|---|
| BNDES 2011 Engenheiro, item 26 | Spanish Language / Portuguese Language | **Spanish Language** | Erro do adjudicador: afirmou que Spanish Language não estava na lista, mas é candidato do BNDES e foi a escolha da passagem 1. A questão pede a função da conjunção espanhola "sino" em texto em espanhol. |
| OAB-PR 2006.2, item 3 | Business Law / Civil Law and Civil Procedure | **Professional Ethics** | Erro do adjudicador: afirmou que Professional Ethics não estava na lista, mas é candidato da OAB. A questão trata de impedimentos e incompatibilidades do Estatuto da Advocacia e da OAB. |
| ENEM 2023, item 89 | History / Philosophy | **Sociology** | Lacuna de candidatos: a questão classifica condutas da Lei Maria da Penha como violência patrimonial, mas o ENEM não tem candidatos de Direito. Sociology é a disciplina que trata violência de gênero, cidadania e direitos como fenômeno social, o enquadramento da prova de Ciências Humanas. |
| COMVEST 2016, item 71 | Philosophy / History | **Sociology** (alternativa: Geography) | Lacuna de candidatos: competências exclusivas da União na organização federativa da Constituição de 1988, sem candidatos de Direito. Sociology cobre a organização política do Estado e o pacto federativo; Geography é alternativa pelo ordenamento territorial e regiões metropolitanas nas alternativas. |
| Residência USP Psiquiatria 2024, item 23 | Family and Community Medicine and Public Health / UNCERTAIN | **Family and Community Medicine and Public Health** | Lacuna de candidatos: efeito de uma resolução da ANVISA sobre a comercialização de um produto; a residência não tem Administrative Law. A definição 1.4 dessa disciplina inclui organização do sistema de saúde e regulação em saúde; foi a escolha da passagem 1. |
| Cirurgia de Cabeça e Pescoço USP 2024, item 73 | UNCERTAIN / Pediatrics | **General Surgery** (alternativa: Pediatrics) | Fronteira: embriologia da laringe e faringe como base de anatomia e malformações, em prova de especialidade cirúrgica, sem candidato de ciências básicas. General Surgery é onde a anatomia e a embriologia cirúrgicas do pescoço são conhecimento de base; Pediatrics é alternativa pelas malformações congênitas. |

Os dois erros de adjudicação contrariam a regra `label_justification_consistency_rule` da taxonomia
1.4, que pede para reconferir os candidatos antes de afirmar que uma disciplina está ausente. São
falhas pontuais do modelo, não lacunas da taxonomia, e por isso foram corrigidos manualmente em vez
de motivar uma nova versão.

## Remoções

Duas questões foram removidas do dataset por material de apoio ausente, categoria
`missing_supporting_data` em `config/removed_questions.json`:

| Exame / edição / item | Motivo |
|---|---|
| COMVEST 2017, item 63 | O enunciado cita "O gráfico abaixo" e é cortado antes de definir a relação ou os valores; sem o gráfico, a questão não é respondível. |
| ENEM 2018, item 136 | O enunciado é só "Com base nessas informações, o banco que transferiu a maior quantia via TED é o banco", sem a tabela de valores. |

A lista passou a ter 19 identidades. As remoções foram aplicadas em três lugares: no dataset final
desta pós-anotação, no `output/04 - revised` (regenerado a partir do estágio 04 original, 41.634
linhas) e no `output/06 - revised` (regenerado a partir do estágio 06, 37.873 linhas). Ver
[dataset_revision.md](dataset_revision.md).

## Dataset final

| Métrica | Valor |
|---|---|
| Linhas | 41.634 |
| UNCERTAIN | 0 |
| `accepted` / `adjudicated` / `manual` | 38.337 / 3.291 / 6 |
| Menor macroárea | Social and Applied Social Sciences, 2.472 (mínimo exigido 2.000) |

Os IDs do dataset final coincidem, na mesma ordem, com os do `output/04 - revised` regenerado.
O dataset base `bench-temp-2/mmlu-pt-revised` foi republicado com as 41.634 linhas na revisão
`e0f638e00f9bc061e4fe16ccbb4318766bf16a86`, conferida contra `output/04 - revised`; os IDs do
dataset anotado final coincidem com os dela na mesma ordem. A full run 1.4 anotou a revisão
anterior, `0e8b69a1…` (41.636 linhas), e as duas remoções foram aplicadas ao seu export.

Comandos usados:

```bash
.venv/bin/python -m mmlu_pt.annotation.knowledge_area.cli finalize \
  --run-dir output/knowledge-annotation-work/full-run-taxonomy-v1.4
.venv/bin/python -m mmlu_pt.revision publish \
  --output output/knowledge-annotation-work/full-run-taxonomy-v1.4/post-annotation \
  --repo bench-temp-2/mmlu-pt-knowledge-annotated
```

## Publicação

O dataset final está em `bench-temp-2/mmlu-pt-knowledge-annotated`, revisão
`fb9d509a62c0e05e3fecd37e7626563294a44229` (8 de outubro de 2026), conferida linha a linha contra
`post-annotation/huggingface`: mesmos IDs, ordem, disciplinas, status e gabaritos. A anotação
anterior (taxonomia 1.2, revisão `9a578d25…`) foi renomeada para `-deprecated` e depois excluída
do Hub; ela sobrevive apenas localmente, em `output/knowledge-annotation-work/full-run-2/exports`.
