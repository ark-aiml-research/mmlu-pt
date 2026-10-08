# Revisão do dataset: remoção de questões com defeito irrecuperável

Data: 7 de outubro de 2026. Lista versionada: [config/removed_questions.json](../config/removed_questions.json).
Ferramenta: `python -m mmlu_pt.revision` ([src/mmlu_pt/revision.py](../src/mmlu_pt/revision.py)).

## Motivação

A segunda full run da anotação de áreas de conhecimento (taxonomia 1.2, `4e053d1ce260f6ae`)
terminou com 23 abstenções (`UNCERTAIN`) em 41.653 questões. O experimento com a taxonomia 1.3
sobre esses 23 casos resolveu 14 por ajuste de definições e candidatos, mas 8 continuaram
indecidíveis porque o enunciado original não existe no dataset. Esses 8 itens não são um problema de
taxonomia: são defeitos de fonte introduzidos na extração ou no pré-processamento e, como itens de
avaliação, só poderiam ser acertados ao acaso.

Uma varredura por padrão de defeito nos estágios `04 - filtered-huggingface` e
`06 - semantic-deduplicated` encontrou mais 7 linhas com os mesmos defeitos que haviam recebido um
rótulo de disciplina na full run. Elas foram incluídas na mesma revisão.

## Itens removidos

A identidade de cada item é o SHA-256 canônico de `exam`, `exam_edition`, `num`, `question` e
`choices`, o mesmo `annotation_id` usado pela anotação. A remoção é aplicada por identidade, nunca
por índice, então a mesma lista vale para qualquer estágio ou subset.

| Categoria | Exame / edição / item | Linhas | Motivo |
|---|---|---|---|
| `self_referential_block` | OBI 2012 Fase 2, nível 1 e nível 2, itens 1, 2 e 3 | 6 | O enunciado é apenas "A resposta da questão N é:" e as alternativas são letras. As três questões formam um bloco de lógica que só se resolve em conjunto; isoladas em linhas independentes, cada uma é indecidível. O dataset é de itens independentes, então não há reparo sem mudar o formato. |
| `stem_replaced_by_latex_instruction` | BLUEX: UNICAMP 2018 item 2; UNICAMP 2020 item 68; UNICAMP 2023 item 9; UNICAMP 2024 itens 7 e 12; USP 2020 item 37; USP 2022 item 65 | 7 | O enunciado foi substituído, no pré-processamento, pela instrução "Seu único objetivo é detectar as expressões matemáticas, convertê-las para o padrão LaTeX…" seguida de um exemplo de função. As alternativas pertencem a outra questão (literatura, história ou gramática) e ficaram órfãs. O enunciado original não é recuperável a partir do dataset. |
| `truncated_stem` | POSCOMP 2024 itens 19 e 20 | 2 | A extração manteve só o cabeçalho "INSTRUÇÕES" e a pergunta ("A percentagem de tempo…", "O tempo médio aproximado…"), sem os dados ou o algoritmo que definem o tempo de cálculo. Sem esse contexto as questões não são respondíveis. |
| `missing_supporting_data` | COMVEST 2017 item 63; ENEM 2018 item 136 | 2 | O enunciado cita um gráfico ou tabela ("O gráfico abaixo", "Com base nessas informações") que não existe no dataset, e o texto não contém os valores necessários. Identificadas como UNCERTAIN na full run 1.4 e confirmadas por inspeção manual em 8 de outubro de 2026; ver [post_annotation.md](post_annotation.md). |
| `missing_supporting_text` | IME 2010 e 2011, item 36 | 2 | Enunciado fragmentário ("What task below could Lammert B. Otten be legally in charge of?") que depende de um texto de apoio ausente; sem ele a questão não é respondível. As duas cópias são idênticas, e nos experimentos 1.3 e 1.4 a anotação oscilou entre English Language e UNCERTAIN para o mesmo conteúdo. Incluído em 8 de outubro de 2026, após o experimento 1.4. |

Total: 19 identidades. Os IDs completos, categorias, motivos e a origem da evidência (abstenção na
full run ou varredura por padrão) estão no arquivo de configuração.

## Validação manual de casos de borda

A remoção passou por validação manual dos casos de borda antes de ser aplicada. Foram inspecionados
individualmente, e **mantidos**, os seguintes grupos, por não apresentarem o mesmo defeito:

- **80 linhas com prefixo "INSTRUÇÕES" ou "Instruções:"** (POSCOMP 2024, BACEN 2005/2006, ENADE 2006/2007).
  Em todas, exceto os dois itens POSCOMP removidos, o enunciado completo segue o prefixo, incluindo
  o texto de apoio compartilhado nas questões em bloco do BACEN e do ENADE. POSCOMP 2024 item 66
  ("Assinale a alternativa correta.") foi mantido porque as alternativas são afirmações
  autocontidas sobre o protocolo IP.
- **19 linhas de COMVEST 2011–2015 e BLUEX USP 2020/2024** cuja última alternativa carrega, após o
  texto real, o início da passagem da questão seguinte ("Texto para as questões…"). A alternativa
  correta e as demais estão íntegras antes do ruído; o item continua respondível e identificável.
  Fica registrado como limpeza futura de formatação, não como remoção.
- **Itens com texto de apoio ausente, mas tarefa identificável**: AFA 2019 item 45 (fragmento em
  inglês com alternativas em inglês) e COMVEST 2015 item 38 (técnica usada por Mendel, infográfico
  ausente). Nesses casos a competência avaliada é reconhecível a partir do enunciado e das
  alternativas; o tratamento é de taxonomia e anotação (ver versão 1.4 no
  [histórico da taxonomia](../src/mmlu_pt/annotation/knowledge_area/TAXONOMY_CHANGELOG.md)),
  não de remoção. Os dois IME item 36 foram inicialmente mantidos nesse grupo e removidos depois do
  experimento 1.4, quando cópias idênticas receberam resultados diferentes: sem o texto de apoio a
  questão não é respondível, e a instabilidade da anotação confirmou a dependência do texto.

Nenhuma questão foi removida por dificuldade, por desacordo entre passagens da anotação ou por
rótulo inesperado. Os critérios foram exclusivamente ausência de enunciado recuperável ou
dependência de um bloco que o formato do dataset não preserva.

## Datasets revisados

| Entrada | Saída | Linhas | Removidas | IDs da lista ausentes |
|---|---|---|---|---|
| `output/04 - filtered-huggingface` (idêntico a `bench-temp/mmlu-pt-filtered`, base da full run) | `output/04 - revised` | 41.653 → 41.634 | 19 | 0 |
| `output/06 - semantic-deduplicated/huggingface` | `output/06 - revised` | 37.887 → 37.873 | 14 | 5 (já eliminados pela deduplicação) |
| `output/knowledge-annotation-work/subsets/full-run-2-uncertain/dataset` | `…/subsets/full-run-2-uncertain-revised` | 23 → 15 | 8 | 7 (não eram UNCERTAIN) |

O subset revisado foi gerado com a lista de 15 itens e serviu de entrada ao experimento 1.4; os
dois IME ainda constam nele. As pastas `04 - revised` e `06 - revised` foram regeneradas a partir
dos estágios originais com a lista de 19 itens. A full run 1.4 anotou a revisão do Hub com
41.636 linhas (17 remoções); as duas remoções seguintes foram aplicadas ao seu export na
pós-anotação ([post_annotation.md](post_annotation.md)), e o Hub precisa ser republicado para
refletir a lista completa.

Cada pasta de saída contém `huggingface/` (dataset local, `load_from_disk`), `data.jsonl` (mesmos
registros e ordem, todas as colunas), `removals.jsonl` (índice de entrada, ID, exame, edição,
item, categoria e motivo de cada linha removida) e `manifest.json` (hashes da entrada, da saída e da
lista; IDs encontrados e ausentes; repositório e revisão do Hub após a publicação). A ordem e o
conteúdo das linhas mantidas são idênticos à entrada; a ferramenta recarrega a saída e verifica isso
antes de publicar a pasta. Os estágios originais não são modificados.

Comandos usados:

```bash
.venv/bin/python -m mmlu_pt.revision revise --input "output/04 - filtered-huggingface" --output "output/04 - revised"
.venv/bin/python -m mmlu_pt.revision revise --input "output/06 - semantic-deduplicated/huggingface" --output "output/06 - revised"
.venv/bin/python -m mmlu_pt.revision revise \
  --input output/knowledge-annotation-work/subsets/full-run-2-uncertain/dataset \
  --output output/knowledge-annotation-work/subsets/full-run-2-uncertain-revised
```

## Publicação e uso

O `04 - revised` é a nova base da anotação de áreas de conhecimento, publicada como
`bench-temp-2/mmlu-pt-revised` (split `train`, mesmas oito colunas públicas):

```bash
.venv/bin/python -m mmlu_pt.revision publish --output "output/04 - revised" --repo bench-temp-2/mmlu-pt-revised
```

O comando grava o commit e a revisão resultante em `output/04 - revised/manifest.json`. A full run
com a taxonomia 1.4 deve fixar essa revisão com `--dataset-revision`. O `06 - revised` substitui o
`06 - semantic-deduplicated` como dataset final deduplicado; o subset revisado é a entrada do
experimento 1.4 com os 15 casos restantes.
