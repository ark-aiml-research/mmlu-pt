# Correção dos datasets finais para a base deduplicada

Em 10 de outubro de 2026, os datasets finais foram corrigidos usando a seleção
de `bench-temp-2/mmlu-pt-deduplicated`, revisão
`54ba11ff039916f347b3090ebedc6cef6272bcea`, com as anotações de
`bench-temp-2/mmlu-pt-knowledge-annotated`, revisão
`fb9d509a62c0e05e3fecd37e7626563294a44229`.

As duas fontes foram baixadas do Hub e comparadas integralmente com os snapshots
locais antes da publicação. O cruzamento por `annotation_id` confirmou a presença
das 37.873 questões deduplicadas na fonte anotada, sem diferenças nos campos
originais. O teste final exclui os IDs usados em dev e conserva as anotações.

| Dataset | Teste anterior | Teste corrigido | Dev |
| --- | ---: | ---: | ---: |
| Ensino médio | 9.759 | 8.705 | 30 |
| Graduação | 31.800 | 29.095 | 45 |
| Total | 41.559 | 37.800 | 75 |

## Publicação e validação

- Ensino médio: `bench-temp-2/mmlu-pt-high-school`, revisão `10b711e8692a7f29084cca15f1923d97b3f9062b`.
- Graduação: `bench-temp-2/mmlu-pt-undergraduate`, revisão `4e7123a1918f574e36bea38f2a9bcc4728100bd9`.
- Snapshot local: `output/fewshot-work/main/datasets`.
- Manifesto e auditoria: `output/fewshot-work/main/{manifest.json,audit.json,audit.md}`.

Todos os splits publicados foram relidos e comparados com o snapshot local:
mesmos schemas, IDs, ordem e conteúdo. As tarefas no light-benchmark fixam essas
revisões e usam versão 1.

Para reconstruir usando as demonstrações atuais e um novo diretório de saída:

```bash
python -m mmlu_pt.fewshot.cli run \
  --source-revision fb9d509a62c0e05e3fecd37e7626563294a44229 \
  --deduplicated-revision 54ba11ff039916f347b3090ebedc6cef6272bcea \
  --dev-from output/fewshot-work/main/datasets \
  --run-dir output/fewshot-work/reproduction --no-push
python -m mmlu_pt.fewshot.cli verify --run-dir output/fewshot-work/main
```

`--source-path` e `--deduplicated-path` permitem usar snapshots locais. A construção
confere cobertura, unicidade e igualdade dos campos originais; reutiliza dev e
mantém sua ordem e justificativas. Não são necessárias novas chamadas de geração.

## Avaliações e derivados

Os experimentos Direct (21 modelos), CoT (21), Direct Base (4) e Reasoning On (3)
foram filtrados por `doc.specific.id`. Cada execução mais recente tem cobertura
completa em 0-shot e 5-shot. Os registros retidos preservam prompts, respostas,
scores e identificadores. As métricas foram recalculadas dos scores individuais.

O Hard v2 conserva a regra de remoção de questões acertadas pelos três membros do
painel em ambos os protocolos. Sua contagem passa de 17.016 para 15.269 questões:
4.584 de ensino médio e 10.685 de graduação. Hard Direct e Hard CoT, estatísticas,
sensibilidade e tabelas de ablação foram reconstruídos sobre essa seleção.

O [relatório de impacto](../output/dataset-correction/report.md) apresenta as
contagens antes/depois e os efeitos nas métricas. Suas tabelas incluem nível,
configuração, exame, disciplina e rankings por modelo/protocolo. O gerador está
em `ablations/revision/dataset_correction.py`.

## Backups

Antes de qualquer alteração, os arquivos dos dois projetos foram copiados para
`backups/dataset-correction-20261010-170232/` em cada repositório. As cópias são
independentes, sem hard links; incluem os artefatos completos e os arquivos de
trabalho, além do commit, estado do Git e diferenças staged/unstaged.

Os manifestos registram caminhos, tamanhos e SHA-256. Foram conferidos 1.418
arquivos no mmlu-pt (4.412.282.972 bytes) e 4.212 no light-benchmark
(15.550.754.913 bytes). Os diretórios de backup, metadados internos do Git e
ambientes de dependências foram excluídos da cópia.

As saídas corrigidas ocupam os caminhos atuais. Os agregados e arquivos de
execução anteriores permanecem nos backups. Os artefatos descontinuados da
Hard v1 também permanecem apenas ali.

## Publicação no Hugging Face

Os snapshots publicados de [ensino médio](https://huggingface.co/datasets/bench-temp-2/mmlu-pt-high-school-hard) e [graduação](https://huggingface.co/datasets/bench-temp-2/mmlu-pt-undergraduate-hard) preservam schemas, IDs,
ordem e conteúdo dos splits locais, conferidos após a publicação.

| Nível | Teste | Dev | Revisão |
| --- | ---: | ---: | --- |
| Ensino médio | 4.584 | 30 | `bb9370609c80641dfce3019040216b5698927b57` |
| Graduação | 10.685 | 45 | `4990ae3f4fa96b9a8799770e48b3e0cfc40a483d` |
| Total | 15.269 | 75 | — |

`output/hard-direct-v2/publication.json` registra as revisões anteriores e finais,
os commits por configuração, hashes dos snapshots e as verificações do Hub.
O manifesto de construção permanece em `output/hard-direct-v2/manifest.json`.
A publicação reutiliza as anotações e predições registradas.

## Segunda correção: cópias de demonstrações dev no teste

Em 10 de outubro de 2026, duas questões de teste foram excluídas por serem cópias
quase idênticas (mesmas alternativas e gabarito) de demonstrações dev da mesma
configuração: no 5-shot, a resposta delas aparecia no prompt.

| Teste excluído | Configuração | Demonstração dev | Cosseno |
| --- | --- | --- | ---: |
| OBI 2021 Fase 1 nível 1, item 9 | `high_school/computing_engineering_architecture_and_design` | OBI 2021 Fase 1 nível 2, item 3 | 0,9971 |
| ENADE 2017 ciências biológicas (licenciatura), item 3 | `undergraduate/mathematics_statistics_and_logic` | ENADE 2017 engenharia civil, item 3 | 0,9931 |

As demonstrações foram sorteadas da base anotada antes da deduplicação, e a checagem
de texto compartilhado só compara questões do mesmo caderno. A deduplicação semântica
removeu as duas cópias usadas em dev e manteve as do teste. As identidades e os motivos
estão em [config/excluded_test_questions.json](../config/excluded_test_questions.json),
aplicado por `mmlu_pt.fewshot` (`--exclude-test`, padrão). O dev não muda.

| Dataset | Teste anterior | Teste corrigido | Dev |
| --- | ---: | ---: | ---: |
| Ensino médio | 8.705 | 8.704 | 30 |
| Graduação | 29.095 | 29.094 | 45 |
| Total | 37.800 | 37.798 | 75 |
| Hard v2 (ensino médio / graduação) | 4.584 / 10.685 | 4.584 / 10.684 | 30 / 45 |

A questão da OBI já não estava no Hard v2 (seis acertos do painel); a do ENADE estava.
Nenhuma outra questão muda de status na seleção do Hard.

### Reconstrução e verificação

O snapshot `output/fewshot-work/main` foi reconstruído sem chamadas de API
(run `9033ec32c267`), reutilizando o dev anterior; todas as demais linhas de teste e
dev são idênticas em conteúdo e ordem. As predições Direct, CoT, Direct Base e
Reasoning On foram refiltradas com `light-benchmark/src/report/filter_hard.py`
(37.798 questões por modelo e protocolo), o Hard v2 foi reconstruído com
`python -m mmlu_pt.hard`, Hard Direct e Hard CoT foram refiltrados contra ele e as
tabelas de `ablations/hard` foram regeneradas. Não houve nova inferência.

O [relatório de impacto](../output/dataset-correction-2/report.md) compara tudo com os
backups: mudanças de microacurácia de no máximo 0,009 pp, de macroacurácia de até
0,18 pp (a área de computação do ensino médio tem 80 questões) e nenhuma mudança de
posição nos rankings.

### Publicação no Hub

Requer um token do Hugging Face com escrita (`HF_TOKEN` ou `hf auth login`). Primeiro os
datasets principais; o comando reconstrói o mesmo snapshot (mesmo run) e publica:

```bash
.venv/bin/python -m mmlu_pt.fewshot.cli run \
  --source-revision fb9d509a62c0e05e3fecd37e7626563294a44229 \
  --source-path output/knowledge-annotation-work/full-run-taxonomy-v1.4/post-annotation/huggingface \
  --deduplicated-revision 54ba11ff039916f347b3090ebedc6cef6272bcea \
  --deduplicated-path "output/06 - revised/huggingface" \
  --dev-from backups/dataset-correction-2-20261010-192821/snapshot/output/fewshot-work/main/datasets \
  --run-dir output/fewshot-work/main
```

Depois o Hard v2, cujos cards recebem a revisão publicada acima:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from datasets import load_from_disk
from huggingface_hub import DatasetCard, HfApi

root = Path("output/hard-direct-v2")
published = json.loads(Path("output/fewshot-work/main/manifest.json").read_text())["published"]
for level, parent in (("high_school", "bench-temp-2/mmlu-pt-high-school"),
                      ("undergraduate", "bench-temp-2/mmlu-pt-undergraduate")):
    repo = parent + "-hard"
    for path in sorted((root / "datasets" / level).iterdir()):
        load_from_disk(str(path)).push_to_hub(repo, config_name=path.name,
                                              commit_message=f"{path.name}: exclude test copies of dev demonstrations")
    card = DatasetCard.load(repo)
    card.text = (root / f"README-{level}.md").read_text().replace("{PARENT_REVISION}", published[parent]["revision"])
    card.push_to_hub(repo, commit_message="dataset card: exclude test copies of dev demonstrations")
    print(repo, HfApi().dataset_info(repo).sha)
PY
```

### Revisões publicadas

Estas revisões substituem as das seções anteriores. Os quatro datasets foram relidos do
Hub e comparados com os snapshots locais (schemas, IDs, ordem e conteúdo); os cards do
Hard apontam para as novas revisões principais. As tarefas do light-benchmark fixam as
revisões principais em `DATASET_REVISIONS`.

| Dataset | Teste | Dev | Revisão |
| --- | ---: | ---: | --- |
| `bench-temp-2/mmlu-pt-high-school` | 8.704 | 30 | `3938710959f204ae54ed257fcfe4ffed7c459046` |
| `bench-temp-2/mmlu-pt-undergraduate` | 29.094 | 45 | `026191bef1e9690c0dc118c81ef6ab427d285bf6` |
| `bench-temp-2/mmlu-pt-high-school-hard` | 4.584 | 30 | `510cba3b920b272b623199051718ebfda3fc9b84` |
| `bench-temp-2/mmlu-pt-undergraduate-hard` | 10.684 | 45 | `7572aceb8a6240a14d6d7d1d99b223304225beba` |

O registro da publicação do Hard está em `output/hard-direct-v2/publication.json`.

### Backups

Os arquivos alterados foram preservados em `backups/dataset-correction-2-20261010-192821/`
nos dois repositórios, com manifestos de caminhos, tamanhos e SHA-256 verificados. Os
experimentos do light-benchmark foram movidos, não copiados, para o snapshot do backup.
