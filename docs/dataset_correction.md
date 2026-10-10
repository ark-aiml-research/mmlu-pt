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
