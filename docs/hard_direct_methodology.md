# MMLU-PT-Hard-Direct v2

A v2 é um subconjunto obtido pela remoção de questões consensualmente fáceis
para o painel. Remove somente questões acertadas pelos três selecionadores
em ambos os protocolos, 0-shot e 5-shot. Mantém questões intermediárias,
sem amostragem ou balanceamento. A presença de todas as áreas não é requisito.

## Entradas e regra

O painel fixo é `gemini-3.8-flash`, `google/gemma-4-31B-it` e
`Qwen/Qwen3.5-27B`, no experimento `exp-mmlu-pt-direct`.
Para cada questão e modelo, calcula-se:

```text
média = (extractive_match_0shot + extractive_match_5shot) / 2
remover = todos os três modelos têm média igual a 1
manter = existe pelo menos um erro entre os seis resultados
```

Os scores registrados são binários. A seleção lê diretamente
`metric.extractive_match`; respostas textuais e suas extrações não são
examinadas. Ambos os protocolos participam da construção.

A associação usa nível acadêmico e ID estável em `doc.specific.id`, nunca o
identificador posicional `doc.id`. Os detalhes Parquet seguem a organização
`details/<modelo>/<execução>/details_mmlu-pt-<nível>-direct:<área>|<shots>_<data>.parquet`.
Cada modelo encontrado deve ter uma predição por questão em cada protocolo.
Scores ausentes, duplicados ou não binários interrompem a execução, assim
como IDs desconhecidos ou conteúdo diferente do snapshot.

O snapshot contém DatasetDicts `<nível>/<área>` com `test` e `dev`.
Conferem-se configuração, enunciado e alternativas serializados no prompt,
rótulos das alternativas e gabarito, além de unicidade e separação global
entre IDs de dev e test. Se houver múltiplas execuções da mesma combinação,
forneça somente a execução desejada: o programa não resolve conflitos.

## Execução e exportação

Na raiz do repositório, com as dependências do ambiente:

```bash
python -m mmlu_pt.hard
```

Padrões:

- `--results-dir /raid/aluno_paulosantana/repositories/light-benchmark/output/exp-mmlu-pt-direct`
- `--source-dir output/fewshot-work/main/datasets`
- `--output-dir output/hard-direct-v2`

Para reproduzir:

```bash
python -m mmlu_pt.hard --output-dir output/hard-direct-v2-reproduction
```

A saída deve ser nova e não pode conter entradas nem estar dentro delas.
A execução prepara artefatos em um diretório temporário e só move a saída
após as verificações. Não requer novas inferências nem publica no Hub.

Mantêm-se todas as colunas, os valores e a ordem original do teste. O dev
original é copiado integralmente para configurações com test não vazio.
Configurações vazias são reportadas e não exportadas. Hugging Face e JSONL
são relidos para comparar schemas e registros.

```text
output/hard-direct-v2/
├── manifest.json
├── audit.jsonl
├── scores.csv
├── mean_scores.csv
├── sensitivity.csv
├── distributions.json
├── report.md
├── datasets/<nível>/<área>/{test,dev}/
└── jsonl/<nível>/<área>/{test,dev}.jsonl
```

O manifesto registra versão, painel, protocolos, regra, contagens, parâmetros
estatísticos e SHA-256 das entradas e dos IDs selecionados em ordem. Os
resultados JSON são apenas identificados por hash, sem copiar configurações
ou credenciais. A auditoria inclui todas as questões originais, com os seis
scores, as três médias, totais de acertos/erros e decisões de remoção/inclusão.
Esses campos permanecem separados do schema público do dataset.

## Resultados e intervalos

`scores.csv` apresenta os modelos encontrados em 0-shot e 5-shot, com micro
por nível/configuração e macro por nível. Os intervalos micro são de Wilson
95%; a macro dá peso igual às áreas presentes, com bootstrap estratificado
dentro de cada área (2.000 reamostragens, seed 42). Contagens binomiais são
equivalentes em distribuição a reamostrar registros para scores binários.
Áreas ausentes têm `n=0` e scores/intervalos indefinidos, nunca nota zero.

`mean_scores.csv` informa micro por nível em cada protocolo, sua média e IC
percentil de 95% por bootstrap de questões (2.000, seed 42). Mantém-se o par
0/5-shot de cada questão: as médias assumem 0, 0,5 ou 1 e reamostram-se suas
contagens multinomiais. Os dois protocolos não são tratados como respostas
independentes; `n` é o número de questões, não o dobro dele.

Todos os intervalos são condicionais ao subconjunto fixo e não corrigem
viés de seleção, dependência entre questões da mesma prova ou incerteza da
escolha do painel. Bootstrap pode subestimar incerteza em áreas pequenas,
especialmente quando todos os resultados são acertos ou erros.

`sensitivity.csv` retira um selecionador, remove itens acertados em ambos
os protocolos pelos dois restantes e avalia o retirado em 0-shot, 5-shot e
na média (`shots=mean`). Os ICs individuais são Wilson e o da média usa
bootstrap pareado por questão. Cada retirada define um subconjunto distinto;
a tabela não é um ranking sobre um conjunto comum e não altera a seleção.

## Referências para reprodução

Para as entradas padrão, espera-se **4.584** questões de ensino médio e
**10.684** de graduação: **15.268** de 37.798 questões originais de teste.
São valores de referência, não quotas aplicadas a outras entradas.

| Modelo | Média ensino médio | Média graduação |
| --- | ---: | ---: |
| Gemini-3.8-Flash | 82,35% | 79,64% |
| Gemma-4-31B-it | 30,22% | 45,52% |
| Qwen3.5-27B | 41,51% | 34,81% |

A construção verifica a equivalência entre médias iguais a 1 e seis
acertos, invariância à ordem dos protocolos e do painel e independência dos
scores de modelos externos ao painel. Reexecuções devem reproduzir auditoria,
scores, distribuições, relatórios e IDs; o timestamp do manifesto muda.

## Alternativas

As [tabelas versionadas de alternativas](../ablations/hard/README.md) guardam
os resultados calculados pela média 0/5-shot, com painéis, limiares e hashes.
As análises podem ser reproduzidas pelos scripts indicados nessa documentação.

[ARC-Challenge](https://arxiv.org/abs/1803.05457) fundamenta seleção por
comportamento de sistemas de referência; [MMLU-Pro](https://arxiv.org/html/2406.01574v6)
remove questões fáceis por consenso de modelos. A unanimidade de três
selecionadores em dois protocolos é a adaptação adotada aqui.

Os scores dos selecionadores são condicionados aos acertos usados na
construção. A média não elimina esse condicionamento: ambos os protocolos
foram usados na seleção. A filtragem pode alterar rankings, conforme
[Phang et al., 2022](https://aclanthology.org/2022.dadc-1.8/).

## Publicação no Hugging Face

Os snapshots publicados de [ensino médio](https://huggingface.co/datasets/bench-temp-2/mmlu-pt-high-school-hard) e [graduação](https://huggingface.co/datasets/bench-temp-2/mmlu-pt-undergraduate-hard) preservam schemas, IDs,
ordem e conteúdo dos splits locais, conferidos após a publicação.

| Nível | Teste | Dev | Revisão |
| --- | ---: | ---: | --- |
| Ensino médio | 4.584 | 30 | `510cba3b920b272b623199051718ebfda3fc9b84` |
| Graduação | 10.684 | 45 | `7572aceb8a6240a14d6d7d1d99b223304225beba` |
| Total | 15.268 | 75 | — |

`output/hard-direct-v2/publication.json` registra as revisões anteriores e finais,
os commits por configuração, hashes dos snapshots e as verificações do Hub.
O manifesto de construção permanece em `output/hard-direct-v2/manifest.json`.
A publicação reutiliza as anotações e predições registradas.
