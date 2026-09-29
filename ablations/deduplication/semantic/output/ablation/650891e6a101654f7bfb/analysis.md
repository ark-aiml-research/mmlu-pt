# Semantic deduplication results analysis

Run: 2026-09-23T14:27:53.138399+00:00. Annotation SHA-256: `a51cbf1099b969b0573abd104184bf91874df64f1dc09bbb5a418ac2110290a6`.
Panel: 466 pairs, 837 items, and 373 groups.

## Main result

The recommended configuration is **`Qwen3-Embedding-8B__question_choices`**, with a final cosine threshold
of **0.980007** (`eps=0.019993`). It was selected
in **50 of 50 outer folds**.

The complete selection procedure achieved a mean outer MCC of **0.920**,
precision of **91.6%**, recall of **99.1%**,
and F1 of **95.2%**. The standard deviation of MCC across
repetitions was 0.0065; this measures sensitivity to splits,
not a confidence interval.

| configuration | mcc | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| Qwen3-Embedding-8B__question_choices | 0.920 | 91.6% | 99.1% | 95.2% |
| Qwen3-Embedding-8B__question | 0.867 | 87.5% | 97.0% | 92.0% |
| bge-m3__question_choices | 0.567 | 89.4% | 53.4% | 66.9% |
| gte-multilingual-base__question_choices | 0.536 | 83.0% | 56.1% | 67.0% |
| gte-multilingual-base__question | 0.423 | 71.4% | 53.9% | 61.4% |
| bge-m3__question | 0.416 | 76.2% | 45.5% | 56.9% |
| llama-embed-nemotron-8b__question_choices | 0.313 | 78.1% | 27.0% | 40.1% |
| llama-embed-nemotron-8b__question | -0.039 | 31.2% | 3.9% | 6.8% |

Including answer choices changed the selected model's mean outer MCC by
**+0.053**. Full-item embeddings should be the reference
representation for further work. Truncation affected **0**
item/configuration combinations, so truncation does not explain the observed differences.
Each model uses its own input protocol, including the same task instruction for Qwen and Nemotron;
this design compares complete configurations and does not isolate model capacity
from the effect of the instruction.

## Descriptive comparison with fuzzy deduplication

The final operating point, fitted on the full panel, produced 185 true positives,
16 false positives, and 0 false negatives.
The previously selected fuzzy baseline produced 128, 6,
and 57, respectively.
The semantic approach added 57 correctly identified duplicates
and 14 false positives, while correcting 4
false positives from the fuzzy baseline.
These figures describe the same data used for final fitting; use the outer metrics
above to estimate the selection procedure's performance.

## Remaining errors

| left_exam | right_exam | label | pairs |
| --- | --- | --- | --- |
| OBI | OBI | related_but_distinct | 16 |

The error examples shown in the notebook contain almost identical passages with
changes in conditions or targets: daytime/nighttime attendance, restrictions on
beds and boxes, or different requested quantities. Very high cosine similarity can
coexist with a difference that determines the answer. The recommended threshold
maximizes MCC on this panel; it was not selected to guarantee a minimum precision
for automatic removal.

There are 7 positive pairs missed in at least one outer repetition.
The recurring example “Compra(s) na Feira” has numerical choices on one side and
ingredient names on the other, despite its `duplicate` label. This warrants reviewing
the extraction/annotation without automatically changing the label. Other cases near
the threshold involve bibliographic references, formatting changes, and reordered
choices. The outer-error table distinguishes these cases from the final fit with no FN.

## Effect of NeMo clusters

| n_clusters | positive_pairs_split | items_marked_for_removal | max_id_also_removed |
| --- | --- | --- | --- |
| 1 | 0 | 232 | 47 |
| 8 | 2 | 230 | 42 |
| 32 | 1 | 231 | 47 |

`positive_pairs_split` counts annotated duplicates that are not compared because
they fall into different clusters. `items_marked_for_removal` includes relations
between items that do not necessarily form annotated pairs; it cannot establish
removal precision. `max_id_also_removed` counts references that are also marked for
removal, confirming that `max_id` is not always a retained representative.
The single-cluster experiment is a control for this small panel, not a recommended
configuration for the full corpus.

## Cost and conclusion

On the B200, inference for the selected representation took 18.3 s,
with peak PyTorch-allocated memory of 16.3 GiB.
Timing excludes downloads and model loading; memory does not represent all memory
reserved by the process. These figures are not production benchmarks.

The selected model is the best option **among the treatments evaluated on this panel**.
The next validation should review false positives and annotate new semantic candidates
outside lexical sampling before defining a removal policy.
Labels were produced by an LLM, and groups prevent item sharing but do not guarantee
that all shared passages or similar topics are separated across folds. High recall
on this sample does not establish duplicate coverage across the full corpus.
