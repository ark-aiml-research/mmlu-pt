# Semantic deduplication results analysis

Run: 2026-09-23T21:13:16.810922+00:00. Annotation SHA-256: `a51cbf1099b969b0573abd104184bf91874df64f1dc09bbb5a418ac2110290a6`.
Panel: 466 pairs, 837 items, and 373 groups.

## Main result

The recommended configuration is **`Octen-Embedding-8B__question_choices`**, with a final cosine threshold
of **0.973705** (`eps=0.026295`). It was selected
in **48 of 50 outer folds**.

The complete selection procedure achieved a mean outer MCC of **0.955**,
precision of **96.0%**, recall of **98.6%**,
and F1 of **97.3%**. The standard deviation of MCC across
repetitions was 0.0102; this measures sensitivity to splits,
not a confidence interval.

| configuration | mcc | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| Octen-Embedding-8B__question_choices | 0.959 | 96.5% | 98.6% | 97.5% |
| Qwen3-Embedding-8B__question_choices | 0.920 | 91.6% | 99.1% | 95.2% |
| llama-embed-nemotron-8b__question_choices | 0.906 | 91.0% | 98.1% | 94.4% |
| Octen-Embedding-4B__question_choices | 0.901 | 93.6% | 94.4% | 94.0% |
| Qwen3-Embedding-8B__question | 0.867 | 87.5% | 97.0% | 92.0% |
| Octen-Embedding-8B__question | 0.859 | 90.5% | 92.7% | 91.6% |
| Nemotron-3-Embed-8B-BF16__question_choices | 0.855 | 85.9% | 97.5% | 91.3% |
| Qwen3-Embedding-4B__question_choices | 0.812 | 88.2% | 89.2% | 88.7% |
| Qwen3-Embedding-4B__question | 0.789 | 87.2% | 87.4% | 87.2% |
| Octen-Embedding-4B__question | 0.777 | 84.3% | 89.5% | 86.8% |
| llama-embed-nemotron-8b__question | 0.725 | 76.8% | 92.3% | 83.8% |
| Octen-Embedding-0.6B__question_choices | 0.674 | 84.2% | 75.3% | 79.5% |
| Nemotron-3-Embed-1B-BF16__question_choices | 0.650 | 87.8% | 67.6% | 76.2% |
| Nemotron-3-Embed-8B-BF16__question | 0.648 | 85.5% | 69.8% | 76.8% |
| Nemotron-3-Embed-1B-BF16__question | 0.612 | 89.3% | 60.2% | 71.9% |
| bge-m3__question_choices | 0.567 | 89.4% | 53.4% | 66.9% |
| gte-multilingual-base__question_choices | 0.536 | 83.0% | 56.1% | 67.0% |
| Octen-Embedding-0.6B__question | 0.506 | 68.3% | 73.7% | 70.7% |
| gte-multilingual-base__question | 0.423 | 71.4% | 53.9% | 61.4% |
| bge-m3__question | 0.416 | 76.2% | 45.5% | 56.9% |

Including answer choices changed the selected model's mean outer MCC by
**+0.100**. Full-item embeddings should be the reference
representation for further work. Truncation affected **0**
item/configuration combinations, so truncation does not explain the observed differences.
Each model uses its own input protocol, including the same task instruction for both Qwen sizes and Llama-Embed-Nemotron,
the document prefix for Octen, and the passage prefix for both Nemotron 3 sizes;
this design compares complete configurations and does not isolate model capacity
from the effect of the instruction.

## Descriptive comparison with fuzzy deduplication

The final operating point, fitted on the full panel, produced 184 true positives,
6 false positives, and 1 false negatives.
The previously selected fuzzy baseline produced 128, 6,
and 57, respectively.
The semantic approach added 56 correctly identified duplicates
and 5 false positives, while correcting 5
false positives from the fuzzy baseline.
These figures describe the same data used for final fitting; use the outer metrics
above to estimate the selection procedure's performance.

## Llama-Embed-Nemotron-8B comparison

With question text and answer choices, Nemotron achieved mean outer MCC
**0.906**, precision **91.0%**,
recall **98.1%**, and F1 **94.4%**.
Its question-only MCC was **0.725**.
Qwen with choices achieved MCC **0.920** on the same splits;
this observed difference is not a claim of statistical significance.
Nemotron with choices was selected in **0 of
50 outer folds**. The full selection procedure's metrics above
include folds that selected any of the evaluated configurations and differ from fixed-treatment metrics.

Nemotron used its official Transformers 4.51.0 implementation with eager bidirectional
attention and float32 masked mean pooling. Its isolated runtime leaves the project
versions unchanged. Full-item inference took **32.5 s**
with peak PyTorch-allocated memory of **20.6 GiB**.
Runtime comparisons include different attention and batching implementations;
they describe this experiment rather than model-only efficiency.

## Remaining errors

| left_exam | right_exam | label | pairs |
| --- | --- | --- | --- |
| OBI | OBI | related_but_distinct | 6 |

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
choices. The outer-error table distinguishes these cases from the final fit, which has 1 false negatives.

## Effect of NeMo clusters

| n_clusters | positive_pairs_split | items_marked_for_removal | max_id_also_removed |
| --- | --- | --- | --- |
| 1 | 0 | 219 | 37 |
| 8 | 0 | 219 | 39 |
| 32 | 0 | 220 | 41 |

`positive_pairs_split` counts annotated duplicates that are not compared because
they fall into different clusters. `items_marked_for_removal` includes relations
between items that do not necessarily form annotated pairs; it cannot establish
removal precision. `max_id_also_removed` counts references that are also marked for
removal, confirming that `max_id` is not always a retained representative.
The single-cluster experiment is a control for this small panel, not a recommended
configuration for the full corpus.

## Cost and conclusion

On the B200, inference for the selected representation took 17.6 s,
with peak PyTorch-allocated memory of 16.2 GiB.
Timing excludes downloads and model loading; memory does not represent all memory
reserved by the process. These figures are not production benchmarks.

The selected model is the best option **among the treatments evaluated on this panel**.
The next validation should review false positives and annotate new semantic candidates
outside lexical sampling before defining a removal policy.
Labels were produced by an LLM, and groups prevent item sharing but do not guarantee
that all shared passages or similar topics are separated across folds. High recall
on this sample does not establish duplicate coverage across the full corpus.
