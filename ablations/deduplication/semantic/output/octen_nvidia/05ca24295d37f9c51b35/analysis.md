# NVIDIA SemDeDup with Octen via vLLM

Run: 2026-09-29T19:31:44.250347+00:00  
Panel: 466 annotated pairs, 837 unique items, 373 grouped components.

## Selected configuration

- Model: `Octen/Octen-Embedding-8B` at `5adcfa292e712091dfc30f0e97f0b2282e6cc66c`
- Embeddings: `VLLMEmbeddingModelStage`, last-token pooling, normalized 4096-dimensional vectors
- `n_clusters=1`, `which_to_keep=hard`
- cosine threshold: `0.975506753`; `eps=0.024493247`

The nested grouped procedure achieved mean outer MCC 0.958 ± 0.004,
precision 96.6%, recall 98.4%, and F1 97.5%.
The final full-panel fit is descriptive: TP=183, FP=5,
TN=276, FN=2. The NVIDIA removal workflow marked
218 of 837 experimental items for removal.

## Limits

The evaluation is transductive because unsupervised embeddings and clusters use all panel texts.
The 466 pairs were enriched through lexical retrieval and labeled by an LLM; only two negatives are
labeled `distinct`. Results therefore select the best setting among 1, 8, and 32 clusters for this
panel and do not establish population-wide optimality. Unannotated removal relations are not counted
as correct or incorrect.
