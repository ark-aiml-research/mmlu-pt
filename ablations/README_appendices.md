# Paper appendices

The four short appendices follow the editorial base in the manuscript ZIP.
Benchmark distributions describe the 37,798 test questions (8,704 high-school
and 29,094 undergraduate); the 75 dev examples are reported separately.
Annotation-run statistics describe the corpus processed by each run. The
19-model deduplication comparison evaluates the pipeline-stage corpora before
item revision and construction of the published splits.

Run from the repository root:

```bash
.venv/bin/python ablations/knowledge_area/make_figures.py
.venv/bin/python ablations/deduplication/evaluation_summary.py
.venv/bin/python ablations/revision/make_summary.py
```

The annotation generator writes `figures/` and `aggregates/` under its directory;
the other two generators write `aggregates/`. Each command accepts
`--output-dir` for generating into a separate directory. The annotation CSVs
separate test distributions from annotation runs, label changes, abstentions
and costs. The deduplication CSVs contain model, exam, aggregation and
composition results verified against the original per-item predictions.
Manifests record validations, dataset revisions and export hashes.

The figure macros default to `ablations/<area>/figures` and can be defined before
including the appendices to point to the manuscript's figure directories.
The short annotation appendix requires graphicx, booktabs, natbib, hyperref
(or url), and xcolor. Existing labels and citation keys are preserved.

`appendix_validation.json` records the source archive, backup, file hashes and
checks performed for this version. LaTeX structure, references, citations and
figure paths were checked statically; the two test-distribution figures were
inspected visually. No LaTeX compiler was available for compiling the manuscript.
