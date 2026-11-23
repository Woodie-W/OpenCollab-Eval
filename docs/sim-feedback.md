# Replay SOL feedback into self-improvement memory

Run the evaluator on a trusted local SOL campaign after collecting official results.

```bash
oc-eval sim-feedback ./campaign > feedback.json
jq '.memory' feedback.json > next-memory.json
```

The campaign contains a task manifest, its hash-bound version-1 source memory,
and frozen source files under `cases/NNN/{before,after}/submission.py`.
The revision prediction in `cases/NNN/after/prediction.json` identifies both
sources and the input memory. Each arm has a verified submission receipt and
a terminal result under `deepseek-official/NNN/{before,after}/`. The optional
`deepseek-b200-snapshot.json` indexes collected terminal evidence by submission
ID. The caller is responsible for collecting authentic receipts and results.

Each completed pair adds its prediction and observed outcome to the optimizer
history, including failed revisions. If available, the source-matched A800
grade supplies local workload coverage. The selected parent is the accepted
Private B200 v1.1 source with the higher SOL score. Ties retain the original.
A pending pair retains an already accepted original. A null selection means the
pair has no confirmed usable parent yet. Paths in the selection are relative
to the campaign. This command reads local evidence and writes JSON to stdout.
