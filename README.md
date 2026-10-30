# LoreHopQA — validation experiments

Code for the experiments that validate **LoreHopQA**, a multi-hop question-answering benchmark
whose items are set in 61 invented worlds, so that the retrieved context is the only source of
the answer.

**Dataset:** [`irlab-udc/lorehopqa`](https://huggingface.co/datasets/irlab-udc/lorehopqa) on the
Hugging Face Hub (gated; access is approved by the authors).

This repository holds the evaluation code only. The generation pipeline, the world
specifications and the acceptance jurors that built the collection are not part of it.

## What the experiments measure

| Paper | Script | Question |
|---|---|---|
| E1 Closed-book answering | `scripts/experiment_1_closed_book.py` | Can a model answer with no documents at all? |
| E2 Ablated evidence | `scripts/experiment_2_ablated_evidence.py` | Is each supporting document necessary? |
| E3 Complete context | `scripts/experiment_3_full_context.py` | Are the items readable when everything is supplied? |
| E4 End-to-end retrieval | `scripts/experiment_5_bm25_retrieval.py`, `scripts/build_es_tasks.py` | Does accuracy track what a retriever returns? |
| E5 First-stage retrieval and exposure | `scripts/build_dense_tasks.py`, `scripts/embed_corpus.py`, `scripts/adapt_embedder.py`, `scripts/adaptation_*.py` | How do lexical, dense and hybrid rankings compare, and how much of a dense model's advantage is target-domain exposure? |

The internal run directories are named `e1`, `e2`, `e3` and `e5`; paper experiments 4 and 5 are
both read out of the `e5` run together with the dense, hybrid and adapted rankings.

## Layout

```
src/lorehopqa/     experiments.py   experiment driver, task building, scoring loop
                   reader.py        split loading, context manipulation, world assignment
                   eval_metrics.py  official HotpotQA metrics
                   runner.py        in-process vLLM engine, grammar-constrained decoding
                   panel.py         the reader panel and its participating/control grouping
scripts/           experiment entry points, retrieval builders, analysis, Slurm runners
configs/           reader panel configuration
results/           aggregated analysis output and the paper's LaTeX tables
```

## Requirements

Python 3.11+, `pydantic`, `orjson`, `pyyaml`. Running a reader additionally needs `vllm` and a
GPU; the analysis scripts run on CPU with the standard library alone, so aggregated results can
be re-derived on a login node.

Retrieval uses Elasticsearch 8.12 with BM25 at `k1 = 0.9`, `b = 0.4` and the standard English
analyser. Dense and hybrid rankings are vector fields over the same documents.

## Running

Each experiment builds a task list once and reuses it across readers, so the same context is
scored by every model:

```bash
python scripts/experiment_1_closed_book.py \
    --config configs/readers.yaml \
    --dist-dir /path/to/dist --split test \
    --output-dir runs/ --model qwen
```

Then aggregate and emit the tables:

```bash
python scripts/analyze_experiments.py --config configs/readers.yaml \
    --split test --output-dir runs/
python scripts/paper_tables.py --experiments runs/ --dist /path/to/dist \
    --split test --tex-dir tables/ --all
```

`scripts/*.sbatch` are the Slurm jobs the reported runs were submitted as, on two 80 GB GPUs
per reader. Every path in them is a parameter: `MNT` and `IMAGE` have no default and must be
set for your site, `CODE_DIR` defaults to the submission directory, and `DIST_DIR`, `HOTPOT`,
`CONFIG`, `HF_CACHE` and `ES` derive from those unless overridden. Partition, `--gres` and
memory are site-specific and will need editing.

## Reproducing from the public release

The released dataset carries the items and the Lorepedia background corpus. Three inputs used by
the reported runs are **not** released, because publishing them would undo the separation the
benchmark depends on: the world specifications, the generation prompts, and the per-item
provenance sidecar that records each transposed item's HotpotQA source.

What that means in practice:

- **E1, E4 and E5 run from the public release.** E1's breakdown by construction path and
  generator family needs the provenance sidecar and is skipped without it; the headline closed-book
  numbers are unaffected.
- **E2 and E3 need the withheld inputs.** Both draw replacement distractors from the same invented
  world as the document they displace, which requires the world assignment. They exit with an
  explicit message if those inputs are absent.
- **Paired comparisons against HotpotQA** need the provenance sidecar, which holds the pairing.
  Pass the HotpotQA development distractor split with `--hotpot`.

Researchers who need the withheld inputs for a specific experiment should contact the authors.

`results/` lets every table in the paper be regenerated without rerunning any model.

## Citation

```bibtex
@misc{filgueiras2026lorehopqa,
  title  = {LoreHopQA: Decoupling Answerability from Parametric Knowledge
            in Multi-Hop Retrieval-Augmented Generation},
  author = {Filgueiras, Juan Luis and Fern\'andez-Pichel, Marcos and Parapar, Javier},
  year   = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE). The dataset is distributed separately under its own terms.
