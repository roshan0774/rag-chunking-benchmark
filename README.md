# RAG Chunking & Embedding Benchmark

[![CI](https://github.com/roshan0774/rag-chunking-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/roshan0774/rag-chunking-benchmark/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roshan0774/rag-chunking-benchmark/blob/main/notebooks/colab_benchmark.ipynb)

**What chunk size should a RAG pipeline use?** The usual answer is "800 characters with some overlap". This repo measures it instead, on two 200-page handbooks built to fail in opposite ways. The result: **the best chunk size depends on how the document is written.** On one handbook the smallest chunks win; on the other they lose.

Every combination of embedding model and chunk size is scored on questions with verified answers, along with latency and index size. It finishes with a local LLM answering questions over the winning configuration.

## Two datasets, two failure modes

| | **Lakeview handbook** | **Ridgeline handbook** |
|---|---|---|
| File | [`data/lakeview_handbook_200p.pdf`](data/lakeview_handbook_200p.pdf) | [`data/ridgeline_handbook_200p.pdf`](data/ridgeline_handbook_200p.pdf) (built by `rag-bench make-data`) |
| Structure | 10 chapters × 20 sections: HR, security, data platform, analytics, AI/ML, finance, IT, compliance, customer ops | 20 policy topics × 10 regional offices |
| Hard part | **Needle in boilerplate.** Thousands of sentences of templated policy filler hide about 30 real facts on 29 pages | **Near-duplicates.** The 10 pages on each topic are identical except for the office name and one value |
| Questions | 33, each tied to a single page; a test confirms every answer appears exactly once in the document | 200, one per page, worded differently from the page text |
| Example | *"Who must approve backfills covering more than 90 days?"* → **Tomasz Reyes** (p. 77) | *"How much is the daily food allowance for travelers from the Denver office?"* → the Denver page's value, not Phoenix's |

Sample needle facts from Lakeview: HQ at 4180 Harbor Point Drive (p. 4); production Snowflake account `LVDS_PROD_EAST` (p. 62); Airflow DAGs prefixed `lvds_` with 3 retries (p. 65); 320 GPU-hours per team per month (p. 113); P1 tickets answered within 15 minutes (p. 184).

## Metrics

| Metric | Meaning |
|---|---|
| `hit_at_1`, `hit_at_5` | The correct page is the top result / among the top 5 |
| `mrr_at_10` | Mean reciprocal rank of the first chunk from the correct page |
| `answer_hit_at_5` | A top-5 chunk is from the correct page **and contains the exact answer**. Big chunks can hit the right page while burying the fact; tiny chunks can cut the fact off from its context |
| `wrong_office_at_1` | Ridgeline only: the top result is the right topic but another office's page |
| `query_ms_p50`, `index_seconds`, `index_mb` | Latency, indexing time and vector memory: the cost side |

**Grid:** `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `bge-small-en-v1.5`, `bge-base-en-v1.5` (with the BGE query instruction), plus an offline hashing baseline, crossed with chunk sizes of 100, 200, 400, 800 and 1600 characters (15% overlap) and whole page.

## Results: offline baseline

CI reruns these on every push using the hashing baseline, a bag-of-words vectorizer that needs no model download:

| Chunk size | Lakeview hit@1 | Lakeview answer_hit@5 | Ridgeline hit@1 | Ridgeline answer_hit@5 | Ridgeline wrong_office@1 |
|---|---|---|---|---|---|
| 100 | **0.727** | 0.394 | 0.230 | 0.205 | 0.205 |
| 200 | **0.727** | 0.667 | 0.430 | 0.330 | 0.075 |
| 400 | 0.667 | 0.576 | 0.380 | 0.545 | 0.055 |
| 800 | 0.545 | **0.697** | 0.405 | 0.735 | 0.130 |
| 1600 | 0.424 | 0.636 | **0.520** | **0.775** | 0.080 |
| whole page | 0.424 | 0.636 | **0.520** | **0.775** | 0.080 |

What the trends show:
- **Lakeview: small chunks win on hit@1.** A one-sentence fact inside a 2,500-character page gets diluted by the boilerplate around it. Hit@1 falls from 0.73 to 0.42 as chunks grow. But 100-character chunks often cut the answer itself in half (answer_hit@5 is only 0.39), so **200 is the practical sweet spot**.
- **Ridgeline: large chunks win.** Small chunks separate the office name from its value. At 100 characters, 1 in 5 top results is the right policy for the wrong office.
- **Cost:** 100-character chunks produce 16–31 times more vectors than whole pages (6,185 vs 200 on Lakeview).

Full tables: [`results/baseline_hash_lakeview/`](results/baseline_hash_lakeview/summary.md), [`results/baseline_hash_ridgeline/`](results/baseline_hash_ridgeline/summary.md).

**Neural embedding models:** run the [Colab notebook](notebooks/colab_benchmark.ipynb) on a free T4 GPU, which takes about 15–20 minutes for both datasets. It writes `results/full_lakeview/` and `results/full_ridgeline/` with summary tables and hit@1 heatmaps.

## Quick start

```bash
git clone https://github.com/roshan0774/rag-chunking-benchmark.git
cd rag-chunking-benchmark
pip install -r requirements.txt

rag-bench run --dataset lakeview                     # 4 models x 6 chunk sizes -> results/
rag-bench run --dataset ridgeline --models bge-small,bge-base --sizes 200,400,800
rag-bench run --dataset lakeview --models hash       # offline, no downloads

rag-bench ask "Who chairs the model review board?" --pdf data/lakeview_handbook_200p.pdf --chunk-size 200
rag-bench ask "How many days a week can Austin employees work from home?" --chunk-size 1600
```

`ask` builds a LangChain LCEL chain: retriever → grounded prompt (page citations, "I don't know" fallback) → `Qwen2.5-1.5B-Instruct`, run locally through `HuggingFacePipeline` + `ChatHuggingFace`. To benchmark your own document, pass `--pdf` and a `--golden` JSONL with rows of the form `{"question", "answer", "page"}`.

## Project layout

```
src/rag_bench/
  handbook.py    generates the Ridgeline handbook + 200 golden questions (seeded, reproducible)
  pipeline.py    PDF loading, chunking, embedding backends (HF models + offline hashing baseline)
  benchmark.py   grid runner, metrics, CSV/Markdown/heatmap reports (raw FAISS IndexFlatIP)
  answer.py      LCEL RAG chain with a local Hugging Face chat model
  cli.py         make-data | run | ask
data/            both handbooks + golden sets
results/         saved benchmark outputs
tests/           offline pytest suite (hashing embeddings, fake chat model, golden-set integrity)
```

## What I'd test next

- **Hybrid BM25 + dense retrieval.** Lakeview's needles are full of exact tokens (`LVDS_PROD_EAST`, `AI-22`, `#sec-alert-7`) that keyword search finds easily
- **Boilerplate filtering.** Drop sentences that repeat across sections before indexing
- **Metadata filtering** for Ridgeline: pull the office out of the query and filter before vector search
- **Parent-child retrieval:** search small chunks, return the whole page to the LLM, to get both datasets' winning behavior at once
- End-to-end answer accuracy with the LLM, not just retrieval

## Note on the data

Both handbooks are fictional. *Lakeview Data Systems* was generated for RAG practice. *Ridgeline Group* is generated from a fixed seed by `src/rag_bench/handbook.py`. Any resemblance to real organizations or people is coincidental.

## License

MIT
