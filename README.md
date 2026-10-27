# RAG Chunking & Embedding Benchmark

[![CI](https://github.com/roshan0774/rag-chunking-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/roshan0774/rag-chunking-benchmark/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roshan0774/rag-chunking-benchmark/blob/main/notebooks/colab_benchmark.ipynb)

**How much do chunk size and embedding model change RAG retrieval quality?** This repo measures it on a 200-page policy handbook instead of guessing. Every combination of model and chunk size is scored on 200 questions with known answers, along with speed and index size. It finishes with a local LLM answering questions over the best configuration.

## Why this dataset is hard

Most RAG demos use documents where every page is about something different. Real policy manuals, benefit guides and claims manuals aren't like that. The handbook here (fictional *Lakeview Group*) has **20 policy topics × 10 regional offices**. The 10 pages on a topic are nearly identical: only the office name and one value differ.

> *Section 3.3 Meal Per Diem: Denver Office.* "...The daily meal per diem for Denver office staff is **$81**..."
> *Section 3.7 Meal Per Diem: Phoenix Office.* "...The daily meal per diem for Phoenix office staff is **$66**..."

Questions are paraphrased so they don't copy the page wording. For example: *"How much is the daily food allowance for travelers from the Denver office?"* A retriever that only matches the topic lands on the wrong office's page, and the LLM then confidently gives the wrong number. The benchmark tracks this failure separately as `wrong_office_at_1`.

## Metrics

| Metric | Meaning |
|---|---|
| `hit_at_1`, `hit_at_5` | The correct page is the top result / among the top 5 |
| `mrr_at_10` | Mean reciprocal rank of the first chunk from the correct page |
| `answer_hit_at_5` | A top-5 chunk is from the correct page **and contains the exact answer value**. Small chunks can hit the right page but miss the fact |
| `wrong_office_at_1` | The top result is the right topic but another office's page (near-duplicate confusion) |
| `query_ms_p50`, `index_seconds`, `index_mb` | Latency, indexing time and vector memory: the cost side of the trade-off |

## Grid

| Embedding models | Chunk sizes (characters, 15% overlap) |
|---|---|
| `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `bge-small-en-v1.5`, `bge-base-en-v1.5` (with query instruction), plus an offline hashing baseline | 100, 200, 400, 800, 1600, whole page |

## Results

**Offline lexical baseline** (hashed bag-of-words, no neural model). This is what CI runs on every push:

| chunk | chunks | hit@1 | hit@5 | MRR@10 | answer_hit@5 | wrong_office@1 |
|---|---|---|---|---|---|---|
| 100 | 3187 | 0.230 | 0.485 | 0.325 | 0.205 | 0.205 |
| 200 | 1375 | 0.440 | 0.675 | 0.534 | 0.330 | 0.065 |
| 400 | 606 | 0.380 | 0.605 | 0.488 | 0.550 | 0.055 |
| 800 | 400 | 0.420 | 0.735 | 0.559 | 0.735 | 0.125 |
| 1600 | 200 | 0.515 | 0.775 | 0.620 | 0.775 | 0.080 |
| page | 200 | 0.515 | 0.775 | 0.620 | 0.775 | 0.080 |

Even this baseline shows the main effect. Very small chunks (100 characters) split the office name away from the value. They score worst on every metric, including a 20% wrong-office rate, and 100-character chunks also make the index 16 times larger than one chunk per page.

**Neural embedding models:** run the [Colab notebook](notebooks/colab_benchmark.ipynb) on a free T4 GPU, which takes about 10 minutes. It writes `results/full/summary.md`, `results.csv` and the `hit_at_1.png` heatmap.

## Quick start

```bash
git clone https://github.com/roshan0774/rag-chunking-benchmark.git
cd rag-chunking-benchmark
pip install -r requirements.txt

rag-bench make-data                                 # 200-page PDF + 200-question golden set
rag-bench run                                       # full grid -> results/
rag-bench run --models bge-small,bge-base --sizes 200,400,800
rag-bench ask "How many days a week can Austin employees work from home?" --model bge-base --chunk-size 400
```

`ask` builds a LangChain LCEL chain: retriever → grounded prompt (office-aware, page citations, "I don't know" fallback) → `Qwen2.5-1.5B-Instruct`, run locally through `HuggingFacePipeline` + `ChatHuggingFace`.

## Project layout

```
src/rag_bench/
  handbook.py    generates the 200-page fictional handbook + golden questions (seeded, reproducible)
  pipeline.py    PDF loading, chunking, embedding backends (HF models + offline hashing baseline)
  benchmark.py   grid runner, metrics, CSV/Markdown/heatmap reports (raw FAISS IndexFlatIP)
  answer.py      LCEL RAG chain with a local Hugging Face chat model
  cli.py         make-data | run | ask
tests/           offline pytest suite (hashing embeddings, fake chat model)
results/         saved benchmark outputs
```

## What I'd test next

- Hybrid BM25 + dense retrieval and a cross-encoder reranker. These should cut `wrong_office_at_1` the most
- Metadata filtering (extract the office from the query and filter before vector search)
- Semantic and heading-aware chunking compared with fixed character windows
- End-to-end answer accuracy with the LLM, not just retrieval

## Note on the data

*Lakeview Group* is fictional. Names, offices and policy values are generated at random from a fixed seed.

## License

MIT
