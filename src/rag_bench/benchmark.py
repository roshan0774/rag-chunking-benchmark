"""Grid benchmark: embedding model x chunk size -> retrieval quality, speed, index size."""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from .pipeline import chunk, get_embedder, load_pages


@dataclass
class Result:
    model: str
    chunk_size: int          # 0 = whole page
    n_chunks: int
    hit_at_1: float          # top-1 chunk is from the correct page
    hit_at_5: float
    mrr_at_10: float
    answer_hit_at_5: float   # a top-5 chunk from the correct page contains the exact answer value
    wrong_office_at_1: float | None  # top-1 is the right topic but another office's page (Ridgeline only)
    index_seconds: float
    query_ms_p50: float
    index_mb: float


def load_golden(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _contains(text: str, value: str) -> bool:
    return re.search(rf"(?<![\w$.]){re.escape(value)}(?![\w])", " ".join(text.split())) is not None


def run_one(model: str, chunk_size: int, pages, golden, embedder=None, pages_per_topic: int = 10) -> Result:
    emb = embedder or get_embedder(model)
    chunks = chunk(pages, chunk_size)
    texts = [c.page_content for c in chunks]
    chunk_pages = np.array([c.metadata["page"] for c in chunks])

    t0 = time.perf_counter()
    vecs = np.asarray(emb.embed_documents(texts), dtype="float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    index_s = time.perf_counter() - t0

    hits1 = hits5 = rr = ans5 = wrong_office = 0
    lat = []
    for g in golden:
        t = time.perf_counter()
        q = np.asarray([emb.embed_query(g["question"])], dtype="float32")
        _, ids = index.search(q, min(10, len(texts)))
        lat.append((time.perf_counter() - t) * 1000)
        ids = ids[0]
        pg = chunk_pages[ids]
        target = g["page"]
        hits1 += pg[0] == target
        hits5 += target in pg[:5]
        rank = next((i + 1 for i, p in enumerate(pg) if p == target), None)
        rr += 1 / rank if rank else 0
        ans5 += any(pg[i] == target and _contains(texts[ids[i]], g["answer"]) for i in range(min(5, len(ids))))
        if pages_per_topic:
            same_topic = (pg[0] - 1) // pages_per_topic == (target - 1) // pages_per_topic
            wrong_office += same_topic and pg[0] != target

    n = len(golden)
    return Result(
        model=model, chunk_size=chunk_size, n_chunks=len(chunks),
        hit_at_1=round(hits1 / n, 3), hit_at_5=round(hits5 / n, 3), mrr_at_10=round(rr / n, 3),
        answer_hit_at_5=round(ans5 / n, 3),
        wrong_office_at_1=round(wrong_office / n, 3) if pages_per_topic else None,
        index_seconds=round(index_s, 2), query_ms_p50=round(statistics.median(lat), 2),
        index_mb=round(vecs.nbytes / 1e6, 2),
    )


def run_grid(pdf, golden_path, models: list[str], sizes: list[int], pages_per_topic: int = 10,
             log=print) -> list[Result]:
    pages = load_pages(pdf)
    golden = load_golden(golden_path)
    log(f"{len(pages)} pages, {len(golden)} questions")
    results = []
    for m in models:
        emb = get_embedder(m)  # load each model once
        for s in sizes:
            r = run_one(m, s, pages, golden, embedder=emb, pages_per_topic=pages_per_topic)
            log(f"  {m:10s} chunk={s or 'page':>5}  chunks={r.n_chunks:5d}  hit@1={r.hit_at_1:.3f}  "
                f"hit@5={r.hit_at_5:.3f}  mrr={r.mrr_at_10:.3f}  ans@5={r.answer_hit_at_5:.3f}  "
                f"wrong_office@1={r.wrong_office_at_1 if r.wrong_office_at_1 is not None else 'n/a'}")
            results.append(r)
    return results


def save(results: list[Result], out_dir: str | Path, title: str = "handbook benchmark") -> dict[str, Path]:
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in results])
    df["chunk"] = df["chunk_size"].map(lambda s: "page" if s == 0 else str(s))
    df.drop(columns="chunk").to_csv(out / "results.csv", index=False)

    cols = ["model", "chunk", "n_chunks", "hit_at_1", "hit_at_5", "mrr_at_10", "answer_hit_at_5",
            "wrong_office_at_1", "query_ms_p50", "index_mb"]
    if df["wrong_office_at_1"].isna().all():
        cols.remove("wrong_office_at_1")
    best = df.sort_values(["hit_at_1", "mrr_at_10"], ascending=False).iloc[0]
    md = [
        f"# Benchmark results: {title}\n",
        f"Best configuration: **{best['model']}**, chunk size **{best['chunk']}**, "
        f"hit@1 = {best['hit_at_1']:.3f}, MRR@10 = {best['mrr_at_10']:.3f}\n",
        df[cols].to_markdown(index=False),
        "",
    ]
    (out / "summary.md").write_text("\n".join(md))

    paths = {"csv": out / "results.csv", "md": out / "summary.md"}
    try:
        paths["chart"] = _chart(df, out / "hit_at_1.png", title)
    except ImportError:
        pass
    return paths


def _chart(df, path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = df.drop_duplicates("chunk").sort_values("chunk_size", key=lambda s: s.replace(0, 10**9))["chunk"]
    pivot = df.pivot(index="model", columns="chunk", values="hit_at_1")[list(order)]
    fig, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 2.5, 0.6 * pivot.shape[0] + 1.6))
    im = ax.imshow(pivot.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.set_xlabel("chunk size (characters)")
    ax.set_title(f"Retrieval hit@1 ({title})")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
