"""rag-bench make-data | run | ask"""

from __future__ import annotations

import argparse

DATASETS = {
    # name: (pdf, golden, pages per near-duplicate group, report title)
    "ridgeline": ("data/ridgeline_handbook_200p.pdf", "data/golden_ridgeline.jsonl", 10,
                  "Ridgeline handbook, 200 near-duplicate questions"),
    "lakeview": ("data/lakeview_handbook_200p.pdf", "data/golden_lakeview.jsonl", 0,
                 "Lakeview handbook, 33 needle questions"),
}


def _csv(s: str, cast=str):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rag-bench", description="Chunk-size x embedding-model RAG retrieval benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("make-data", help="Generate the fictional 200-page handbook PDF + 200-question golden set")
    d.add_argument("--out", default="data")
    d.add_argument("--seed", type=int, default=7)

    r = sub.add_parser("run", help="Run the benchmark grid")
    r.add_argument("--dataset", choices=list(DATASETS), default="ridgeline",
                   help="ridgeline = generated near-duplicate handbook; lakeview = needle-in-boilerplate handbook")
    r.add_argument("--pdf", help="Override the dataset's PDF")
    r.add_argument("--golden", help="Override the dataset's golden JSONL")
    r.add_argument("--models", default="minilm,mpnet,bge-small,bge-base",
                   help="Comma list from: hash, minilm, mpnet, bge-small, bge-base")
    r.add_argument("--sizes", default="100,200,400,800,1600,0", help="Chunk sizes in characters; 0 = whole page")
    r.add_argument("--out", default="results")

    a = sub.add_parser("ask", help="Answer a question with a local LLM using the chosen retrieval config")
    a.add_argument("question")
    a.add_argument("--pdf", default="data/ridgeline_handbook_200p.pdf")
    a.add_argument("--model", default="bge-base")
    a.add_argument("--chunk-size", type=int, default=400)
    a.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")

    args = p.parse_args(argv)

    if args.cmd == "make-data":
        from .handbook import build_dataset

        pdf, golden = build_dataset(args.out, args.seed)
        print(f"wrote {pdf}\nwrote {golden}")
    elif args.cmd == "run":
        from .benchmark import run_grid, save

        pdf, golden, group, title = DATASETS[args.dataset]
        res = run_grid(args.pdf or pdf, args.golden or golden, _csv(args.models), _csv(args.sizes, int),
                       pages_per_topic=0 if args.pdf else group)
        for k, v in save(res, args.out, title).items():
            print(f"{k}: {v}")
    elif args.cmd == "ask":
        from .answer import build_chain, build_retriever, local_chat_model

        chain = build_chain(build_retriever(args.pdf, args.model, args.chunk_size), local_chat_model(args.llm))
        out = chain.invoke(args.question)
        print(f"\n{out['answer']}\n\nretrieved pages: {out['pages']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
