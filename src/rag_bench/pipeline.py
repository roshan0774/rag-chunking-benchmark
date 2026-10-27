"""Corpus loading, chunking, and embedding backends."""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Short name -> (Hugging Face model id, query prefix). BGE models retrieve better with a query instruction.
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
    "bge-base": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: "),
}


def load_pages(pdf: str | Path) -> list[Document]:
    pages = []
    for d in PyPDFLoader(str(pdf)).load():
        text = re.sub(r"[ \t]+", " ", re.sub(r"-\n(\w)", r"\1", d.page_content)).strip()
        if text:
            pages.append(Document(page_content=text, metadata={"page": int(d.metadata.get("page", 0)) + 1}))
    return pages


def chunk(pages: list[Document], size: int, overlap_ratio: float = 0.15) -> list[Document]:
    """size <= 0 means one chunk per page (page-level baseline)."""
    if size <= 0:
        return [Document(page_content=p.page_content, metadata=dict(p.metadata)) for p in pages]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=int(size * overlap_ratio),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)


class HashingEmbeddings(Embeddings):
    """Offline lexical baseline (hashed unigrams + bigrams). No download; used in tests and CI."""

    dim = 1024
    _tok = re.compile(r"[a-z0-9$]+")

    def _vec(self, text: str) -> list[float]:
        toks = self._tok.findall(text.lower())
        v = [0.0] * self.dim
        for f in toks + [f"{a}_{b}" for a, b in itertools.pairwise(toks)]:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class PrefixedQueries(Embeddings):
    def __init__(self, inner: Embeddings, prefix: str):
        self.inner, self.prefix = inner, prefix

    def embed_documents(self, texts):
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        return self.inner.embed_query(self.prefix + text)


def get_embedder(name: str, device: str | None = None, batch_size: int = 64) -> Embeddings:
    if name == "hash":
        return HashingEmbeddings()
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from: hash, {', '.join(MODELS)}")
    from langchain_huggingface import HuggingFaceEmbeddings

    model_id, prefix = MODELS[name]
    kw = {"device": device} if device else {}
    hf = HuggingFaceEmbeddings(model_name=model_id, model_kwargs=kw,
                               encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size})
    return PrefixedQueries(hf, prefix) if prefix else hf
