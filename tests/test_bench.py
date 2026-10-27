import json
from collections import Counter

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_bench.answer import build_chain, format_docs
from rag_bench.benchmark import _contains, load_golden, run_one, save
from rag_bench.cli import main
from rag_bench.handbook import OFFICES, TOPICS, build_dataset, generate
from rag_bench.pipeline import HashingEmbeddings, chunk, load_pages


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("data"))


def test_generate_is_deterministic_and_unique():
    pages, qas = generate(seed=7)
    assert len(pages) == len(qas) == len(TOPICS) * len(OFFICES) == 200
    assert generate(seed=7)[1] == qas
    # within a topic, every office has a different value (otherwise the question is ambiguous)
    by_topic = Counter((q.topic, q.answer) for q in qas)
    assert max(by_topic.values()) == 1


def test_each_answer_appears_on_its_page(dataset):
    pdf, golden = dataset
    pages = {p.metadata["page"]: p.page_content for p in load_pages(pdf)}
    rows = load_golden(golden)
    assert len(pages) == 200
    assert all(_contains(pages[r["page"]], r["answer"]) for r in rows)


def test_contains_respects_boundaries():
    assert _contains("rate is $81. Questions", "$81")
    assert not _contains("rate is $810.", "$81")
    assert not _contains("ext 41234", "4123")


def test_chunk_sizes(dataset):
    pages = load_pages(dataset[0])
    assert len(chunk(pages, 0)) == 200
    small = chunk(pages, 200)
    assert len(small) > 200 and all(len(c.page_content) <= 200 for c in small)
    assert all("page" in c.metadata for c in small)


def test_run_one_metrics_are_sane(dataset):
    pdf, golden = dataset
    r = run_one("hash", 800, load_pages(pdf), load_golden(golden), embedder=HashingEmbeddings())
    assert 0 <= r.hit_at_1 <= r.hit_at_5 <= 1
    assert r.hit_at_1 <= r.mrr_at_10 <= r.hit_at_5 + 1e-9
    assert r.answer_hit_at_5 <= r.hit_at_5
    assert r.hit_at_5 > 0.3  # the crude lexical baseline still beats random (5/200 = 0.025)


def test_cli_run_writes_reports(dataset, tmp_path):
    pdf, golden = dataset
    out = tmp_path / "res"
    assert main(["run", "--pdf", str(pdf), "--golden", str(golden), "--models", "hash",
                 "--sizes", "400,0", "--out", str(out)]) == 0
    assert (out / "results.csv").exists() and (out / "hit_at_1.png").exists()
    assert "Best configuration" in (out / "summary.md").read_text()


def test_answer_chain_wiring(dataset):
    from langchain_community.vectorstores import FAISS

    store = FAISS.from_documents(chunk(load_pages(dataset[0]), 400), HashingEmbeddings())
    chain = build_chain(store.as_retriever(search_kwargs={"k": 3}), FakeListChatModel(responses=["$81 [p. 23]"]))
    out = chain.invoke("What is the daily meal per diem for Denver office staff?")
    assert out["answer"] == "$81 [p. 23]"
    assert len(out["pages"]) >= 1


def test_format_docs_labels_pages(dataset):
    docs = load_pages(dataset[0])[:2]
    assert format_docs(docs).startswith("[p. 1]")


def test_save_handles_multiple_models(tmp_path, dataset):
    pdf, golden = dataset
    pages, gold = load_pages(pdf), load_golden(golden)[:20]
    res = [run_one("hash", s, pages, gold, embedder=HashingEmbeddings()) for s in (400, 0)]
    res += [run_one("hash2", s, pages, gold, embedder=HashingEmbeddings()) for s in (400, 0)]
    paths = save(res, tmp_path)
    assert json.dumps(str(paths["chart"]))
