"""Answer questions with the winning retrieval config and a local Hugging Face LLM (LangChain LCEL)."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .pipeline import chunk, get_embedder, load_pages

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about an employee handbook using ONLY the excerpts below. "
     "Policies differ by office, so use the excerpt for the office in the question. "
     "Cite the page like [p. 12]. If the excerpts do not contain the answer, say \"I don't know.\"\n\n"
     "Excerpts:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs) -> str:
    return "\n\n".join(f"[p. {d.metadata['page']}] {d.page_content}" for d in docs)


def build_retriever(pdf, model: str = "bge-base", chunk_size: int = 400, k: int = 4):
    store = FAISS.from_documents(chunk(load_pages(pdf), chunk_size), get_embedder(model),
                                 distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    return store.as_retriever(search_kwargs={"k": k})


def local_chat_model(model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 200) -> BaseChatModel:
    """HuggingFacePipeline runs the model locally; ChatHuggingFace applies its chat template."""
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        device_map="auto",
        pipeline_kwargs={"max_new_tokens": max_new_tokens, "do_sample": False, "return_full_text": False},
    )
    return ChatHuggingFace(llm=llm)


def build_chain(retriever, chat_model):
    """question -> {answer, sources}"""
    retrieve = RunnableLambda(lambda q: {"question": q, "docs": retriever.invoke(q)})
    generate = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | RunnablePassthrough.assign(answer=PROMPT | chat_model | StrOutputParser())
    )
    return retrieve | generate | RunnableLambda(
        lambda x: {"answer": x["answer"].strip(), "pages": sorted({d.metadata["page"] for d in x["docs"]})}
    )
