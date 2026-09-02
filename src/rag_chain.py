import argparse
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

from src.config import (
    FINAL_CHILD_K,
    GROQ_API_KEY,
    GROQ_MODEL,
)
from src.context_builder import (
    expand_to_parents,
    format_parent_context,
    load_parent_store,
)
from src.hybrid_retriever import (
    create_embeddings,
    create_hybrid_retriever,
    create_vector_store,
)


SYSTEM_PROMPT = """
You are a research-paper analysis assistant.

Follow these rules:

1. Answer using only the supplied research-paper context.
2. Do not use outside knowledge.
3. Cite factual claims using only the exact source labels provided
   in the context, such as [S1] or [S2].
4. Never add line numbers, dagger symbols, page numbers, or modify
   a source label. For example, use [S1], not 【S1†L1-L5】.
5. Never cite a source label that is not present in the context.
6. If the context does not contain enough evidence, respond exactly:
   "I could not find enough information in the uploaded paper to answer this question."
7. Keep the answer clear, direct, and academically written.
""".strip()


def normalize_citations(answer: str) -> str:
    """Normalize citations into the standard [S1] format."""

    normalized_answer = re.sub(
        r"【\s*(S\d+)(?:†[^】]*)?\s*】",
        r"[\1]",
        answer,
    )
    normalized_answer = re.sub(
        r"(?<!\s)(?=\[S\d+\])",
        " ",
        normalized_answer,
    )

    return normalized_answer.strip()


def create_answer_chain():
    """Create the grounded Groq answer-generation chain."""

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY was not found. Add it to the .env file."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            (
                "human",
                """
Research-paper context:

{context}

Question:

{question}

Provide a grounded answer using only exact citations
such as [S1].
""".strip(),
            ),
        ]
    )

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=700,
        timeout=60,
        max_retries=2,
    )

    return (
        prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(normalize_citations)
    )


def answer_question(
    question: str,
    hybrid_retriever,
    parent_store: dict,
    answer_chain,
) -> tuple[str, list]:
    """Retrieve RRF evidence and generate a grounded answer."""

    fused_candidates = hybrid_retriever.invoke(question)
    selected_children = fused_candidates[:FINAL_CHILD_K]

    parent_documents = expand_to_parents(
        selected_children,
        parent_store,
    )

    if not parent_documents:
        return (
            "I could not find enough information in the "
            "uploaded paper to answer this question.",
            [],
        )

    context = format_parent_context(
        parent_documents
    )

    answer = answer_chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return answer, parent_documents


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ask a grounded question about the indexed paper."
        )
    )
    parser.add_argument(
        "--query",
        default=(
            "What two types of memory does RAG combine?"
        ),
        help=(
            "Question to ask about the indexed research paper."
        ),
    )
    args = parser.parse_args()

    embeddings = create_embeddings()
    vector_store = create_vector_store(
        embeddings
    )
    hybrid_retriever = create_hybrid_retriever(
        vector_store
    )

    parent_store = load_parent_store()
    answer_chain = create_answer_chain()

    answer, source_documents = answer_question(
        question=args.query,
        hybrid_retriever=hybrid_retriever,
        parent_store=parent_store,
        answer_chain=answer_chain,
    )

    print(f"\nQuestion:\n{args.query}")
    print(f"\nAnswer:\n{answer}")
    print("\nRetrieved sources:")

    for source_number, document in enumerate(
        source_documents,
        start=1,
    ):
        metadata = document.metadata

        print(
            f"[S{source_number}] "
            f"{metadata.get('source')} | "
            f"{metadata.get('section_title')} | "
            f"pages {metadata.get('page_start', '?')}-"
            f"{metadata.get('page_end', '?')}"
        )


if __name__ == "__main__":
    main()