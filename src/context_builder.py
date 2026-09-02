import argparse
import json

from langchain_core.documents import Document

from src.config import (
    FINAL_CHILD_K,
    MAX_PARENT_CONTEXTS,
    PARENTS_PATH,
)
from src.hybrid_retriever import (
    create_embeddings,
    create_hybrid_retriever,
    create_vector_store,
)


def load_parent_store() -> dict:
    """Load parent sections from the JSON parent store."""

    if not PARENTS_PATH.exists():
        raise FileNotFoundError(
            f"Parent store not found: {PARENTS_PATH}"
        )

    with PARENTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def expand_to_parents(
    child_documents: list[Document],
    parent_store: dict,
) -> list[Document]:
    """
    Replace retrieved child chunks with their larger parent sections.

    Parent order follows the RRF child ranking.
    Duplicate parents are removed.
    """

    selected_parents = []
    seen_parent_ids = set()

    for child in child_documents:
        parent_id = child.metadata.get("parent_id")

        if not parent_id or parent_id in seen_parent_ids:
            continue

        parent_record = parent_store.get(parent_id)

        if parent_record is None:
            continue

        parent_document = Document(
            page_content=parent_record["page_content"],
            metadata=parent_record["metadata"],
        )

        selected_parents.append(parent_document)
        seen_parent_ids.add(parent_id)

        if len(selected_parents) >= MAX_PARENT_CONTEXTS:
            break

    return selected_parents


def format_parent_context(
    parent_documents: list[Document],
) -> str:
    """Format parent sections for the final LLM prompt."""

    formatted_sections = []

    for source_number, document in enumerate(
        parent_documents,
        start=1,
    ):
        metadata = document.metadata

        source_header = (
            f"[S{source_number}] "
            f"{metadata.get('source', 'Unknown source')} | "
            f"Section: "
            f"{metadata.get('section_title', 'Unknown section')} | "
            f"Pages: {metadata.get('page_start', '?')}-"
            f"{metadata.get('page_end', '?')}"
        )

        formatted_sections.append(
            f"{source_header}\n{document.page_content}"
        )

    return "\n\n".join(formatted_sections)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test RRF child-to-parent context expansion."
    )
    parser.add_argument(
        "--query",
        default="What two types of memory does RAG combine?",
        help="Question used to test parent expansion.",
    )
    args = parser.parse_args()

    embeddings = create_embeddings()
    vector_store = create_vector_store(embeddings)
    hybrid_retriever = create_hybrid_retriever(
        vector_store
    )

    fused_candidates = hybrid_retriever.invoke(
        args.query
    )
    selected_children = fused_candidates[:FINAL_CHILD_K]

    parent_store = load_parent_store()
    parent_documents = expand_to_parents(
        selected_children,
        parent_store,
    )

    context = format_parent_context(parent_documents)

    print(f"\nQuery: {args.query}")
    print(f"Fused RRF candidates: {len(fused_candidates)}")
    print(f"Selected children: {len(selected_children)}")
    print(f"Selected unique parents: {len(parent_documents)}")
    print(f"Context characters: {len(context)}\n")

    for rank, parent in enumerate(
        parent_documents,
        start=1,
    ):
        metadata = parent.metadata
        preview = " ".join(
            parent.page_content.split()
        )[:400]

        print(
            f"{rank}. {metadata.get('parent_id')} | "
            f"{metadata.get('section_title')} | "
            f"pages {metadata.get('page_start', '?')}-"
            f"{metadata.get('page_end', '?')}"
        )
        print(f"   {preview}\n")


if __name__ == "__main__":
    main()