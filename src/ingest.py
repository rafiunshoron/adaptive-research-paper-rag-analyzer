import argparse
import time
from pathlib import Path

from src.config import CHROMA_DIR, PARENTS_PATH
from src.hierarchical_chunker import (
    build_parent_child_documents,
    save_parents,
)
from src.hybrid_retriever import (
    create_embeddings,
    create_vector_store,
    index_children,
)
from src.pdf_parser import parse_pdf


def ingest_pdf(
    pdf_path: str | Path,
) -> dict:
    """
    Parse, chunk, and index one research-paper PDF.

    The active Chroma collection and parent store are replaced
    whenever a new paper is ingested.
    """

    pdf_path = Path(pdf_path)
    start_time = time.perf_counter()

    parsed_documents = parse_pdf(
        pdf_path
    )

    parent_documents, child_documents = (
        build_parent_child_documents(
            parsed_documents
        )
    )

    if not parent_documents:
        raise ValueError(
            "No parent chunks were created."
        )

    if not child_documents:
        raise ValueError(
            "No child chunks were created."
        )

    save_parents(
        parent_documents
    )

    embeddings = create_embeddings()

    vector_store = create_vector_store(
        embeddings
    )

    index_children(
        child_documents,
        vector_store,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    return {
        "pdf": pdf_path.name,
        "parsed_elements": len(
            parsed_documents
        ),
        "parent_chunks": len(
            parent_documents
        ),
        "child_chunks": len(
            child_documents
        ),
        "elapsed_seconds": round(
            elapsed_seconds,
            2,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest one research-paper PDF into the RAG system."
        )
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the research-paper PDF.",
    )

    args = parser.parse_args()

    result = ingest_pdf(
        args.pdf_path
    )

    print("\nIngestion completed")
    print(f"PDF: {result['pdf']}")
    print(
        f"Parsed elements: "
        f"{result['parsed_elements']}"
    )
    print(
        f"Parent chunks: "
        f"{result['parent_chunks']}"
    )
    print(
        f"Child chunks: "
        f"{result['child_chunks']}"
    )
    print(
        f"Elapsed time: "
        f"{result['elapsed_seconds']} seconds"
    )
    print(f"Parent store: {PARENTS_PATH}")
    print(f"Chroma directory: {CHROMA_DIR}")


if __name__ == "__main__":
    main()