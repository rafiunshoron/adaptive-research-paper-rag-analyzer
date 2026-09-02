import argparse
import re

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    BM25_K,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    DENSE_K,
    EMBEDDING_MODEL,
)
from src.hierarchical_chunker import (
    build_parent_child_documents,
    save_parents,
)
from src.pdf_parser import parse_pdf


BGE_QUERY_PROMPT = (
    "Represent this sentence for searching relevant passages: "
)


def tokenize_for_bm25(
    text: str,
) -> list[str]:
    """Lowercase and tokenize text for consistent BM25 matching."""

    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def create_embeddings(
    show_progress: bool = True,
) -> HuggingFaceEmbeddings:
    """Create the local CPU-friendly BGE embedding model."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
        query_encode_kwargs={
            "prompt": BGE_QUERY_PROMPT,
            "normalize_embeddings": True,
        },
        show_progress=show_progress,
    )


def create_vector_store(
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    """Connect to the persistent local Chroma collection."""

    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(
            CHROMA_DIR
        ),
    )


def create_runtime_vector_store(
    embeddings: HuggingFaceEmbeddings,
    collection_name: str,
) -> Chroma:
    """Create a session-specific in-memory Chroma collection."""

    if not collection_name.strip():
        raise ValueError(
            "A runtime collection name is required."
        )

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def index_children(
    children: list[Document],
    vector_store: Chroma,
) -> None:
    """Replace the active Chroma index with the supplied children."""

    if not children:
        raise ValueError(
            "No child documents were provided for indexing."
        )

    child_ids = [
        document.metadata["child_id"]
        for document in children
    ]

    vector_store.reset_collection()

    vector_store.add_documents(
        documents=children,
        ids=child_ids,
    )


def load_indexed_children(
    vector_store: Chroma,
) -> list[Document]:
    """Load indexed children so BM25 uses identical retrieval units."""

    stored_data = vector_store.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    documents = (
        stored_data.get("documents")
        or []
    )

    metadatas = (
        stored_data.get("metadatas")
        or []
    )

    if not documents:
        raise ValueError(
            "The Chroma collection is empty. "
            "Index a PDF first."
        )

    return [
        Document(
            page_content=text,
            metadata=metadata,
        )
        for text, metadata in zip(
            documents,
            metadatas,
        )
    ]


def create_hybrid_retriever(
    vector_store: Chroma,
) -> EnsembleRetriever:
    """Create dense and BM25 retrieval with weighted RRF fusion."""

    child_documents = load_indexed_children(
        vector_store
    )

    dense_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": DENSE_K,
        },
    )

    bm25_retriever = (
        BM25Retriever.from_documents(
            child_documents,
            preprocess_func=tokenize_for_bm25,
        )
    )

    bm25_retriever.k = BM25_K

    return EnsembleRetriever(
        retrievers=[
            dense_retriever,
            bm25_retriever,
        ],
        weights=[
            0.5,
            0.5,
        ],
        c=60,
        id_key="child_id",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Index a PDF and test hybrid child retrieval."
        )
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file",
    )

    parser.add_argument(
        "--query",
        default=(
            "What two types of memory does RAG combine?"
        ),
        help="Question used to test retrieval",
    )

    args = parser.parse_args()

    parsed_documents = parse_pdf(
        args.pdf_path
    )

    parents, children = (
        build_parent_child_documents(
            parsed_documents
        )
    )

    save_parents(parents)

    embedding_model = create_embeddings()

    chroma_store = create_vector_store(
        embedding_model
    )

    index_children(
        children,
        chroma_store,
    )

    hybrid_retriever = (
        create_hybrid_retriever(
            chroma_store
        )
    )

    results = hybrid_retriever.invoke(
        args.query
    )

    print(
        f"Indexed children: "
        f"{len(children)}"
    )

    print(
        f"Fused candidates: "
        f"{len(results)}"
    )

    print(f"Query: {args.query}")

    for rank, document in enumerate(
        results[:10],
        start=1,
    ):
        metadata = document.metadata

        print(
            f"{rank}. "
            f"{metadata['child_id']} | "
            f"{metadata['section_title']} | "
            f"pages "
            f"{metadata.get('page_start', '?')}-"
            f"{metadata.get('page_end', '?')}"
        )