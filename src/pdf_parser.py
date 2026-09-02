import argparse
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf


SKIPPED_ELEMENT_TYPES = {
    "Header",
    "Footer",
    "PageBreak",
}

SUPPORTED_STRATEGIES = {
    "fast",
    "hi_res",
}


def parse_pdf(
    pdf_path: str | Path,
    strategy: str = "hi_res",
) -> list[Document]:
    """Parse a research-paper PDF into structured documents."""

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file: {pdf_path}"
        )

    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported PDF strategy: {strategy}"
        )

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy=strategy,
        infer_table_structure=(
            strategy == "hi_res"
        ),
        languages=["eng"],
    )

    documents: list[Document] = []

    for element_index, element in enumerate(
        elements
    ):
        text = str(element).strip()

        element_type = getattr(
            element,
            "category",
            "Unknown",
        )

        if (
            not text
            or element_type
            in SKIPPED_ELEMENT_TYPES
        ):
            continue

        page_number = getattr(
            element.metadata,
            "page_number",
            None,
        )

        table_html = getattr(
            element.metadata,
            "text_as_html",
            None,
        )

        metadata = {
            "source": pdf_path.name,
            "element_type": element_type,
            "element_index": element_index,
        }

        if page_number is not None:
            metadata["page_number"] = int(
                page_number
            )

        if table_html:
            metadata["text_as_html"] = table_html

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    if not documents:
        raise ValueError(
            "No usable content was extracted "
            f"from {pdf_path.name}"
        )

    return documents


def print_summary(
    documents: list[Document],
) -> None:
    """Print extracted document-element counts."""

    element_counts = Counter(
        document.metadata["element_type"]
        for document in documents
    )

    print(
        f"Extracted elements: "
        f"{len(documents)}"
    )

    for element_type, count in (
        element_counts.most_common()
    ):
        print(f"{element_type}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Parse a research-paper PDF."
        )
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file",
    )

    parser.add_argument(
        "--strategy",
        choices=sorted(
            SUPPORTED_STRATEGIES
        ),
        default="hi_res",
        help=(
            "PDF parsing strategy. Use fast for "
            "text-based PDFs and hi_res for "
            "layout-sensitive local processing."
        ),
    )

    args = parser.parse_args()

    parsed_documents = parse_pdf(
        args.pdf_path,
        strategy=args.strategy,
    )

    print_summary(parsed_documents)