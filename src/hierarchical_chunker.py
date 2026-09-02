import argparse
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from src.config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARENTS_PATH,
)
from src.pdf_parser import parse_pdf


FRONT_MATTER_SECTION_TITLE = (
    "Document Front Matter: "
    "Paper Title and Author Names"
)


def find_front_matter_end(
    documents: list[Document],
) -> int | None:
    """Find the Abstract or Introduction that ends front matter."""

    for index, document in enumerate(
        documents
    ):
        if (
            document.metadata.get(
                "element_type"
            )
            != "Title"
        ):
            continue

        title = " ".join(
            document.page_content
            .lower()
            .split()
        )

        if (
            title == "abstract"
            or title == "introduction"
            or title.endswith(
                " introduction"
            )
        ):
            return index

    return None


def group_elements_by_section(
    documents: list[Document],
    preserve_front_matter: bool = False,
) -> list[tuple[str, list[Document]]]:
    """
    Group parsed elements using Unstructured Title elements.

    When preserve_front_matter is enabled, all content before
    Abstract or Introduction is preserved as one bibliographic
    section containing the paper title, authors and affiliations.
    """

    sections: list[
        tuple[str, list[Document]]
    ] = []

    remaining_documents = documents

    if preserve_front_matter:
        front_matter_end = (
            find_front_matter_end(
                documents
            )
        )

        if (
            front_matter_end is not None
            and front_matter_end > 0
        ):
            sections.append(
                (
                    FRONT_MATTER_SECTION_TITLE,
                    documents[
                        :front_matter_end
                    ],
                )
            )

            remaining_documents = (
                documents[
                    front_matter_end:
                ]
            )

    current_title = (
        "Document Front Matter"
    )

    current_documents: list[
        Document
    ] = []

    for document in remaining_documents:
        element_type = (
            document.metadata.get(
                "element_type"
            )
        )

        if element_type == "Title":
            if current_documents:
                sections.append(
                    (
                        current_title,
                        current_documents,
                    )
                )

                current_documents = []

            current_title = (
                document.page_content.strip()
            )

        else:
            current_documents.append(
                document
            )

    if current_documents:
        sections.append(
            (
                current_title,
                current_documents,
            )
        )

    return sections


def build_parent_child_documents(
    parsed_documents: list[Document],
    preserve_front_matter: bool = False,
) -> tuple[list[Document], list[Document]]:
    """Create larger parent chunks and smaller searchable chunks."""

    if not parsed_documents:
        raise ValueError(
            "No parsed PDF documents were provided."
        )

    parent_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE,
            chunk_overlap=(
                PARENT_CHUNK_OVERLAP
            ),
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )
    )

    child_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=(
                CHILD_CHUNK_OVERLAP
            ),
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )
    )

    sections = group_elements_by_section(
        parsed_documents,
        preserve_front_matter=(
            preserve_front_matter
        ),
    )

    parents: list[Document] = []
    children: list[Document] = []

    source = (
        parsed_documents[0]
        .metadata["source"]
    )

    source_id = Path(source).stem

    for (
        section_title,
        section_documents,
    ) in sections:
        section_text = "\n\n".join(
            document.page_content
            for document
            in section_documents
        )

        page_numbers = [
            document.metadata[
                "page_number"
            ]
            for document
            in section_documents
            if "page_number"
            in document.metadata
        ]

        page_start = (
            min(page_numbers)
            if page_numbers
            else None
        )

        page_end = (
            max(page_numbers)
            if page_numbers
            else None
        )

        parent_pieces = (
            parent_splitter.split_text(
                section_text
            )
        )

        for parent_piece in parent_pieces:
            parent_number = len(
                parents
            )

            parent_id = (
                f"{source_id}-parent-"
                f"{parent_number:04d}"
            )

            parent_text = (
                f"{section_title}\n\n"
                f"{parent_piece}"
            ).strip()

            parent_metadata = {
                "parent_id": parent_id,
                "source": source,
                "section_title": (
                    section_title
                ),
            }

            if page_start is not None:
                parent_metadata[
                    "page_start"
                ] = page_start

                parent_metadata[
                    "page_end"
                ] = page_end

            parent_document = Document(
                page_content=parent_text,
                metadata=parent_metadata,
            )

            parents.append(
                parent_document
            )

            child_pieces = (
                child_splitter.split_text(
                    parent_text
                )
            )

            for (
                child_number,
                child_piece,
            ) in enumerate(
                child_pieces
            ):
                child_id = (
                    f"{parent_id}-child-"
                    f"{child_number:03d}"
                )

                if not child_piece.startswith(
                    section_title
                ):
                    child_piece = (
                        f"{section_title}\n\n"
                        f"{child_piece}"
                    )

                child_metadata = {
                    **parent_metadata,
                    "child_id": child_id,
                }

                children.append(
                    Document(
                        page_content=(
                            child_piece
                        ),
                        metadata=(
                            child_metadata
                        ),
                    )
                )

    if not parents or not children:
        raise ValueError(
            "The PDF did not produce usable "
            "parent and child chunks."
        )

    return parents, children


def create_parent_store(
    parents: list[Document],
) -> dict[str, dict]:
    """Create an in-memory parent lookup dictionary."""

    return {
        document.metadata[
            "parent_id"
        ]: {
            "page_content": (
                document.page_content
            ),
            "metadata": (
                document.metadata
            ),
        }
        for document in parents
    }


def save_parents(
    parents: list[Document],
    output_path: Path = PARENTS_PATH,
) -> None:
    """Persist parent chunks for the local CLI workflow."""

    parent_store = (
        create_parent_store(
            parents
        )
    )

    output_path.write_text(
        json.dumps(
            parent_store,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create parent and child chunks "
            "from a research paper."
        )
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file",
    )

    args = parser.parse_args()

    parsed = parse_pdf(
        args.pdf_path
    )

    (
        parent_documents,
        child_documents,
    ) = build_parent_child_documents(
        parsed
    )

    save_parents(
        parent_documents
    )

    print(
        f"Parsed elements: "
        f"{len(parsed)}"
    )

    print(
        f"Parent chunks: "
        f"{len(parent_documents)}"
    )

    print(
        f"Child chunks: "
        f"{len(child_documents)}"
    )

    print(
        f"Parent store: "
        f"{PARENTS_PATH}"
    )