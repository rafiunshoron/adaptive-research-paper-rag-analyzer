import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st


# Load Streamlit Cloud secrets before importing src.config.
if not os.getenv("GROQ_API_KEY"):
    try:
        cloud_api_key = st.secrets.get(
            "GROQ_API_KEY"
        )

        if cloud_api_key:
            os.environ[
                "GROQ_API_KEY"
            ] = cloud_api_key

    except Exception:
        pass


from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from src.conversation import (
    create_conversation_graph,
)
from src.hierarchical_chunker import (
    build_parent_child_documents,
    create_parent_store,
)
from src.hybrid_retriever import (
    create_embeddings,
    create_hybrid_retriever,
    create_runtime_vector_store,
    index_children,
)
from src.pdf_parser import parse_pdf


MAX_FILE_SIZE_MB = 20


st.set_page_config(
    page_title=(
        "Research Paper RAG Analyzer"
    ),
    page_icon="📄",
    layout="wide",
)


@st.cache_resource(
    show_spinner=False
)
def load_embedding_model():
    """Load the embedding model once per application process."""

    return create_embeddings(
        show_progress=False
    )


def initialize_session_state() -> None:
    """Initialize browser-session state."""

    defaults = {
        "graph": None,
        "vector_store": None,
        "hybrid_retriever": None,
        "parent_store": None,
        "thread_id": None,
        "paper_name": None,
        "messages": [],
        "paper_stats": None,
        "notice": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = (
                value
            )


def create_new_conversation() -> None:
    """Start a new runtime conversation for the active paper."""

    checkpointer = InMemorySaver()

    st.session_state.graph = (
        create_conversation_graph(
            checkpointer=checkpointer,
            hybrid_retriever=(
                st.session_state
                .hybrid_retriever
            ),
            parent_store=(
                st.session_state
                .parent_store
            ),
        )
    )

    st.session_state.thread_id = str(
        uuid.uuid4()
    )

    st.session_state.messages = []


def delete_runtime_collection() -> None:
    """Delete the active temporary Chroma collection."""

    vector_store = (
        st.session_state.vector_store
    )

    if vector_store is not None:
        try:
            vector_store.delete_collection()
        except Exception:
            pass


def clear_active_paper() -> None:
    """Clear the uploaded paper and runtime conversation."""

    delete_runtime_collection()

    st.session_state.graph = None
    st.session_state.vector_store = None
    st.session_state.hybrid_retriever = None
    st.session_state.parent_store = None
    st.session_state.thread_id = None
    st.session_state.paper_name = None
    st.session_state.messages = []
    st.session_state.paper_stats = None


def process_uploaded_pdf(
    uploaded_file,
) -> None:
    """Parse, chunk, embed and index one uploaded PDF."""

    file_size_mb = (
        uploaded_file.size
        / (1024 * 1024)
    )

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"PDF must be smaller than "
            f"{MAX_FILE_SIZE_MB} MB."
        )

    temporary_path = None
    new_vector_store = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temporary_file:
            temporary_file.write(
                uploaded_file.getvalue()
            )

            temporary_path = Path(
                temporary_file.name
            )

        parsed_documents = parse_pdf(
            temporary_path,
            strategy="fast",
        )

        # Use the original uploaded filename in citations.
        for document in parsed_documents:
            document.metadata[
                "source"
            ] = uploaded_file.name

        # Runtime mode preserves title-page and author information.
        parents, children = (
            build_parent_child_documents(
                parsed_documents,
                preserve_front_matter=True,
            )
        )

        parent_store = (
            create_parent_store(
                parents
            )
        )

        embeddings = (
            load_embedding_model()
        )

        collection_name = (
            f"paper-"
            f"{uuid.uuid4().hex}"
        )

        new_vector_store = (
            create_runtime_vector_store(
                embeddings=embeddings,
                collection_name=(
                    collection_name
                ),
            )
        )

        index_children(
            children=children,
            vector_store=(
                new_vector_store
            ),
        )

        hybrid_retriever = (
            create_hybrid_retriever(
                new_vector_store
            )
        )

        checkpointer = (
            InMemorySaver()
        )

        graph = (
            create_conversation_graph(
                checkpointer=(
                    checkpointer
                ),
                hybrid_retriever=(
                    hybrid_retriever
                ),
                parent_store=(
                    parent_store
                ),
            )
        )

        delete_runtime_collection()

        st.session_state.vector_store = (
            new_vector_store
        )

        st.session_state.hybrid_retriever = (
            hybrid_retriever
        )

        st.session_state.parent_store = (
            parent_store
        )

        st.session_state.graph = (
            graph
        )

        st.session_state.thread_id = str(
            uuid.uuid4()
        )

        st.session_state.paper_name = (
            uploaded_file.name
        )

        st.session_state.messages = []

        st.session_state.paper_stats = {
            "elements": len(
                parsed_documents
            ),
            "parents": len(
                parents
            ),
            "children": len(
                children
            ),
        }

        st.session_state.notice = (
            f"{uploaded_file.name} "
            f"is ready."
        )

    except Exception:
        if new_vector_store is not None:
            try:
                new_vector_store.delete_collection()
            except Exception:
                pass

        raise

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink(
                missing_ok=True
            )


def render_sources(
    sources: list[dict],
    retrieval_query: str | None = None,
) -> None:
    """Display source metadata and the standalone query."""

    if not sources:
        return

    with st.expander(
        "Sources and retrieval details"
    ):
        if retrieval_query:
            st.caption(
                "Retrieval query: "
                f"{retrieval_query}"
            )

        for (
            source_number,
            source,
        ) in enumerate(
            sources,
            start=1,
        ):
            st.markdown(
                f"**[S{source_number}]** "
                f"{source.get('source', 'Unknown source')}  \n"
                f"{source.get('section_title', 'Unknown section')} · "
                f"pages "
                f"{source.get('page_start', '?')}-"
                f"{source.get('page_end', '?')}"
            )


def render_route(route: str | None) -> None:
    """Show which adaptive graph branch handled the message."""

    if route == "direct":
        st.caption("Route: Direct response · retrieval skipped")

    elif route == "retrieval":
        st.caption("Route: Paper retrieval · hybrid search used")


initialize_session_state()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header(
        "Research Paper RAG"
    )

    st.caption(
        "Upload one text-based research PDF "
        "and ask grounded questions."
    )

    uploaded_pdf = st.file_uploader(
        "Upload research paper",
        type=["pdf"],
        accept_multiple_files=False,
    )

    process_clicked = st.button(
        "Process paper",
        type="primary",
        use_container_width=True,
        disabled=(
            uploaded_pdf is None
        ),
    )

    if process_clicked:
        try:
            with st.spinner(
                "Parsing and indexing "
                "the paper..."
            ):
                process_uploaded_pdf(
                    uploaded_pdf
                )

            st.rerun()

        except Exception as error:
            st.error(
                "Could not process the PDF: "
                f"{error}"
            )

    if st.session_state.paper_name:
        st.divider()

        st.subheader(
            "Active paper"
        )

        st.write(
            "📄 "
            f"{st.session_state.paper_name}"
        )

        statistics = (
            st.session_state.paper_stats
        )

        if statistics:
            st.caption(
                f"{statistics['parents']} "
                f"parent chunks · "
                f"{statistics['children']} "
                f"searchable chunks"
            )

        if st.button(
            "New conversation",
            use_container_width=True,
        ):
            create_new_conversation()
            st.rerun()

        if st.button(
            "Remove paper",
            use_container_width=True,
        ):
            clear_active_paper()
            st.rerun()

    st.divider()

    st.caption(
        "Runtime-only demo: uploaded "
        "documents and conversations may "
        "disappear when the application restarts."
    )


# =========================================================
# MAIN CHAT
# =========================================================

st.title(
    "Research Paper RAG Analyzer"
)

st.caption(
    "Adaptive query routing, hybrid dense + BM25 retrieval, "
    "RRF fusion, parent-context expansion and grounded citations."
)

if st.session_state.notice:
    st.success(
        st.session_state.notice
    )

    st.session_state.notice = None


if st.session_state.graph is None:
    st.info(
        "Upload and process a research-paper "
        "PDF to begin the conversation."
    )

else:
    st.caption(
        "Active paper: "
        f"{st.session_state.paper_name}"
    )

    for message in (
        st.session_state.messages
    ):
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

            if (
                message["role"]
                == "assistant"
            ):
                render_route(
                    message.get("route")
                )

                render_sources(
                    sources=message.get(
                        "sources",
                        [],
                    ),
                    retrieval_query=(
                        message.get(
                            "retrieval_query"
                        )
                    ),
                )

    question = st.chat_input(
        "Ask a question about the paper"
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message(
            "user"
        ):
            st.markdown(
                question
            )

        try:
            with st.chat_message(
                "assistant"
            ):
                with st.spinner(
                    "Routing the question..."
                ):
                    config = {
                        "configurable": {
                            "thread_id": (
                                st.session_state
                                .thread_id
                            )
                        }
                    }

                    result = (
                        st.session_state
                        .graph
                        .invoke(
                            {
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": question,
                                    }
                                ]
                            },
                            config,
                        )
                    )

                answer = (
                    result["messages"][
                        -1
                    ].content
                )

                route = result.get(
                    "route",
                    "retrieval",
                )

                sources = result.get(
                    "sources",
                    [],
                )

                retrieval_query = (
                    result.get(
                        "retrieval_query",
                        question,
                    )
                )

                st.markdown(
                    answer
                )

                render_route(
                    route
                )

                render_sources(
                    sources=sources,
                    retrieval_query=(
                        retrieval_query
                    ),
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "route": route,
                    "sources": sources,
                    "retrieval_query": (
                        retrieval_query
                    ),
                }
            )

        except Exception as error:
            st.error(
                "Could not generate an answer: "
                f"{error}"
            )
