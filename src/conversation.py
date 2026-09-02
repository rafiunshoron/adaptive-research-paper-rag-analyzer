import argparse
import re
from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import (
    END,
    MessagesState,
    START,
    StateGraph,
)

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
from src.rag_chain import create_answer_chain


PAPER_OVERVIEW_MAX_CHARACTERS = 5000

FALLBACK_ANSWER = (
    "I could not find enough information in the uploaded paper "
    "to answer this question."
)


class ConversationState(MessagesState):
    """State shared by the adaptive conversational RAG graph."""

    route: str
    retrieval_query: str
    retrieved_children: list[Document]
    parent_documents: list[Document]
    context: str
    sources: list[dict]


ROUTER_SYSTEM_PROMPT = """
You are a routing component for a research-paper chat application.

Classify the user's latest message into exactly one route:

- retrieval: The message asks about the uploaded paper, including
  its title, authors, concepts, methods, datasets, experiments,
  results, tables, conclusions, or a follow-up to a previous
  paper-related answer.
- direct: The message is a greeting, thanks, goodbye, casual
  conversation, a question about the assistant's capabilities, or
  a clearly unrelated request that should not search the paper.

Use the conversation history to identify short follow-up questions.
If a message is plausibly related to the paper, choose retrieval.

Return only one lowercase word: direct or retrieval.
""".strip()


REWRITE_SYSTEM_PROMPT = """
Your only task is to rewrite the user's latest question into a
clear, standalone research-paper retrieval query.

You are given a short overview of the uploaded paper and any
available conversation history.

Important rules:

1. If the question is vague, use the paper overview to identify
   the exact model, component, method, dataset, or concept.
2. Resolve unclear references such as "it", "that", "this",
   "the first one", "the second one", "this approach", or
   "that model".
3. Pay particular attention to the latest assistant answer.
4. Preserve the user's original intent.
5. Correct minor spelling or grammatical errors when necessary.
6. Do not answer the question.
7. Do not explain the rewriting.
8. Return only one standalone retrieval question.
9. If the question is already clear and standalone, return it
   unchanged.

Example:

Paper overview:
The model selectively fine-tunes the final six transformer
layers of BERT.

Latest question:
How many layers are finetuned?

Standalone retrieval query:
How many BERT transformer layers are fine-tuned in the model?
""".strip()


DIRECT_SYSTEM_PROMPT = """
You are the conversational interface for a research-paper analysis
application.

Follow these rules:

1. Respond naturally and briefly to greetings, thanks, goodbyes,
   and casual conversation.
2. If asked what you can do, explain that you can analyze the
   currently uploaded research paper.
3. If the request is unrelated to the paper, politely explain that
   this application is focused on the uploaded research paper and
   invite the user to ask a paper-related question.
4. Do not claim that you searched the paper.
5. Do not provide citations or invent paper information.
6. Keep the response to one or two short sentences.
""".strip()


DIRECT_RULE_PATTERNS = (
    r"^(hi|hello|hey|good morning|good afternoon|good evening)$",
    r"^(hi|hello|hey)( there)? (how are you|how is it going|how's it going)$",
    r"^(how are you|how is it going|how's it going)$",
    r"^(thanks|thank you|thank you very much|great|okay|ok|got it)$",
    r"^(bye|goodbye|see you|see you later)$",
    r"^(what can you do|who are you|help)$",
)


def build_paper_overview(parent_store: dict) -> str:
    """Build a compact overview from the earliest parent sections."""

    overview_parts = []
    current_length = 0

    for parent_record in parent_store.values():
        page_content = parent_record.get("page_content", "").strip()

        if not page_content:
            continue

        remaining_characters = (
            PAPER_OVERVIEW_MAX_CHARACTERS - current_length
        )

        if remaining_characters <= 0:
            break

        selected_content = page_content[:remaining_characters]
        overview_parts.append(selected_content)
        current_length += len(selected_content)

    return "\n\n".join(overview_parts)


def normalize_rule_text(text: str) -> str:
    """Normalize a message for deterministic routing rules."""

    normalized = re.sub(r"[^\w\s']", " ", text.lower())
    return " ".join(normalized.split())


def rule_route(question: str) -> Literal["direct"] | None:
    """Route obvious conversational messages without an LLM call."""

    normalized_question = normalize_rule_text(question)

    for pattern in DIRECT_RULE_PATTERNS:
        if re.fullmatch(pattern, normalized_question):
            return "direct"

    return None


def create_router_chain():
    """Create the LLM router for messages not handled by rules."""

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY was not found. Add it to the environment."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    f"{ROUTER_SYSTEM_PROMPT}\n\n"
                    "Uploaded paper overview:\n"
                    "{paper_overview}"
                ),
            ),
            MessagesPlaceholder("conversation_history"),
            ("human", "Latest message:\n\n{question}"),
        ]
    )

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=10,
        timeout=60,
        max_retries=2,
    )

    return prompt | llm | StrOutputParser()


def create_question_rewrite_chain():
    """Create the document-aware query rewriting chain."""

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY was not found. Add it to the environment."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    f"{REWRITE_SYSTEM_PROMPT}\n\n"
                    "Uploaded paper overview:\n"
                    "{paper_overview}"
                ),
            ),
            MessagesPlaceholder("conversation_history"),
            (
                "human",
                (
                    "Latest question:\n\n{question}\n\n"
                    "Return only the standalone retrieval question."
                ),
            ),
        ]
    )

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=140,
        timeout=60,
        max_retries=2,
    )

    return prompt | llm | StrOutputParser()


def create_direct_response_chain():
    """Create the response chain used when retrieval is unnecessary."""

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY was not found. Add it to the environment."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DIRECT_SYSTEM_PROMPT),
            MessagesPlaceholder("conversation_history"),
            ("human", "{question}"),
        ]
    )

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=120,
        timeout=60,
        max_retries=2,
    )

    return prompt | llm | StrOutputParser()


def create_conversation_graph(
    checkpointer: InMemorySaver,
    hybrid_retriever,
    parent_store: dict,
):
    """Create the adaptive conversational research-paper RAG graph."""

    router_chain = create_router_chain()
    rewrite_chain = create_question_rewrite_chain()
    direct_response_chain = create_direct_response_chain()
    answer_chain = create_answer_chain()
    paper_overview = build_paper_overview(parent_store)

    def latest_question(state: ConversationState) -> str:
        return str(state["messages"][-1].content).strip()

    def previous_messages(state: ConversationState) -> list:
        return state["messages"][:-1][-6:]

    def route_question_node(state: ConversationState) -> dict:
        """Choose direct conversation or paper retrieval."""

        question = latest_question(state)
        route = rule_route(question)

        if route is None:
            try:
                route = router_chain.invoke(
                    {
                        "paper_overview": paper_overview,
                        "conversation_history": previous_messages(state),
                        "question": question,
                    }
                ).strip().lower()
            except Exception:
                # Retrieval is the safer fallback for a paper assistant.
                route = "retrieval"

        if route not in {"direct", "retrieval"}:
            route = "retrieval"

        return {
            "route": route,
            "retrieval_query": "",
            "retrieved_children": [],
            "parent_documents": [],
            "context": "",
            "sources": [],
        }

    def choose_route(
        state: ConversationState,
    ) -> Literal["direct", "retrieval"]:
        """Return the conditional edge selected by the router."""

        if state.get("route") == "direct":
            return "direct"

        return "retrieval"

    def direct_response_node(state: ConversationState) -> dict:
        """Respond without searching the uploaded paper."""

        answer = direct_response_chain.invoke(
            {
                "conversation_history": previous_messages(state),
                "question": latest_question(state),
            }
        ).strip()

        if not answer:
            answer = (
                "Hello! Ask me anything about the uploaded "
                "research paper."
            )

        return {
            "messages": [AIMessage(content=answer)],
            "route": "direct",
            "retrieval_query": "",
            "sources": [],
        }

    def rewrite_question_node(state: ConversationState) -> dict:
        """Convert the latest paper question into a standalone query."""

        current_question = latest_question(state)

        standalone_question = rewrite_chain.invoke(
            {
                "paper_overview": paper_overview,
                "conversation_history": previous_messages(state),
                "question": current_question,
            }
        ).strip().strip('"')

        if not standalone_question:
            standalone_question = current_question

        return {"retrieval_query": standalone_question}

    def retrieve_documents_node(state: ConversationState) -> dict:
        """Run hybrid retrieval and keep the strongest child chunks."""

        retrieval_query = state.get("retrieval_query") or latest_question(
            state
        )
        fused_candidates = hybrid_retriever.invoke(retrieval_query)

        return {
            "retrieved_children": fused_candidates[:FINAL_CHILD_K],
        }

    def build_context_node(state: ConversationState) -> dict:
        """Expand retrieved children into larger parent contexts."""

        parent_documents = expand_to_parents(
            state.get("retrieved_children", []),
            parent_store,
        )
        context = format_parent_context(parent_documents)

        sources = []

        for document in parent_documents:
            metadata = document.metadata
            sources.append(
                {
                    "source": metadata.get("source"),
                    "section_title": metadata.get("section_title"),
                    "page_start": metadata.get("page_start"),
                    "page_end": metadata.get("page_end"),
                }
            )

        return {
            "parent_documents": parent_documents,
            "context": context,
            "sources": sources,
        }

    def generate_answer_node(state: ConversationState) -> dict:
        """Generate a grounded answer from the selected context."""

        context = state.get("context", "")
        retrieval_query = state.get("retrieval_query") or latest_question(
            state
        )

        if context:
            answer = answer_chain.invoke(
                {
                    "question": retrieval_query,
                    "context": context,
                }
            )
        else:
            answer = FALLBACK_ANSWER

        return {
            "messages": [AIMessage(content=answer)],
            "route": "retrieval",
            # Keep the final checkpoint light.
            "retrieved_children": [],
            "parent_documents": [],
            "context": "",
        }

    builder = StateGraph(ConversationState)

    builder.add_node("route_question", route_question_node)
    builder.add_node("direct_response", direct_response_node)
    builder.add_node("rewrite_question", rewrite_question_node)
    builder.add_node("retrieve_documents", retrieve_documents_node)
    builder.add_node("build_context", build_context_node)
    builder.add_node("generate_answer", generate_answer_node)

    builder.add_edge(START, "route_question")

    builder.add_conditional_edges(
        "route_question",
        choose_route,
        {
            "direct": "direct_response",
            "retrieval": "rewrite_question",
        },
    )

    builder.add_edge("direct_response", END)
    builder.add_edge("rewrite_question", "retrieve_documents")
    builder.add_edge("retrieve_documents", "build_context")
    builder.add_edge("build_context", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile(checkpointer=checkpointer)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run adaptive conversational research-paper RAG with "
            "runtime memory."
        )
    )
    parser.add_argument(
        "--session-id",
        default="default-session",
        help="Runtime thread ID used to separate conversation histories.",
    )
    args = parser.parse_args()

    embeddings = create_embeddings()
    vector_store = create_vector_store(embeddings)
    hybrid_retriever = create_hybrid_retriever(vector_store)
    parent_store = load_parent_store()
    checkpointer = InMemorySaver()

    graph = create_conversation_graph(
        checkpointer=checkpointer,
        hybrid_retriever=hybrid_retriever,
        parent_store=parent_store,
    )

    config = {
        "configurable": {
            "thread_id": args.session_id,
        }
    }

    print(f"\nRuntime session: {args.session_id}")
    print("Conversation history will be cleared when this program stops.")
    print("Type 'exit' to close the conversation.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nConversation closed.")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Conversation closed.")
            break

        result = graph.invoke(
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

        route = result.get("route", "retrieval")
        answer = result["messages"][-1].content

        print(f"\nRoute: {route}")

        if route == "retrieval":
            retrieval_query = result.get("retrieval_query", question)
            print(f"Retrieval query: {retrieval_query}")

        print(f"\nAssistant: {answer}")

        sources = result.get("sources", [])

        if sources:
            print("\nSources:")

            for source_number, source in enumerate(sources, start=1):
                print(
                    f"[S{source_number}] "
                    f"{source.get('source')} | "
                    f"{source.get('section_title')} | "
                    f"pages {source.get('page_start', '?')}-"
                    f"{source.get('page_end', '?')}"
                )

        print()


if __name__ == "__main__":
    main()
