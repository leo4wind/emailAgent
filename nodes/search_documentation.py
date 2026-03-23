from typing import Literal
from langgraph.types import Command
from knowledge_base.pgvector_rag import KnowledgeBaseError, hybrid_search
from states import EmailAgentState


def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Search knowledge base for relevant information"""

    classification = state.get("classification", {})
    query_parts = [
        classification.get("intent", ""),
        classification.get("topic", ""),
        classification.get("summary", ""),
        state.get("email_content", ""),
    ]
    query = " ".join(part.strip() for part in query_parts if part and part.strip())
    try:
        documents = hybrid_search(
            query=query,
            top_k=5,
        )
        search_results = [
            (
                f"[{doc.category}] {doc.title} | chunk={doc.chunk_index} | "
                f"score={doc.final_score:.3f} | content={doc.content}"
            )
            for doc in documents
        ]
        if not search_results:
            search_results = ["No matching knowledge base content was found."]
    except KnowledgeBaseError as e:
        search_results = [f"Search temporarily unavailable: {str(e)}"]
    except Exception as e:
        search_results = [f"Search temporarily unavailable: {str(e)}"]

    return Command(
        update={"search_results": search_results},
        goto="draft_response"
    )
