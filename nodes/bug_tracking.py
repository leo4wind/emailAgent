from typing import TypedDict, Literal, Optional, List, Dict
from langgraph.types import Command
from langchain_core.messages import HumanMessage

# 定义状态结构
class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    email_content: str
    sender_email: str
    email_id: str
    classification: Optional[EmailClassification]
    search_results: Optional[List[str]]
    customer_history: Optional[Dict]
    draft_response: Optional[str]
    messages: List[HumanMessage]

def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """记录 Bug 票据"""
    print("--- 正在记录 Bug ---")
    ticket_id = "BUG-12345"
    return Command(
        update={"search_results": [f"Bug ticket {ticket_id} created"]},
        goto="draft_response"
    )