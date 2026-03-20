from typing import Literal, Optional, List, Dict
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from states import EmailClassification, EmailAgentState

def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """记录 Bug 票据"""
    print("--- 正在记录 Bug ---")
    ticket_id = "BUG-12345"
    return Command(
        update={"search_results": [f"Bug ticket {ticket_id} created"]},
        goto="draft_response"
    )