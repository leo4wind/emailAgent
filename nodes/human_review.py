from typing import TypedDict, Literal, Optional, List, Dict
from langgraph.types import interrupt, Command
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

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", "__end__"]]:
    """人工审核节点 (会暂停执行)"""
    print("--- 等待人工审核 ---")
    
    # interrupt 会保存当前状态并暂停
    human_decision = interrupt({
        "email_id": state.get('email_id'),
        "draft_response": state.get('draft_response'),
        "action": "Please review and approve/edit this response"
    })

    if human_decision.get("approved"):
        return Command(
            update={"draft_response": human_decision.get("edited_response", state.get('draft_response'))},
            goto="send_reply"
        )
    else:
        print("人工拒绝了该回复。")
        return Command(update={}, goto="__end__")