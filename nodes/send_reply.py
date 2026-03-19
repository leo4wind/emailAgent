from typing import TypedDict, Literal, Optional, List, Dict
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

def send_reply(state: EmailAgentState) -> dict:
    """发送邮件"""
    print(f"--- 邮件已发送 ---")
    print(f"内容预览: {state['draft_response'][:100]}...")
    return {}