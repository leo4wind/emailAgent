from typing import TypedDict, Literal, Optional, List, Dict
from langchain_core.messages import HumanMessage

# 定义状态结构（为了独立性，这里重复定义）
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

def read_email(state: EmailAgentState) -> dict:
    """提取并解析邮件内容"""
    print("--- 正在读取邮件 ---")
    return {
        "messages": [HumanMessage(content=f"Processing email: {state['email_content']}")]
    }