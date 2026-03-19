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

def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """搜索知识库"""
    print("--- 正在搜索文档 ---")
    classification = state.get('classification', {})
    # 模拟搜索逻辑
    search_results = [
        "Reset password via Settings > Security > Change Password",
        "Password must be at least 12 characters"
    ]
    return Command(update={"search_results": search_results}, goto="draft_response")