from typing import Literal, Optional, List, Dict
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from states import EmailClassification, EmailAgentState

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