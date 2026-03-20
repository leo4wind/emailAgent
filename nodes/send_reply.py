from typing import Literal, Optional, List, Dict
from langchain_core.messages import HumanMessage
from states import EmailClassification, EmailAgentState

def send_reply(state: EmailAgentState) -> dict:
    """发送邮件"""
    print(f"--- 邮件已发送 ---")
    print(f"内容预览: {state['draft_response'][:100]}...")
    return {}