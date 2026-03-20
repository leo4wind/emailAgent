from typing import TypedDict, Literal, Optional, List, Dict
from langchain_core.messages import HumanMessage
from states import EmailClassification, EmailAgentState

def read_email(state: EmailAgentState) -> dict:
    """提取并解析邮件内容"""
    print("--- 正在读取邮件 ---")
    return {
        "messages": [HumanMessage(content=f"Processing email: {state['email_content']}")]
    }