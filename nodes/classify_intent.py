from typing import TypedDict, Literal, Optional, List, Dict
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from .config import llm

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

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """对邮件意图进行分类并路由"""
    print("--- 正在分类意图 ---")
    structured_llm = llm.with_structured_output(EmailClassification)
    
    classification_prompt = f"""
    Analyze this customer email and classify it:
    Email: {state['email_content']}
    From: {state['sender_email']}
    Provide classification including intent, urgency, topic, and summary.
    """
    
    classification = structured_llm.invoke(classification_prompt)
    
    # 路由逻辑
    if classification['intent'] == 'billing' or classification['urgency'] == 'critical':
        goto = "human_review"
    elif classification['intent'] in ['question', 'feature']:
        goto = "search_documentation"
    elif classification['intent'] == 'bug':
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    return Command(
        update={"classification": classification},
        goto=goto
    )