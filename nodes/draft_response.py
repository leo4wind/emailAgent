from typing import Literal, Optional, List, Dict
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from .config import llm
from states import EmailClassification, EmailAgentState

def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """生成草稿"""
    print("--- 正在生成回复草稿 ---")
    classification = state.get('classification', {})
    
    context = ""
    if state.get('search_results'):
        context = "\n".join([f"- {doc}" for doc in state['search_results']])

    draft_prompt = f"""
    Draft a response to: {state['email_content']}
    Context: {context}
    Intent: {classification.get('intent')}
    Urgency: {classification.get('urgency')}
    Guidelines: Professional and helpful.
    """
    
    response = llm.invoke(draft_prompt)
    
    # 判定是否需要人工审核
    needs_review = (
        classification.get('urgency') in ['high', 'critical'] or 
        classification.get('intent') == 'complex'
    )
    goto = "human_review" if needs_review else "send_reply"
    
    return Command(update={"draft_response": response.content}, goto=goto)