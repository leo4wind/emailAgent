import os
from typing import TypedDict, Literal, Optional, List, Dict
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 1. 定义状态结构
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

# 2. 初始化模型 - 移至节点文件中

# 3. 导入节点函数
from nodes.read_email import read_email
from nodes.classify_intent import classify_intent
from nodes.search_documentation import search_documentation
from nodes.bug_tracking import bug_tracking
from nodes.draft_response import draft_response
from nodes.human_review import human_review
from nodes.send_reply import send_reply

# 4. 构建图
workflow = StateGraph(EmailAgentState)

workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("search_documentation", search_documentation, retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0))
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

# 编译，使用内存保存状态（实现持久化以支持 interrupt）
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)