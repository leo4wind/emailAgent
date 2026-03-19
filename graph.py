from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import MemorySaver

from states import EmailAgentState
from nodes.read_email import read_email
from nodes.classify_intent import classify_intent
from nodes.search_documentation import search_documentation
from nodes.bug_tracking import bug_tracking
from nodes.draft_response import draft_response
from nodes.human_review import human_review
from nodes.send_reply import send_reply

# 构建图
def build_graph():
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
    return app