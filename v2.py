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

# 2. 初始化模型
# 注意：文档中使用的是 gpt-5-nano 占位符，这里改为 gpt-4o 或 gpt-3.5-turbo
llm = ChatOpenAI(
    model="gpt-4o", 
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE")
)

# 3. 定义节点函数

def read_email(state: EmailAgentState) -> dict:
    """提取并解析邮件内容"""
    print("--- 正在读取邮件 ---")
    return {
        "messages": [HumanMessage(content=f"Processing email: {state['email_content']}")]
    }

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

def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """记录 Bug 票据"""
    print("--- 正在记录 Bug ---")
    ticket_id = "BUG-12345"
    return Command(
        update={"search_results": [f"Bug ticket {ticket_id} created"]},
        goto="draft_response"
    )

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

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """人工审核节点 (会暂停执行)"""
    print("--- 等待人工审核 ---")
    
    # interrupt 会保存当前状态并暂停
    human_decision = interrupt({
        "email_id": state.get('email_id'),
        "draft_response": state.get('draft_response'),
        "action": "Please review and approve/edit this response"
    })

    if human_decision.get("approved"):
        return Command(
            update={"draft_response": human_decision.get("edited_response", state.get('draft_response'))},
            goto="send_reply"
        )
    else:
        print("人工拒绝了该回复。")
        return Command(update={}, goto=END)

def send_reply(state: EmailAgentState) -> dict:
    """发送邮件"""
    print(f"--- 邮件已发送 ---")
    print(f"内容预览: {state['draft_response'][:100]}...")
    return {}

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

# 5. 测试运行逻辑
# 5. 测试运行逻辑
if __name__ == "__main__":
    # 配置线程 ID 
    config = {"configurable": {"thread_id": "test_conversation_005"}}
    
    # 模拟一封紧急账单邮件
    initial_input = {
        "email_content": "I was charged twice for my subscription! This is urgent!",
        "sender_email": "customer@example.com",
        "email_id": "email_123",
        "messages": []
    }

    print("\n--- 启动 Agent ---")
    
    # 第一次运行
    # 注意：如果路径直接跳到了 human_review，此时 state 可能还没有 draft_response 键
    result = app.invoke(initial_input, config)
    
    # 检查是否因为 interrupt 暂停
    # 在新版 langgraph 中，可以通过查看返回值或 snapshot 来判断
    snapshot = app.get_state(config)
    
    if snapshot.next: # 如果有下一步，说明被 interrupt 暂停了
        print("\n[系统提示] 检测到中断，进入人工介入环节。")
        
        # 安全地获取分类和草稿（可能为空）
        current_state = snapshot.values
        classification = current_state.get("classification", {})
        draft = current_state.get("draft_response", "尚未生成草稿（由分类直接升级为人工处理）")
        
        print(f"邮件意图: {classification.get('intent')}")
        print(f"紧急程度: {classification.get('urgency')}")
        print(f"当前草稿内容: {draft}")
        
        # 模拟人工操作
        print("\n--- 人工正在处理并提供回复 ---")
        human_feedback = Command(
            resume={
                "approved": True,
                "edited_response": "We sincerely apologize for the double charge. I've initiated an immediate refund."
            }
        )
        
        # 恢复运行
        final_result = app.invoke(human_feedback, config)
        print("\n--- 流程结束 ---")
        if "draft_response" in final_result:
            print(f"最终发送的邮件内容: {final_result['draft_response']}")