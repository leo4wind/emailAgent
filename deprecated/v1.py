#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
极简 LangGraph 客服邮件处理演示（独立单文件版本）
最后更新：2025

=================== 使用方法 ===================

1. 创建独立虚拟环境（强烈推荐）
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

2. 安装依赖（只需执行一次）
    pip install -U langgraph langchain langchain-openai python-dotenv pydantic

3. 创建 .env 文件（同目录），内容示例：
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

4. 运行
   python main.py

可选：使用本地模型（例如 Ollama）
把 llm 那几行替换成：
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5:7b", temperature=0.6)

================================================
"""

import os
from typing import Literal, TypedDict, Annotated
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# 加载 .env 文件（如果存在）
load_dotenv()

# ==============================================
#               状态定义
# ==============================================
class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex", "other"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str


class AgentState(TypedDict):
    email_content: str
    sender: str
    classification: EmailClassification | None
    search_results: list[str] | None
    draft_response: str | None
    messages: Annotated[list[HumanMessage], "add_messages"]
    current_step: str | None


# ==============================================
#               LLM 配置
# ==============================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    openai_api_key = os.getenv("OPENAI_API_KEY"),          # 你的 key
    openai_api_base = os.getenv("OPENAI_API_BASE"),        # 中转地址，例如 https://api.abc.work/v1
    # timeout=60,
    # 可选：使用代理
    # http_client=...,
)


# ==============================================
#               节点函数
# ==============================================

def classify_email(state: AgentState) -> Command:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个邮件分类助手。请严格按照以下 JSON 格式输出，不要多说任何其他文字：

{{
  "intent": "question" | "bug" | "billing" | "feature" | "complex" | "other",
  "urgency": "low" | "medium" | "high" | "critical",
  "topic": "一句话主题概括",
  "summary": "2-3句话总结客户问题"
}}

紧急程度参考：
- critical：金钱损失、服务中断、法律、安全
- high：严重影响使用、情绪激动、重复投诉
- medium：普通技术问题、功能咨询
- low：建议、感谢、非紧急
"""),
        ("human", "发件人：{sender}\n内容：\n{email}\n请分类。")
    ])

    chain = prompt | llm.with_structured_output(EmailClassification)

    cls = chain.invoke({
        "sender": state["sender"],
        "email": state["email_content"]
    })

    # 路由决策
    if cls["urgency"] in ("high", "critical") or cls["intent"] == "complex":
        next_node = "human_review_pause"
    elif cls["intent"] in ("question", "feature"):
        next_node = "fake_search"
    elif cls["intent"] == "bug":
        next_node = "fake_bug_track"
    else:
        next_node = "draft"

    return Command(
        update={"classification": cls, "current_step": "classified"},
        goto=next_node
    )


def fake_search(state: AgentState) -> dict:
    topic = (state["classification"] or {}).get("topic", "").lower()
    fake_results = {
        "password reset": ["登录页面 → 忘记密码 → 邮箱验证", "密码至少12位，建议字母+数字+符号"],
        "double charge": ["请提供订单号/交易时间，我们将立即核查", "重复扣款通常24小时内退回"],
        "dark mode": ["深色模式已排入2025 Q3 规划", "感谢宝贵建议！"],
    }.get(topic, ["暂无匹配知识库内容"])

    return {
        "search_results": fake_results,
        "current_step": "searched"
    }


def fake_bug_track(state: AgentState) -> dict:
    ticket = f"BUG-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return {
        "search_results": [f"已创建问题单：{ticket}"],
        "current_step": "bug_tracked"
    }


def draft(state: AgentState) -> Command:
    cls = state.get("classification", {})
    docs = state.get("search_results", [])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业、亲切、有温度的客服。
请根据以下信息写一封回复邮件：

分类：{cls}
知识：{docs}

要求：
- 语气礼貌、专业、温暖
- 结构：问候 → 确认诉求 → 解决方案 → 下一步 → 结束语
- 中文回复
"""),
        ("human", "原邮件：\n{email}")
    ])

    response = (prompt | llm).invoke({
        "cls": cls,
        "docs": "\n".join(docs) if docs else "无相关文档",
        "email": state["email_content"]
    })

    needs_human = cls.get("urgency") in ("high", "critical") or cls.get("intent") == "complex"

    return Command(
        update={"draft_response": response.content, "current_step": "drafted"},
        goto="human_review_pause" if needs_human else "send"
    )


def human_review_pause(state: AgentState) -> dict:
    """只負責觸發中斷，不做決策"""
    # 如果還沒有草稿，先給一個占位（可選）
    if not state.get("draft_response"):
        state["draft_response"] = "我們正在緊急處理您的重複扣款問題，請稍候..."

    payload = {
        "type": "need_human_review",
        "original_email": state["email_content"],
        "classification": state.get("classification"),
        "draft": state.get("draft_response"),
        "message": "請審核/修改回覆草稿，確認後點繼續，或直接拒絕並說明原因"
    }

    interrupt(payload)
    
    # interrupt 會自動暫停，這裡不需要 return Command
    return {"current_step": "waiting_human_review"}  # 可選：更新狀態


def human_review_process(state: AgentState) -> Command:
    """resume 後執行的節點，處理人工輸入"""
    # 從輸入中取得人工決策（invoke 時傳入的資料）
    # 注意：這裡假設 decision 會被傳到 state 或直接作為輸入
    # 但更穩定的做法是讓 resume 時的 input 直接帶決策資料

    approved = state.get("human_approved", False)  # 或從其他地方取
    final_text = state.get("human_final_text") or state.get("draft_response", "")

    if approved:
        return Command(
            update={
                "draft_response": final_text,
                "current_step": "approved"
            },
            goto="send"
        )
    else:
        return Command(
            update={"current_step": "rejected"},
            goto=END
        )


def send(state: AgentState) -> dict:
    print("\n" + "═" * 70)
    print("              模拟发送邮件成功")
    print("═" * 70)
    print(f"收件人: {state['sender']}")
    print(f"回复内容:\n\n{state.get('draft_response', '(无内容)')} ")
    print("═" * 70 + "\n")
    return {"current_step": "sent"}


# ==============================================
#               构建图
# ==============================================
workflow = StateGraph(AgentState)

workflow.add_node("classify", classify_email)
workflow.add_node("fake_search", fake_search)
workflow.add_node("fake_bug_track", fake_bug_track)
workflow.add_node("draft", draft)
workflow.add_node("human_review_pause", human_review_pause)
workflow.add_node("human_review_process", human_review_process)
workflow.add_node("send", send)

workflow.add_edge(START, "classify")
workflow.add_edge("send", END)

# 使用内存检查点（支持中断后继续）
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# ==============================================
#               主程序
# ==============================================
def main():
    print("===  LangGraph 客服邮件处理单文件演示  ===\n")

    email_example = {
        "email_content": "我被扣了两次钱！非常生气！请立刻解决！",
        "sender": "furious.customer.2026@gmail.com",
        "messages": [],
        "current_step": None,
    }

    thread = {"configurable": {"thread_id": "demo-single-file-2026"}}

    print("正在处理...\n")
    result = app.invoke(email_example, thread)

    if "__interrupt__" in result:
        print("[已暂停 - 等待人工审核]")
        print("中断信息：")
        print(result["__interrupt__"])

        # 模拟人工通过并修改
        human_decision_input = {
            "human_approved": True,
            "human_final_text": "非常抱歉給您帶來不好的體驗！\n我們已立即為您處理重複扣款，款項將於24小時內退回原支付帳戶。\n感謝您的反饋與耐心。"
        }

        print("\n模拟人工审核通过，继续执行...\n")
        final = app.invoke(human_decision_input, thread)
        print("最终状态：", final.get("current_step", "unknown"))

    else:
        print("流程直接完成，无需人工介入")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("错误：请先在当前目录创建 .env 文件，并填入 OPENAI_API_KEY")
        print("示例：")
        print('OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        exit(1)

    main()