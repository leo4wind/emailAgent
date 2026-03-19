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

# 导入状态和应用
from states import EmailClassification, EmailAgentState
from main import app

def test_billing_critical():
    """测试紧急账单邮件分支：classify_intent -> human_review -> interrupt -> resume -> send_reply"""
    print("\n=== 测试：紧急账单邮件 ===")
    config = {"configurable": {"thread_id": "test_billing_critical"}}

    initial_input = {
        "email_content": "I was charged twice for my subscription! This is urgent!",
        "sender_email": "customer@example.com",
        "email_id": "email_billing",
        "messages": []
    }

    # 第一次运行
    result = app.invoke(initial_input, config)
    snapshot = app.get_state(config)

    if snapshot.next:
        print("检测到中断，模拟人工批准")
        human_feedback = Command(
            resume={
                "approved": True,
                "edited_response": "We sincerely apologize for the double charge. I've initiated an immediate refund."
            }
        )
        final_result = app.invoke(human_feedback, config)
        print("流程结束")
        assert "draft_response" in final_result
    else:
        print("未中断，直接结束")

def test_question_low():
    """测试问题邮件分支：classify_intent -> search_documentation -> draft_response -> send_reply"""
    print("\n=== 测试：问题邮件 (低紧急) ===")
    config = {"configurable": {"thread_id": "test_question_low"}}

    initial_input = {
        "email_content": "How do I reset my password?",
        "sender_email": "user@example.com",
        "email_id": "email_question",
        "messages": []
    }

    result = app.invoke(initial_input, config)
    snapshot = app.get_state(config)

    if snapshot.next:
        print("检测到中断，模拟人工批准")
        human_feedback = Command(resume={"approved": True})
        final_result = app.invoke(human_feedback, config)
    else:
        print("流程直接完成")

def test_bug_medium():
    """测试 Bug 邮件分支：classify_intent -> bug_tracking -> draft_response -> send_reply"""
    print("\n=== 测试：Bug 邮件 (中等紧急) ===")
    config = {"configurable": {"thread_id": "test_bug_medium"}}

    initial_input = {
        "email_content": "The app crashes when I try to save my settings.",
        "sender_email": "developer@example.com",
        "email_id": "email_bug",
        "messages": []
    }

    result = app.invoke(initial_input, config)
    snapshot = app.get_state(config)

    if snapshot.next:
        print("检测到中断，模拟人工批准")
        human_feedback = Command(resume={"approved": True})
        final_result = app.invoke(human_feedback, config)
    else:
        print("流程直接完成")

def test_complex_high():
    """测试复杂邮件分支：classify_intent -> draft_response -> human_review -> interrupt -> resume -> send_reply"""
    print("\n=== 测试：复杂邮件 (高紧急) ===")
    config = {"configurable": {"thread_id": "test_complex_high"}}

    initial_input = {
        "email_content": "I have a complex issue with multiple features not working as expected. This is high priority.",
        "sender_email": "vip@example.com",
        "email_id": "email_complex",
        "messages": []
    }

    result = app.invoke(initial_input, config)
    snapshot = app.get_state(config)

    if snapshot.next:
        print("检测到中断，模拟人工批准")
        human_feedback = Command(resume={"approved": True})
        final_result = app.invoke(human_feedback, config)
    else:
        print("流程直接完成")

def test_feature_low():
    """测试功能请求邮件分支：classify_intent -> search_documentation -> draft_response -> send_reply"""
    print("\n=== 测试：功能请求邮件 (低紧急) ===")
    config = {"configurable": {"thread_id": "test_feature_low"}}

    initial_input = {
        "email_content": "Can you add a dark mode feature?",
        "sender_email": "user2@example.com",
        "email_id": "email_feature",
        "messages": []
    }

    result = app.invoke(initial_input, config)
    snapshot = app.get_state(config)

    if snapshot.next:
        print("检测到中断，模拟人工批准")
        human_feedback = Command(resume={"approved": True})
        final_result = app.invoke(human_feedback, config)
    else:
        print("流程直接完成")

if __name__ == "__main__":
    test_billing_critical()
    test_question_low()
    test_bug_medium()
    test_complex_high()
    test_feature_low()
    print("\n所有测试完成")