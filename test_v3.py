from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

from main import app


def assert_search_results_are_ranked_documents(search_results):
    assert isinstance(search_results, list), "search_results should be a list"
    assert search_results, "search_results should not be empty"

    first_result = search_results[0]
    assert not first_result.startswith(
        "Search temporarily unavailable:"
    ), f"Live search failed: {first_result}"
    assert (
        first_result != "No matching knowledge base content was found."
    ), "Expected at least one knowledge base hit."

    for index, item in enumerate(search_results, start=1):
        assert item.startswith("["), f"Result #{index} should start with [category]: {item}"
        assert " | chunk=" in item, f"Result #{index} missing chunk marker: {item}"
        assert " | score=" in item, f"Result #{index} missing score marker: {item}"
        assert " | content=" in item, f"Result #{index} missing content marker: {item}"


def assert_any_result_contains(search_results, keywords, message):
    lowered_results = [item.lower() for item in search_results]
    assert any(
        keyword.lower() in item
        for item in lowered_results
        for keyword in keywords
    ), message


def assert_bug_ticket_result(search_results):
    assert isinstance(search_results, list), "search_results should be a list"
    assert search_results, "bug tracking should produce a ticket result"
    assert any(
        "bug ticket" in item.lower() and "created" in item.lower()
        for item in search_results
    ), f"Expected bug ticket creation message, got: {search_results}"


def run_email_flow(thread_id, email_content, sender_email, email_id):
    config = {"configurable": {"thread_id": thread_id}}
    initial_input = {
        "email_content": email_content,
        "sender_email": sender_email,
        "email_id": email_id,
        "messages": [],
    }

    print(f"\n=== 运行端到端测试: {thread_id} ===")
    try:
        app.invoke(initial_input, config)
    except Exception as exc:
        raise AssertionError(
            f"{thread_id}: app.invoke failed before reaching the final state. "
            f"Check OPENAI_API_BASE / OPENAI_API_KEY connectivity. Root error: {exc}"
        ) from exc

    snapshot = app.get_state(config)
    interrupted = bool(snapshot.next)

    if snapshot.next:
        print(f"{thread_id}: 检测到中断，自动模拟人工批准")
        resume_payload = {"approved": True}
        snapshot_values = snapshot.values
        if not snapshot_values.get("draft_response"):
            resume_payload["edited_response"] = (
                f"Manual review approved for {email_id}. "
                "We have received your request and will follow up shortly."
            )
        try:
            app.invoke(Command(resume=resume_payload), config)
        except Exception as exc:
            raise AssertionError(
                f"{thread_id}: resume invoke failed. Root error: {exc}"
            ) from exc
        snapshot = app.get_state(config)

    values = snapshot.values
    print(f"{thread_id}: classification = {values.get('classification')}")
    print(f"{thread_id}: search_results count = {len(values.get('search_results') or [])}")
    print(f"{thread_id}: interrupted = {interrupted}")
    return values, interrupted


def test_graph_question_e2e():
    values, interrupted = run_email_flow(
        thread_id="test_v3_graph_question_e2e",
        email_content="How does the email agent read incoming email and route between nodes?",
        sender_email="integration@example.com",
        email_id="email_graph_question",
    )

    classification = values.get("classification") or {}
    assert classification.get("intent") == "question", classification
    assert interrupted is False, "question flow should not require human review"

    search_results = values.get("search_results")
    assert_search_results_are_ranked_documents(search_results)
    assert len(search_results) <= 5, "search_results should respect top_k=5"
    assert_any_result_contains(
        search_results,
        ["read", "email", "node", "graph", "workflow"],
        "Expected retrieved content to relate to the graph/email workflow query.",
    )


def test_draft_response_question_e2e():
    values, interrupted = run_email_flow(
        thread_id="test_v3_draft_response_question_e2e",
        email_content="Where is the documentation for draft response generation in this email agent?",
        sender_email="integration@example.com",
        email_id="email_draft_response_question",
    )

    classification = values.get("classification") or {}
    assert classification.get("intent") == "question", classification
    assert interrupted is False, "question flow should not require human review"

    search_results = values.get("search_results")
    assert_search_results_are_ranked_documents(search_results)
    assert_any_result_contains(
        search_results,
        ["draft", "response", "reply", "context"],
        "Expected retrieved content to relate to draft response documentation.",
    )


def test_feature_request_e2e():
    values, interrupted = run_email_flow(
        thread_id="test_v3_feature_request_e2e",
        email_content="Can you add a dark mode feature to the email agent UI?",
        sender_email="feature@example.com",
        email_id="email_feature_request",
    )

    classification = values.get("classification") or {}
    assert classification.get("intent") == "feature", classification
    assert interrupted is False, "feature flow should not require human review"

    search_results = values.get("search_results")
    assert_search_results_are_ranked_documents(search_results)
    assert_any_result_contains(
        search_results,
        ["feature", "email", "agent", "workflow"],
        "Expected retrieved content to contain feature-related workflow context.",
    )


def test_bug_tracking_high_urgency_e2e():
    values, interrupted = run_email_flow(
        thread_id="test_v3_bug_tracking_high_urgency_e2e",
        email_content="The app crashes when I try to save my settings, and I can reproduce it every time.",
        sender_email="bug@example.com",
        email_id="email_bug_tracking_high",
    )

    classification = values.get("classification") or {}
    assert classification.get("intent") == "bug", classification
    assert classification.get("urgency") in ["high", "critical"], classification
    assert interrupted is True, "high-urgency bug flow should enter human review"

    search_results = values.get("search_results")
    assert_bug_ticket_result(search_results)
    assert values.get("draft_response"), "bug flow should still generate a draft response"


def test_bug_tracking_lower_urgency_e2e():
    values, interrupted = run_email_flow(
        thread_id="test_v3_bug_tracking_lower_urgency_e2e",
        email_content="There is a bug where the export button sometimes looks misaligned on the settings page.",
        sender_email="bug-low@example.com",
        email_id="email_bug_tracking_low",
    )

    classification = values.get("classification") or {}
    assert classification.get("intent") == "bug", classification
    assert classification.get("urgency") in ["low", "medium"], classification
    assert interrupted is False, "lower-urgency bug flow should not require human review"

    search_results = values.get("search_results")
    assert_bug_ticket_result(search_results)
    assert values.get("draft_response"), "bug flow should still generate a draft response"


def test_human_review_billing_e2e():
    values, interrupted = run_email_flow(
        thread_id="test_v3_human_review_billing_e2e",
        email_content="I was charged twice for my subscription and need an urgent refund today.",
        sender_email="billing@example.com",
        email_id="email_billing_review",
    )

    classification = values.get("classification") or {}
    assert classification.get("intent") == "billing", classification
    assert interrupted is True, "billing flow should enter human review"
    assert values.get("draft_response"), "billing flow should have a draft response after approval"


if __name__ == "__main__":
    test_graph_question_e2e()
    test_draft_response_question_e2e()
    test_feature_request_e2e()
    test_bug_tracking_high_urgency_e2e()
    test_bug_tracking_lower_urgency_e2e()
    test_human_review_billing_e2e()
    print("\n所有 test_v3 端到端测试完成")
