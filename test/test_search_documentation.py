import unittest
from types import SimpleNamespace
from unittest.mock import patch

from knowledge_base.pgvector_rag import KnowledgeBaseError
from nodes.search_documentation import search_documentation


class TestSearchDocumentation(unittest.TestCase):
    def build_state(self, **overrides):
        state = {
            "email_content": "How do I reset my password?",
            "sender_email": "user@example.com",
            "email_id": "email-question-1",
            "classification": {
                "intent": "question",
                "urgency": "low",
                "topic": "password",
                "summary": "User wants to reset account password",
            },
            "messages": [],
        }
        state.update(overrides)
        return state

    @patch("nodes.search_documentation.hybrid_search")
    def test_search_documentation_returns_ranked_results(self, mock_hybrid_search):
        mock_hybrid_search.return_value = [
            SimpleNamespace(
                category="account",
                title="Password Reset Guide",
                chunk_index=0,
                final_score=0.91234,
                content="Reset password from Settings > Security > Change Password.",
            ),
            SimpleNamespace(
                category="security",
                title="Password Policy",
                chunk_index=1,
                final_score=0.75432,
                content="Use at least 12 characters with numbers and symbols.",
            ),
        ]

        result = search_documentation(self.build_state())

        self.assertEqual(result.goto, "draft_response")
        self.assertEqual(len(result.update["search_results"]), 2)
        self.assertIn("[account] Password Reset Guide", result.update["search_results"][0])
        self.assertIn("score=0.912", result.update["search_results"][0])
        self.assertIn("content=Reset password", result.update["search_results"][0])
        mock_hybrid_search.assert_called_once_with(
            query=(
                "question password User wants to reset account password "
                "How do I reset my password?"
            ),
            top_k=5,
        )

    @patch("nodes.search_documentation.hybrid_search")
    def test_search_documentation_returns_no_match_message_when_empty(self, mock_hybrid_search):
        mock_hybrid_search.return_value = []

        result = search_documentation(self.build_state())

        self.assertEqual(result.goto, "draft_response")
        self.assertEqual(
            result.update["search_results"],
            ["No matching knowledge base content was found."],
        )

    @patch("nodes.search_documentation.hybrid_search")
    def test_search_documentation_handles_knowledge_base_error(self, mock_hybrid_search):
        mock_hybrid_search.side_effect = KnowledgeBaseError("database unavailable")

        result = search_documentation(self.build_state())

        self.assertEqual(result.goto, "draft_response")
        self.assertEqual(
            result.update["search_results"],
            ["Search temporarily unavailable: database unavailable"],
        )

    @patch("nodes.search_documentation.hybrid_search")
    def test_search_documentation_handles_unexpected_error(self, mock_hybrid_search):
        mock_hybrid_search.side_effect = RuntimeError("unexpected failure")

        result = search_documentation(self.build_state())

        self.assertEqual(result.goto, "draft_response")
        self.assertEqual(
            result.update["search_results"],
            ["Search temporarily unavailable: unexpected failure"],
        )

    @patch("nodes.search_documentation.hybrid_search")
    def test_search_documentation_builds_query_with_only_non_empty_parts(self, mock_hybrid_search):
        mock_hybrid_search.return_value = []
        state = self.build_state(
            email_content="Need help with invoices",
            classification={
                "intent": "billing",
                "urgency": "medium",
                "topic": "",
                "summary": "",
            },
        )

        search_documentation(state)

        mock_hybrid_search.assert_called_once_with(
            query="billing Need help with invoices",
            top_k=5,
        )


if __name__ == "__main__":
    unittest.main()
