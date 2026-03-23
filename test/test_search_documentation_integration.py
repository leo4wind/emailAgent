import os
import re
import unittest
from pathlib import Path

from dotenv import load_dotenv
from psycopg2 import sql

from knowledge_base.pgvector_rag import get_postgresql_connection
from nodes.search_documentation import search_documentation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
RESULT_PATTERN = re.compile(
    r"^\[[^\]]+\] .+ \| chunk=\d+ \| score=\d+\.\d{3} \| content=.+",
    re.DOTALL,
)


class TestSearchDocumentationIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(dotenv_path=ENV_FILE)

        required_keys = [
            "OPENAI_API_KEY",
            "OPENAI_API_BASE",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
        ]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            raise unittest.SkipTest(
                "Live search_documentation integration tests require: "
                + ", ".join(missing_keys)
            )

        with get_postgresql_connection() as conn:
            with conn.cursor() as cursor:
                table_name = os.getenv("PGVECTOR_TABLE_NAME", "knowledge_base_documents")
                cursor.execute(
                    sql.SQL("SELECT COUNT(*) FROM {table_name}").format(
                        table_name=sql.Identifier(table_name)
                    )
                )
                row_count = cursor.fetchone()[0]

        if row_count <= 0:
            raise unittest.SkipTest(
                "Knowledge base table is empty; import documents before running this test."
            )

    def build_state(self, **overrides):
        state = {
            "email_content": "How does the email agent read incoming email and route between nodes?",
            "sender_email": "integration@example.com",
            "email_id": "integration-search-docs-1",
            "classification": {
                "intent": "question",
                "urgency": "low",
                "topic": "email agent workflow",
                "summary": "User wants documentation about reading email and routing in the graph",
            },
            "messages": [],
        }
        state.update(overrides)
        return state

    def assert_search_results_are_ranked_documents(self, search_results):
        self.assertIsInstance(search_results, list)
        self.assertTrue(search_results, "search_results should not be empty")

        first_result = search_results[0]
        self.assertFalse(
            first_result.startswith("Search temporarily unavailable:"),
            f"Live search failed: {first_result}",
        )
        self.assertNotEqual(
            first_result,
            "No matching knowledge base content was found.",
            "Expected at least one knowledge base hit for the integration query.",
        )

        for item in search_results:
            self.assertRegex(item, RESULT_PATTERN)

    def test_search_documentation_returns_structured_results_for_graph_question(self):
        result = search_documentation(self.build_state())

        self.assertEqual(result.goto, "draft_response")
        search_results = result.update["search_results"]
        self.assert_search_results_are_ranked_documents(search_results)
        self.assertLessEqual(len(search_results), 5)
        self.assertTrue(
            any(
                keyword in item.lower()
                for item in search_results
                for keyword in ["read", "email", "node", "graph", "workflow"]
            ),
            "Expected retrieved content to relate to the graph/email workflow query.",
        )

    def test_search_documentation_uses_classification_context_in_real_retrieval(self):
        state = self.build_state(
            email_content="Where is the documentation for draft response generation in this email agent?",
            email_id="integration-search-docs-2",
            classification={
                "intent": "question",
                "urgency": "low",
                "topic": "draft response",
                "summary": "User needs docs about reply drafting behavior",
            },
        )

        result = search_documentation(state)

        self.assertEqual(result.goto, "draft_response")
        search_results = result.update["search_results"]
        self.assert_search_results_are_ranked_documents(search_results)
        self.assertTrue(
            any(
                keyword in item.lower()
                for item in search_results
                for keyword in ["draft", "response", "reply", "context"]
            ),
            "Expected retrieved content to relate to draft response documentation.",
        )


if __name__ == "__main__":
    unittest.main()
