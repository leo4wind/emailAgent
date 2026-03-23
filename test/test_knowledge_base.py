import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from knowledge_base.pgvector_rag import (
    build_knowledge_chunks,
    hybrid_search,
    preprocess_text,
    rerank_documents,
)
from scripts.import_knowledge_files import import_documents, load_documents


class WorkspaceTempDir:
    def __init__(self, name: str):
        self.path = Path("test") / ".tmp" / name

    def __enter__(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)


class TestKnowledgeBaseUtilities(unittest.TestCase):
    def test_preprocess_text_splits_long_content(self):
        text = (
            "First paragraph with enough content to remain meaningful.\n\n"
            + ("Second paragraph. " * 80)
        )

        chunks = preprocess_text(text, chunk_size=200, chunk_overlap=30)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_build_knowledge_chunks_preserves_metadata(self):
        chunks = build_knowledge_chunks(
            source_id="doc-1",
            title="Password Reset",
            category="account",
            text="Reset password from settings.\n\nUse a strong password.",
            metadata={"lang": "en", "product": "email-agent"},
            chunk_size=40,
            chunk_overlap=10,
        )

        self.assertTrue(chunks)
        self.assertEqual(chunks[0].source_id, "doc-1")
        self.assertEqual(chunks[0].title, "Password Reset")
        self.assertEqual(chunks[0].category, "account")
        self.assertEqual(chunks[0].metadata["product"], "email-agent")

    @patch("knowledge_base.pgvector_rag._fetch_text_candidates")
    @patch("knowledge_base.pgvector_rag._fetch_vector_candidates")
    @patch("knowledge_base.pgvector_rag.generate_embeddings")
    def test_hybrid_search_combines_vector_and_bm25(
        self,
        mock_generate_embeddings,
        mock_fetch_vector_candidates,
        mock_fetch_text_candidates,
    ):
        from knowledge_base.pgvector_rag import RetrievedChunk

        mock_generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_fetch_vector_candidates.return_value = [
            RetrievedChunk(
                id="1",
                source_id="doc-1",
                title="Reset Password",
                category="account",
                content="Reset your password in account settings.",
                metadata={},
                chunk_index=0,
                vector_distance=0.1,
            )
        ]
        mock_fetch_text_candidates.return_value = [
            RetrievedChunk(
                id="2",
                source_id="doc-2",
                title="Password Policy",
                category="account",
                content="Use uppercase lowercase numbers and symbols.",
                metadata={},
                chunk_index=0,
            )
        ]

        results = hybrid_search("how to reset password", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0].final_score, results[1].final_score)
        self.assertTrue(all(result.final_score >= 0 for result in results))

    @patch("knowledge_base.pgvector_rag.get_openai_client")
    def test_rerank_documents_falls_back_when_model_fails(self, mock_get_openai_client):
        from knowledge_base.pgvector_rag import RetrievedChunk

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = RuntimeError("rerank unavailable")
        mock_get_openai_client.return_value = mock_client

        documents = [
            RetrievedChunk(
                id="1",
                source_id="doc-1",
                title="Reset Password",
                category="account",
                content="Reset password instructions.",
                metadata={},
                chunk_index=0,
                final_score=0.7,
            ),
            RetrievedChunk(
                id="2",
                source_id="doc-2",
                title="Password Policy",
                category="account",
                content="Strong password requirements.",
                metadata={},
                chunk_index=1,
                final_score=0.5,
            ),
        ]

        reranked = rerank_documents("reset password", documents, top_k=2)

        self.assertEqual([doc.id for doc in reranked], ["1", "2"])


class TestKnowledgeImportScript(unittest.TestCase):
    def test_load_documents_reads_markdown_and_text(self):
        with WorkspaceTempDir("load_documents") as root:
            (root / "guide.md").write_text("# Guide\n\nReset password here.", encoding="utf-8")
            (root / "faq.txt").write_text("FAQ content", encoding="utf-8")

            documents = load_documents(root, category="support")

        self.assertEqual(len(documents), 2)
        self.assertEqual({doc["category"] for doc in documents}, {"support"})

    @patch("scripts.import_knowledge_files.batch_insert_knowledge")
    def test_import_documents_builds_chunks_and_inserts(self, mock_batch_insert_knowledge):
        mock_batch_insert_knowledge.return_value = 3

        with WorkspaceTempDir("import_documents") as root:
            (root / "guide.md").write_text("# Guide\n\nReset password here.", encoding="utf-8")

            inserted = import_documents(
                input_path=root,
                category="support",
                chunk_size=100,
                chunk_overlap=10,
            )

        self.assertEqual(inserted, 3)
        self.assertTrue(mock_batch_insert_knowledge.called)


if __name__ == "__main__":
    unittest.main()
