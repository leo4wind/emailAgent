import os
import unittest
from uuid import uuid4

from psycopg2 import sql
from psycopg2.extras import Json

from knowledge_base import get_postgresql_connection


class TestPostgreSQLPgVectorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
        os.environ.setdefault("POSTGRES_PORT", "5432")
        os.environ.setdefault("POSTGRES_DB", "postgres")
        os.environ.setdefault("POSTGRES_USER", "postgres")
        os.environ.setdefault("POSTGRES_PASSWORD", "1234")
        os.environ.setdefault("POSTGRES_SSLMODE", "prefer")
        os.environ.setdefault("POSTGRES_CONNECT_TIMEOUT", "10")

        cls.table_name = f"test_pgvector_{uuid4().hex[:12]}"
        cls.conn = get_postgresql_connection()
        cls.conn.autocommit = True

        with cls.conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE {table_name} (
                        id SERIAL PRIMARY KEY,
                        category VARCHAR(100) NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector(3) NOT NULL
                    )
                    """
                ).format(table_name=sql.Identifier(cls.table_name))
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "conn") and cls.conn:
            with cls.conn.cursor() as cursor:
                cursor.execute(
                    sql.SQL("DROP TABLE IF EXISTS {table_name}").format(
                        table_name=sql.Identifier(cls.table_name)
                    )
                )
            cls.conn.close()

    def setUp(self):
        with self.conn.cursor() as cursor:
            cursor.execute(
                sql.SQL("TRUNCATE TABLE {table_name}").format(
                    table_name=sql.Identifier(self.table_name)
                )
            )

    def test_database_connection_works(self):
        with self.conn.cursor() as cursor:
            cursor.execute(
                "SELECT current_database(), current_user, version()"
            )
            database_name, user_name, version = cursor.fetchone()

        self.assertTrue(database_name)
        self.assertTrue(user_name)
        self.assertIn("PostgreSQL", version)

    def test_pgvector_similarity_and_metadata_query(self):
        with self.conn.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    INSERT INTO {table_name} (category, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s::vector), (%s, %s, %s, %s::vector), (%s, %s, %s, %s::vector)
                    """
                ).format(table_name=sql.Identifier(self.table_name)),
                [
                    "account",
                    "Reset your password from account settings.",
                    Json({"lang": "en", "source": "guide"}),
                    "[1,0,0]",
                    "billing",
                    "Billing invoices are available on the billing page.",
                    Json({"lang": "en", "source": "faq"}),
                    "[0,1,0]",
                    "account",
                    "Enable MFA in security settings.",
                    Json({"lang": "zh", "source": "guide"}),
                    "[0.8,0.2,0]",
                ],
            )

            cursor.execute(
                sql.SQL(
                    """
                    SELECT category, content, metadata->>'source' AS source_name
                    FROM {table_name}
                    WHERE metadata @> %s::jsonb
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                    """
                ).format(table_name=sql.Identifier(self.table_name)),
                [Json({"lang": "en"}), "[1,0,0]"],
            )
            category, content, source_name = cursor.fetchone()

        self.assertEqual(category, "account")
        self.assertIn("password", content.lower())
        self.assertEqual(source_name, "guide")


if __name__ == "__main__":
    unittest.main()

