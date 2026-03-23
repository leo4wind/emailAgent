import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_base import (
    batch_insert_knowledge,
    build_knowledge_chunks,
    execute_sql_file,
    get_postgresql_connection,
)

load_dotenv()


def load_repo_documents(project_root: Path) -> List[Dict[str, str]]:
    repo_docs = []
    doc_dir = project_root / "doc"
    for file_path in sorted(doc_dir.glob("*")):
        if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md"}:
            repo_docs.append(
                {
                    "source_id": f"repo:{file_path.name}",
                    "title": file_path.stem,
                    "category": "documentation",
                    "content": file_path.read_text(encoding="utf-8"),
                    "metadata": {
                        "source": "repo_doc",
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                    },
                }
            )
    return repo_docs


def build_seed_documents(project_root: Path) -> List[Dict[str, object]]:
    seed_docs: List[Dict[str, object]] = [
        {
            "source_id": "seed:password_reset",
            "title": "Password Reset Guide",
            "category": "account",
            "content": (
                "Users can reset passwords from Settings > Security > Change Password. "
                "Require at least 12 characters with uppercase, lowercase, numbers, and symbols. "
                "If email verification fails, ask the user to retry after 5 minutes."
            ),
            "metadata": {"source": "seed", "lang": "en", "topic": "password"},
        },
        {
            "source_id": "seed:billing_refund",
            "title": "Billing Refund Policy",
            "category": "billing",
            "content": (
                "Duplicate charges should be refunded after confirming transaction IDs. "
                "Normal refund processing takes 3 to 5 business days. "
                "Urgent billing issues should be escalated to human review immediately."
            ),
            "metadata": {"source": "seed", "lang": "en", "topic": "billing"},
        },
        {
            "source_id": "seed:mfa_setup",
            "title": "MFA Setup Instructions",
            "category": "security",
            "content": (
                "Users can enable MFA from Settings > Security > Multi-factor Authentication. "
                "Support email OTP and authenticator apps. "
                "If the OTP is not received, verify the mailbox spam folder and resend once."
            ),
            "metadata": {"source": "seed", "lang": "en", "topic": "security"},
        },
    ]
    seed_docs.extend(load_repo_documents(project_root))
    return seed_docs


def delete_existing_sources(table_name: str, source_ids: List[str]) -> int:
    if not source_ids:
        return 0

    with get_postgresql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {table_name} WHERE source_id = ANY(%s)",
                (source_ids,),
            )
            deleted_count = cursor.rowcount
        conn.commit()
    return deleted_count


def main() -> None:
    project_root = PROJECT_ROOT
    table_name = os.getenv("PGVECTOR_TABLE_NAME", "knowledge_base_documents")
    sql_path = project_root / "sql" / "create_knowledge_base_tables.sql"

    execute_sql_file(str(sql_path))

    documents = build_seed_documents(project_root)
    deleted_count = delete_existing_sources(
        table_name=table_name,
        source_ids=[document["source_id"] for document in documents],
    )

    chunks = []
    for document in documents:
        chunks.extend(
            build_knowledge_chunks(
                source_id=str(document["source_id"]),
                title=str(document["title"]),
                category=str(document["category"]),
                text=str(document["content"]),
                metadata=dict(document.get("metadata", {})),
                chunk_size=800,
                chunk_overlap=120,
            )
        )

    inserted_count = batch_insert_knowledge(chunks, table_name=table_name)
    print(
        json.dumps(
            {
                "table_name": table_name,
                "documents": len(documents),
                "deleted_old_chunks": deleted_count,
                "inserted_chunks": inserted_count,
                "sql_path": str(sql_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
