from pathlib import Path

from knowledge_base import execute_sql_file


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    sql_path = project_root / "sql" / "create_knowledge_base_tables.sql"
    execute_sql_file(str(sql_path))
    print(f"Knowledge base schema initialized from: {sql_path}")


if __name__ == "__main__":
    main()
