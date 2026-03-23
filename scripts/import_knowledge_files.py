import argparse
import json
from pathlib import Path
from typing import List

from knowledge_base import batch_insert_knowledge, build_knowledge_chunks


SUPPORTED_SUFFIXES = {".txt", ".md"}


def load_documents(input_path: Path, category: str) -> List[dict]:
    documents: List[dict] = []

    if input_path.is_file():
        files = [input_path]
        base_dir = input_path.parent
    else:
        files = sorted(
            file_path
            for file_path in input_path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        base_dir = input_path

    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        documents.append(
            {
                "source_id": str(file_path.relative_to(base_dir)),
                "title": file_path.stem,
                "category": category,
                "content": content,
                "metadata": {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "suffix": file_path.suffix.lower(),
                },
            }
        )

    return documents


def import_documents(
    input_path: Path,
    category: str,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    documents = load_documents(input_path, category)
    chunks = []

    for document in documents:
        chunks.extend(
            build_knowledge_chunks(
                source_id=document["source_id"],
                title=document["title"],
                category=document["category"],
                text=document["content"],
                metadata=document["metadata"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return batch_insert_knowledge(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import local knowledge files into pgvector.")
    parser.add_argument("input_path", help="File or directory containing .txt/.md knowledge files.")
    parser.add_argument("--category", required=True, help="Category to assign to imported documents.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size for preprocessing.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap for preprocessing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    inserted_count = import_documents(
        input_path=input_path,
        category=args.category,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(
        json.dumps(
            {
                "input_path": str(input_path),
                "category": args.category,
                "inserted_chunks": inserted_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
