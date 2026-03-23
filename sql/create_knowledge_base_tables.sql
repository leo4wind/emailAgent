CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS knowledge_base_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id TEXT NOT NULL,
    title TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(content, ''))
    ) STORED,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kb_source_id
    ON knowledge_base_documents (source_id);

CREATE INDEX IF NOT EXISTS idx_kb_category
    ON knowledge_base_documents (category);

CREATE INDEX IF NOT EXISTS idx_kb_metadata_gin
    ON knowledge_base_documents
    USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_kb_content_tsv
    ON knowledge_base_documents
    USING GIN (content_tsv);

CREATE INDEX IF NOT EXISTS idx_kb_embedding_ivfflat
    ON knowledge_base_documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_kb_updated_at ON knowledge_base_documents;

CREATE TRIGGER trg_kb_updated_at
BEFORE UPDATE ON knowledge_base_documents
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();
