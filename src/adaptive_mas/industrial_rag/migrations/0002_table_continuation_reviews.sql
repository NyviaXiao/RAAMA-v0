CREATE TABLE IF NOT EXISTS rag_continuation_state (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1), revision INTEGER NOT NULL
);
INSERT INTO rag_continuation_state(singleton, revision) VALUES(1,0)
ON CONFLICT(singleton) DO NOTHING;
CREATE TABLE IF NOT EXISTS rag_table_continuation_reviews (
  child_table_id TEXT PRIMARY KEY REFERENCES rag_tables(table_id),
  previous_table_id TEXT NOT NULL REFERENCES rag_tables(table_id),
  version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  content_sha256 TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('confirmed','rejected','revoked')),
  reviewer_id TEXT NOT NULL,
  reviewed_at TEXT NOT NULL,
  note TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_table_continuation_events (
  event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  child_table_id TEXT NOT NULL,
  previous_table_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  action TEXT NOT NULL CHECK(action IN ('confirm','reject','revoke')),
  reviewer_id TEXT NOT NULL,
  reviewed_at TEXT NOT NULL,
  note TEXT NOT NULL
);
