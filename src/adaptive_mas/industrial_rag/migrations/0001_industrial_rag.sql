CREATE TABLE IF NOT EXISTS rag_documents (
  document_id TEXT PRIMARY KEY, security_code TEXT, entity_name TEXT, title TEXT NOT NULL,
  document_type TEXT NOT NULL, report_period TEXT, accounting_basis TEXT, aliases_json TEXT NOT NULL,
  metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_document_versions (
  version_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES rag_documents(document_id),
  source_id TEXT NOT NULL UNIQUE, content_sha256 TEXT NOT NULL, source_url TEXT NOT NULL,
  rights_note TEXT NOT NULL, published_at TEXT, first_available_at TEXT, captured_at TEXT NOT NULL,
  ingested_at TEXT NOT NULL, valid_from TEXT, valid_to TEXT, revised_at TEXT,
  time_basis TEXT NOT NULL, parser_name TEXT NOT NULL, parser_version TEXT NOT NULL,
  parser_confidence DOUBLE PRECISION, parser_status TEXT NOT NULL, parser_error_code TEXT, parser_error TEXT, metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS rag_versions_lookup_idx ON rag_document_versions(document_id, first_available_at);
CREATE TABLE IF NOT EXISTS rag_sections (
  section_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  title TEXT NOT NULL, level INTEGER NOT NULL, parent_section_id TEXT, start_page INTEGER, end_page INTEGER, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_spans (
  span_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  page_number INTEGER, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL, text TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_blocks (
  block_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  section_id TEXT, page_number INTEGER, block_type TEXT NOT NULL, text TEXT NOT NULL,
  parent_block_id TEXT, table_id TEXT, row_start INTEGER, row_end INTEGER, parser_confidence DOUBLE PRECISION,
  quarantine_reason TEXT, payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS rag_blocks_version_page_idx ON rag_blocks(version_id, page_number);
CREATE TABLE IF NOT EXISTS rag_tables (
  table_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  title TEXT, header_rows INTEGER NOT NULL, continued_from_table_id TEXT, continuation_confidence DOUBLE PRECISION,
  unit_text TEXT, accounting_basis TEXT, report_period TEXT, page_start INTEGER, page_end INTEGER, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_cells (
  cell_id TEXT PRIMARY KEY, table_id TEXT NOT NULL REFERENCES rag_tables(table_id), version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  row_index INTEGER NOT NULL, column_index INTEGER NOT NULL, raw_text TEXT NOT NULL, normalized_value TEXT,
  value_kind TEXT NOT NULL, page_number INTEGER, unit TEXT, multiplier TEXT, currency TEXT, accounting_basis TEXT,
  report_period TEXT, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_facts (
  fact_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  security_code TEXT NOT NULL, entity_name TEXT NOT NULL, metric TEXT NOT NULL, raw_value TEXT NOT NULL,
  normalized_value TEXT NOT NULL, unit TEXT NOT NULL, multiplier TEXT NOT NULL, currency TEXT NOT NULL,
  report_period TEXT NOT NULL, consolidation TEXT NOT NULL, confidence DOUBLE PRECISION NOT NULL, review_status TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS rag_facts_lookup_idx ON rag_facts(security_code, metric, report_period);
CREATE TABLE IF NOT EXISTS rag_quarantine (
  quarantine_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  error_code TEXT NOT NULL, reason TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'open',
  reviewed_by TEXT, reviewed_at TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_access_grants (
  principal_id TEXT NOT NULL, document_id TEXT NOT NULL REFERENCES rag_documents(document_id),
  can_read SMALLINT NOT NULL, changed_at TEXT NOT NULL, PRIMARY KEY(principal_id, document_id)
);
CREATE TABLE IF NOT EXISTS rag_access_state (singleton INTEGER PRIMARY KEY CHECK(singleton=1), revision INTEGER NOT NULL);
INSERT INTO rag_access_state(singleton, revision) VALUES(1,0) ON CONFLICT(singleton) DO NOTHING;
CREATE TABLE IF NOT EXISTS rag_outbox (
  event_id BIGSERIAL PRIMARY KEY, event_key TEXT NOT NULL UNIQUE, event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
  consumer_id TEXT, lease_until DOUBLE PRECISION, created_at TEXT NOT NULL, acknowledged_at TEXT
);
CREATE INDEX IF NOT EXISTS rag_outbox_claim_idx ON rag_outbox(status, lease_until, event_id);
CREATE TABLE IF NOT EXISTS rag_consumer_receipts (
  consumer_id TEXT NOT NULL, event_key TEXT NOT NULL, result_json TEXT NOT NULL, completed_at TEXT NOT NULL,
  PRIMARY KEY(consumer_id, event_key)
);
CREATE TABLE IF NOT EXISTS rag_corpus_releases (
  release_id TEXT PRIMARY KEY, spec_json TEXT NOT NULL, spec_sha256 TEXT NOT NULL,
  status TEXT NOT NULL, created_at TEXT NOT NULL, published_at TEXT
);
CREATE TABLE IF NOT EXISTS rag_release_versions (
  release_id TEXT NOT NULL REFERENCES rag_corpus_releases(release_id),
  version_id TEXT NOT NULL REFERENCES rag_document_versions(version_id),
  PRIMARY KEY(release_id, version_id)
);
CREATE TABLE IF NOT EXISTS rag_release_backends (
  release_id TEXT NOT NULL REFERENCES rag_corpus_releases(release_id), backend TEXT NOT NULL,
  physical_index_handle TEXT NOT NULL, backend_version TEXT NOT NULL,
  source_manifest_sha256 TEXT NOT NULL, index_manifest_sha256 TEXT NOT NULL,
  document_count INTEGER NOT NULL, record_count INTEGER NOT NULL,
  embedding_model TEXT NOT NULL, embedding_dimension INTEGER, ready_at TEXT NOT NULL,
  PRIMARY KEY(release_id, backend)
);
CREATE TABLE IF NOT EXISTS rag_active_release (singleton INTEGER PRIMARY KEY CHECK(singleton=1), release_id TEXT REFERENCES rag_corpus_releases(release_id), updated_at TEXT NOT NULL);
INSERT INTO rag_active_release(singleton, release_id, updated_at) VALUES(1,NULL,'1970-01-01T00:00:00+00:00') ON CONFLICT(singleton) DO NOTHING;
CREATE TABLE IF NOT EXISTS rag_release_audit (
  audit_id BIGSERIAL PRIMARY KEY, action TEXT NOT NULL, release_id TEXT NOT NULL,
  previous_release_id TEXT, reason TEXT NOT NULL, actor_id TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_query_cache (
  cache_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
);
