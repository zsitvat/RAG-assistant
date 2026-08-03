INDEX_NAME = "idx:chunks"
KEY_PREFIX = "chunk"
TOP_K = 8
MIN_CONFIDENCE_THRESHOLD = 0.5
CONTEXT_TOKEN_BUDGET = 1800

VECTOR_DIMENSION = 384
DISTANCE_METRIC = "COSINE"
INDEXING_ALGORITHM = "HNSW"
VECTOR_DATATYPE = "FLOAT32"

METADATA_SCHEMA = [
    {"name": "doc_id", "type": "tag"},
    {"name": "section_id", "type": "tag"},
    {"name": "categories", "type": "tag"},
    {"name": "rule_ids", "type": "tag"},
    {"name": "section", "type": "text", "attrs": {"no_stem": True}},
]
