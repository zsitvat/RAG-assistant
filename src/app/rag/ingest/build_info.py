import hashlib
from pathlib import Path

from app.rag.ingest.chunker import CHUNK_OVERLAP, CHUNK_SIZE, SHORT_SECTION_MERGE_THRESHOLD
from app.rag.model import IndexBuildInfo


class IndexBuildInfoBuilder:
    """Computes the corpus build info used to decide whether ingestion can be skipped."""

    def __init__(self, corpus_dir: Path, rules_path: Path) -> None:
        """Stores the corpus and rules paths used to compute the build hash."""
        self._corpus_dir = corpus_dir
        self._rules_path = rules_path

    def compute_hash(self) -> str:
        """Hashes the corpus docx files and rules file to detect content changes."""
        hasher = hashlib.sha256()
        for path in sorted(self._corpus_dir.glob("*.docx")):
            hasher.update(path.name.encode())
            hasher.update(path.read_bytes())
        hasher.update(self._rules_path.read_bytes())
        return hasher.hexdigest()

    def build(
        self, embedding_model: str, embedding_revision: str, dimension: int
    ) -> IndexBuildInfo:
        """Builds the IndexBuildInfo describing the corpus, chunking and embedding config."""
        return IndexBuildInfo(
            corpus_hash=self.compute_hash(),
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            short_section_merge_threshold=SHORT_SECTION_MERGE_THRESHOLD,
            embedding_model=embedding_model,
            embedding_revision=embedding_revision,
            dimension=dimension,
        )
