"""ChromaDB vector store."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import chromadb


DEFAULT_DB_PATH = Path.home() / ".sentrysearch" / "db"
DOUBAO_MODEL = "doubao-embedding-vision-251215"
QWEN_MODEL = "qwen3-vl-embedding"
DEFAULT_SEGMENTATION = "chunk"


class BackendMismatchError(RuntimeError):
    """Raised when search backend/model doesn't match the indexed backend/model."""


def _collection_name(
    backend: str,
    model: str | None = None,
    segmentation: str = DEFAULT_SEGMENTATION,
) -> str:
    """Return ChromaDB collection name for a backend and optional model."""
    prefix = "dashcam_chunks" if segmentation == "chunk" else "dashcam_shots"
    if backend == "gemini":
        return prefix
    if backend == "doubao":
        return f"{prefix}_doubao"
    if backend == "qwen":
        return f"{prefix}_qwen"
    if model:
        return f"{prefix}_local_{model}"
    # Legacy: local backend without model distinction
    return f"{prefix}_local"


def detect_index_details(
    db_path: str | Path | None = None,
    segmentation: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Return ``(backend, model, segmentation)`` for the first index with data."""
    db_path = str(db_path or DEFAULT_DB_PATH)
    if not Path(db_path).exists():
        return None, None, None
    client = chromadb.PersistentClient(path=db_path)
    existing = {c.name for c in client.list_collections()}

    segmentations = [segmentation] if segmentation else ["chunk", "shot"]
    for mode in segmentations:
        prefix = "dashcam_chunks" if mode == "chunk" else "dashcam_shots"
        static_candidates = [
            (prefix, "gemini", None),
            (f"{prefix}_doubao", "doubao", DOUBAO_MODEL),
            (f"{prefix}_qwen", "qwen", QWEN_MODEL),
        ]
        for name, backend, default_model in static_candidates:
            if name not in existing:
                continue
            col = client.get_collection(name)
            if col.count() == 0:
                continue
            meta = col.metadata or {}
            return backend, meta.get("embedding_model", default_model), mode

        local_prefix = f"{prefix}_local_"
        for name in sorted(existing):
            if not name.startswith(local_prefix):
                continue
            col = client.get_collection(name)
            if col.count() == 0:
                continue
            meta = col.metadata or {}
            model = meta.get("embedding_model")
            if model is None:
                model = name.removeprefix(local_prefix)
            return "local", model, mode

        legacy_name = f"{prefix}_local"
        if mode == "chunk" and legacy_name in existing:
            col = client.get_collection(legacy_name)
            if col.count() > 0:
                meta = col.metadata or {}
                return "local", meta.get("embedding_model", "qwen8b"), mode

    return None, None, None


def detect_index(
    db_path: str | Path | None = None,
    segmentation: str | None = None,
) -> tuple[str | None, str | None]:
    """Return ``(backend, model)`` for the first index with data.

    Returns ``(None, None)`` when no index contains data.
    Checks gemini first, then Doubao, then model-specific local collections, then the
    legacy ``dashcam_chunks_local`` collection (treated as qwen8b).
    """
    backend, model, _segmentation = detect_index_details(db_path, segmentation=segmentation)
    return backend, model


def detect_backend(db_path: str | Path | None = None) -> str | None:
    """Return the backend that has indexed data, or None if empty."""
    backend, _ = detect_index(db_path)
    return backend


def _make_chunk_id(source_file: str, start_time: float) -> str:
    """Deterministic chunk ID from source file + start time."""
    raw = f"{source_file}:{start_time}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class SentryStore:
    """Persistent vector store backed by ChromaDB."""

    def __init__(self, db_path: str | Path | None = None, backend: str = "gemini",
                 model: str | None = None, segmentation: str = DEFAULT_SEGMENTATION):
        db_path = str(db_path or DEFAULT_DB_PATH)
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=db_path)
        self._backend = backend
        self._segmentation = segmentation
        if backend == "doubao" and model is None:
            model = DOUBAO_MODEL
        if backend == "qwen" and model is None:
            model = QWEN_MODEL
        self._model = model
        # Separate collection per backend+model so incompatible vectors never mix.
        col_name = _collection_name(backend, model, segmentation=segmentation)
        metadata = {
            "hnsw:space": "cosine",
            "embedding_backend": backend,
            "segmentation_mode": segmentation,
        }
        if model:
            metadata["embedding_model"] = model
        self._collection = self._client.get_or_create_collection(
            name=col_name,
            metadata=metadata,
        )

    @property
    def collection(self) -> chromadb.Collection:
        return self._collection

    def get_backend(self) -> str:
        """Return the backend this index was built with."""
        meta = self._collection.metadata or {}
        return meta.get("embedding_backend", "gemini")

    def get_model(self) -> str | None:
        """Return the model this index was built with, or None."""
        meta = self._collection.metadata or {}
        return meta.get("embedding_model")

    def get_segmentation(self) -> str:
        """Return the segmentation mode this index was built with."""
        meta = self._collection.metadata or {}
        return meta.get("segmentation_mode", DEFAULT_SEGMENTATION)

    def check_backend(self, backend: str) -> None:
        """Raise BackendMismatchError if *backend* doesn't match the index."""
        indexed_backend = self.get_backend()
        if indexed_backend != backend:
            raise BackendMismatchError(
                f"This index was built with the {indexed_backend} backend. "
                f"Search with --backend {indexed_backend} or re-index with "
                f"--backend {backend}."
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunk(
        self,
        chunk_id: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """Store a single chunk embedding with metadata.

        Required metadata keys: source_file, start_time, end_time.
        An indexed_at ISO timestamp is added automatically.
        """
        meta = {
            "source_file": metadata["source_file"],
            "start_time": float(metadata["start_time"]),
            "end_time": float(metadata["end_time"]),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        # Carry over any extra metadata the caller provides
        for key in metadata:
            if key not in meta and key != "embedding":
                meta[key] = metadata[key]

        self._collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            metadatas=[meta],
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        """Batch-store chunks. Each dict must have 'embedding' and metadata keys."""
        now = datetime.now(timezone.utc).isoformat()
        ids = []
        embeddings = []
        metadatas = []

        for chunk in chunks:
            chunk_id = _make_chunk_id(chunk["source_file"], chunk["start_time"])
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            meta = {
                "source_file": chunk["source_file"],
                "start_time": float(chunk["start_time"]),
                "end_time": float(chunk["end_time"]),
                "indexed_at": now,
            }
            for key, value in chunk.items():
                if key not in meta and key not in {"embedding", "chunk_path"}:
                    meta[key] = value
            metadatas.append(meta)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict]:
        """Return top N results with distances and metadata."""
        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
        )

        hits = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            hits.append({
                "source_file": meta["source_file"],
                "start_time": meta["start_time"],
                "end_time": meta["end_time"],
                "score": 1.0 - distance,  # cosine distance → similarity
                "distance": distance,
            })
        return hits

    def is_indexed(self, source_file: str) -> bool:
        """Check whether any chunks from source_file are already stored."""
        results = self._collection.get(
            where={"source_file": source_file},
            limit=1,
        )
        return len(results["ids"]) > 0

    def remove_file(self, source_file: str) -> int:
        """Remove all chunks for a given source file. Returns count removed."""
        results = self._collection.get(where={"source_file": source_file})
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def get_stats(self) -> dict:
        """Return store statistics."""
        total = self._collection.count()
        if total == 0:
            return {"total_chunks": 0, "unique_source_files": 0, "source_files": []}

        # Fetch all metadata (only the fields we need)
        all_meta = self._collection.get(include=["metadatas"])
        source_files = sorted({m["source_file"] for m in all_meta["metadatas"]})
        return {
            "total_chunks": total,
            "unique_source_files": len(source_files),
            "source_files": source_files,
        }
