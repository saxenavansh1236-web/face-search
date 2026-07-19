"""
vector_store.py

Thin wrapper around ChromaDB. Owns the collection used to store and
search face embeddings. Each stored face carries metadata:
  - source_url: where the image came from
  - person_id: groups multiple photos of the same person together
  - thumbnail: base64 JPEG preview (for the admin dashboard)
  - created_at: ISO timestamp
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import chromadb

from app.core.config import settings

_client = chromadb.PersistentClient(path=settings.db_path)

_collection = _client.get_or_create_collection(
    name=settings.collection_name,
    metadata={"hnsw:space": settings.distance_metric},
)


def add_face(
    face_id: str,
    embedding: List[float],
    source_url: str,
    person_id: Optional[str] = None,
    thumbnail: Optional[str] = None,
) -> None:
    """Stores a single face embedding under `face_id`, with metadata."""
    _collection.add(
        embeddings=[embedding],
        metadatas=[{
            "source_url": source_url or "",
            "person_id": person_id or face_id,
            "thumbnail": thumbnail or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }],
        ids=[face_id],
    )


def update_face(
    face_id: str,
    embedding: List[float],
    source_url: Optional[str] = None,
    person_id: Optional[str] = None,
    thumbnail: Optional[str] = None,
) -> bool:
    """
    Replaces an existing face's embedding/photo (re-index) without
    changing its ID. Returns False if the ID doesn't exist yet.
    """
    existing = _collection.get(ids=[face_id])
    if not existing.get("ids"):
        return False

    old_meta = (existing.get("metadatas") or [{}])[0] or {}
    new_meta = {
        "source_url": source_url if source_url is not None else old_meta.get("source_url", ""),
        "person_id": person_id if person_id is not None else old_meta.get("person_id", face_id),
        "thumbnail": thumbnail if thumbnail is not None else old_meta.get("thumbnail", ""),
        "created_at": old_meta.get("created_at", datetime.now(timezone.utc).isoformat()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    _collection.delete(ids=[face_id])
    _collection.add(embeddings=[embedding], metadatas=[new_meta], ids=[face_id])
    return True


def search_faces(embedding: List[float], top_k: int) -> Dict[str, Any]:
    """Finds the `top_k` stored embeddings closest to `embedding`."""
    return _collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
    )


def count_faces() -> int:
    return _collection.count()


def get_all_faces() -> List[Dict[str, Any]]:
    """Returns every indexed face's id + metadata, for the admin dashboard."""
    raw = _collection.get()
    ids = raw.get("ids", [])
    metadatas = raw.get("metadatas", [])
    faces = []
    for face_id, meta in zip(ids, metadatas):
        meta = meta or {}
        faces.append({
            "id": face_id,
            "source_url": meta.get("source_url", ""),
            "person_id": meta.get("person_id", face_id),
            "thumbnail": meta.get("thumbnail", ""),
            "created_at": meta.get("created_at", ""),
        })
    faces.sort(key=lambda f: f["created_at"], reverse=True)
    return faces


def delete_face(face_id: str) -> bool:
    existing = _collection.get(ids=[face_id])
    if not existing.get("ids"):
        return False
    _collection.delete(ids=[face_id])
    return True


def delete_many(face_ids: List[str]) -> int:
    """Deletes multiple faces by ID. Returns how many actually existed."""
    existing = _collection.get(ids=face_ids)
    found_ids = existing.get("ids", [])
    if found_ids:
        _collection.delete(ids=found_ids)
    return len(found_ids)


def update_metadata(face_id: str, source_url: Optional[str] = None, person_id: Optional[str] = None) -> bool:
    """Edits source_url/person_id on an existing face without touching its embedding."""
    existing = _collection.get(ids=[face_id], include=["embeddings", "metadatas"])
    if not existing.get("ids"):
        return False

    embedding = existing["embeddings"][0]
    old_meta = (existing.get("metadatas") or [{}])[0] or {}
    new_meta = dict(old_meta)
    if source_url is not None:
        new_meta["source_url"] = source_url
    if person_id is not None:
        new_meta["person_id"] = person_id
    new_meta["updated_at"] = datetime.now(timezone.utc).isoformat()

    _collection.delete(ids=[face_id])
    _collection.add(embeddings=[embedding], metadatas=[new_meta], ids=[face_id])
    return True