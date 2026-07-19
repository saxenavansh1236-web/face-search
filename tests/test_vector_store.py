"""
test_vector_store.py

Tests the ChromaDB wrapper logic (add/search/delete/update) using
fake embeddings, without needing DeepFace to actually generate them.
"""

from app.services.vector_store import (
    add_face, search_faces, delete_face, count_faces,
    get_all_faces, update_metadata, delete_many,
)


def _fake_embedding(seed_value):
    """512-dim fake embedding, distinct per seed for testing distance ordering."""
    return [float(seed_value)] * 512


def test_add_and_count_face():
    before = count_faces()
    add_face("pytest_face_1", _fake_embedding(0.1), source_url="test", person_id="person_a")
    assert count_faces() == before + 1
    delete_face("pytest_face_1")


def test_search_returns_closest_match():
    add_face("pytest_close", _fake_embedding(0.5), source_url="test", person_id="close_person")
    add_face("pytest_far", _fake_embedding(9.0), source_url="test", person_id="far_person")

    results = search_faces(_fake_embedding(0.5), top_k=2)
    assert results["ids"][0][0] == "pytest_close"

    delete_face("pytest_close")
    delete_face("pytest_far")


def test_delete_face_removes_it():
    add_face("pytest_delete_me", _fake_embedding(0.2), source_url="test")
    assert delete_face("pytest_delete_me") is True
    assert delete_face("pytest_delete_me") is False


def test_update_metadata():
    add_face("pytest_update", _fake_embedding(0.3), source_url="old_source", person_id="old_person")
    updated = update_metadata("pytest_update", source_url="new_source", person_id="new_person")
    assert updated is True

    faces = get_all_faces()
    face = next(f for f in faces if f["id"] == "pytest_update")
    assert face["source_url"] == "new_source"
    assert face["person_id"] == "new_person"

    delete_face("pytest_update")


def test_bulk_delete():
    add_face("pytest_bulk_1", _fake_embedding(0.4), source_url="test")
    add_face("pytest_bulk_2", _fake_embedding(0.6), source_url="test")

    deleted_count = delete_many(["pytest_bulk_1", "pytest_bulk_2", "nonexistent_id"])
    assert deleted_count == 2