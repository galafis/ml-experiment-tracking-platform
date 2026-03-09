# -*- coding: utf-8 -*-
"""Unit tests for FileArtifactStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.storage.file_store import FileArtifactStore


class TestFileArtifactStore:
    """Tests for the local file artifact store."""

    def test_save_and_load_artifact(self, file_store: FileArtifactStore, tmp_path: Path):
        """Saving an artifact should make it loadable by URI."""
        source = tmp_path / "model.pkl"
        source.write_bytes(b"fake_model_data")

        uri = file_store.save_artifact("run_123", source)
        assert uri == "run_123/model.pkl"

        data = file_store.load_artifact(uri)
        assert data == b"fake_model_data"

    def test_list_artifacts(self, file_store: FileArtifactStore, tmp_path: Path):
        """Listing artifacts should return all files for a run."""
        for name in ("a.txt", "b.csv", "c.pkl"):
            source = tmp_path / name
            source.write_bytes(b"data")
            file_store.save_artifact("run_456", source)

        artifacts = file_store.list_artifacts("run_456")
        assert len(artifacts) == 3
        names = {a["name"] for a in artifacts}
        assert names == {"a.txt", "b.csv", "c.pkl"}

    def test_list_artifacts_empty_run(self, file_store: FileArtifactStore):
        """Listing artifacts for a nonexistent run should return empty."""
        assert file_store.list_artifacts("nonexistent") == []

    def test_delete_artifact(self, file_store: FileArtifactStore, tmp_path: Path):
        """Deleting an artifact should remove it from disk."""
        source = tmp_path / "deleteme.txt"
        source.write_bytes(b"will_be_deleted")
        uri = file_store.save_artifact("run_del", source)

        assert file_store.delete_artifact(uri) is True
        assert file_store.delete_artifact(uri) is False

    def test_load_nonexistent_raises(self, file_store: FileArtifactStore):
        """Loading a nonexistent artifact should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            file_store.load_artifact("no_such/file.txt")

    def test_sha256_checksum(self, file_store: FileArtifactStore, tmp_path: Path):
        """Artifact listing should include valid SHA-256 checksums."""
        source = tmp_path / "checksummed.bin"
        source.write_bytes(b"checksum_test_data")
        file_store.save_artifact("run_ck", source)

        artifacts = file_store.list_artifacts("run_ck")
        assert len(artifacts) == 1
        assert len(artifacts[0]["sha256"]) == 64  # SHA-256 hex digest
