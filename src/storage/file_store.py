# -*- coding: utf-8 -*-
"""
File-based artifact storage backend.

Provides a simple local file system store for binary artifacts such as
serialized models, plot images, data files, and configuration exports.
Artifacts are organized by run ID in a flat directory structure.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileArtifactStore:
    """Local file-system artifact store.

    Stores artifacts under ``<root_dir>/<run_id>/<filename>``.

    Parameters
    ----------
    root_dir : str
        Root directory for all artifacts. Created automatically.
    """

    def __init__(self, root_dir: str = "./artifacts") -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root_dir(self) -> str:
        return str(self._root)

    def save_artifact(self, run_id: str, source_path: Path) -> str:
        """Copy a local file into the artifact store.

        Parameters
        ----------
        run_id : str
            Run to associate the artifact with.
        source_path : Path
            Source file to copy.

        Returns
        -------
        str
            URI of the stored artifact (relative to ``root_dir``).
        """
        dest_dir = self._root / run_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / source_path.name
        shutil.copy2(str(source_path), str(dest_path))
        uri = f"{run_id}/{source_path.name}"
        logger.info("Saved artifact %s -> %s", source_path, dest_path)
        return uri

    def load_artifact(self, uri: str) -> bytes:
        """Load an artifact's raw bytes by its URI.

        Parameters
        ----------
        uri : str
            Artifact URI (relative path like ``<run_id>/<filename>``).

        Returns
        -------
        bytes
            Raw file contents.

        Raises
        ------
        FileNotFoundError
            If the artifact does not exist on disk.
        """
        path = self._root / uri
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        return path.read_bytes()

    def load_artifact_path(self, uri: str) -> Path:
        """Return the absolute local path for an artifact.

        Parameters
        ----------
        uri : str
            Artifact URI.

        Returns
        -------
        Path
            Absolute path to the artifact file.

        Raises
        ------
        FileNotFoundError
            If the artifact does not exist on disk.
        """
        path = self._root / uri
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        return path.resolve()

    def list_artifacts(self, run_id: str) -> list[dict[str, str]]:
        """List all artifacts stored for a given run.

        Returns
        -------
        list[dict[str, str]]
            Each entry contains ``name``, ``uri``, ``size_bytes``, and
            ``sha256`` keys.
        """
        run_dir = self._root / run_id
        if not run_dir.exists():
            return []

        artifacts = []
        for path in sorted(run_dir.iterdir()):
            if path.is_file():
                artifacts.append(
                    {
                        "name": path.name,
                        "uri": f"{run_id}/{path.name}",
                        "size_bytes": str(path.stat().st_size),
                        "sha256": self._sha256(path),
                    }
                )
        return artifacts

    def delete_artifact(self, uri: str) -> bool:
        """Delete a single artifact by URI.

        Returns
        -------
        bool
            ``True`` if the file existed and was deleted.
        """
        path = self._root / uri
        if path.exists():
            path.unlink()
            logger.info("Deleted artifact %s", path)
            return True
        return False

    def delete_run_artifacts(self, run_id: str) -> int:
        """Delete all artifacts for a run.

        Returns
        -------
        int
            Number of files deleted.
        """
        run_dir = self._root / run_id
        if not run_dir.exists():
            return 0

        count = 0
        for path in run_dir.iterdir():
            if path.is_file():
                path.unlink()
                count += 1
        try:
            run_dir.rmdir()
        except OSError:
            pass
        logger.info("Deleted %d artifacts for run %s", count, run_id)
        return count

    @staticmethod
    def _sha256(path: Path) -> str:
        """Compute SHA-256 checksum for a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
