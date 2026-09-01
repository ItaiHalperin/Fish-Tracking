#!/usr/bin/env python3
"""
ML Storage — structured local storage for datasets and model artefacts.

Provides:
    MLStorage        — top-level façade
    ├── DatasetManager   — versioned dataset storage
    └── ModelRegistry    — model weights & metrics storage

Directory layout:
    ml_storage/
    ├── datasets/
    │   ├── v1_<description>/
    │   │   ├── version.json
    │   │   └── <data files>
    │   └── v2_<description>/
    │       └── ...
    └── model_registry/
        ├── detection/
        │   └── <model_name>/
        │       └── run_<timestamp>/
        │           ├── weights/best.pt
        │           ├── results.csv
        │           └── run.json
        └── classifier/
            └── <model_name>/
                └── run_<timestamp>/
                    ├── weights/best.pt
                    ├── results.csv
                    └── run.json
"""

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _slugify(text: str) -> str:
    """Convert a description string into a filesystem-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


class DatasetManager:
    """Manages versioned datasets under ``<root>/``."""

    VERSION_DIR_PATTERN = re.compile(r"^v(\d+)_(.+)$")

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _parse_version_dirs(self) -> list[dict]:
        """Return sorted list of ``{version, description, path, created_at}`` dicts."""
        if not self.root.exists():
            return []

        versions = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            m = self.VERSION_DIR_PATTERN.match(d.name)
            if not m:
                continue
            info = {"version": int(m.group(1)), "slug": m.group(2), "path": d}
            # Read optional version.json for richer metadata
            meta_file = d / "version.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                info["description"] = meta.get("description", m.group(2))
                info["created_at"] = meta.get("created_at")
            else:
                info["description"] = m.group(2)
                info["created_at"] = None
            versions.append(info)

        versions.sort(key=lambda v: v["version"])
        return versions

    def _next_version_number(self) -> int:
        existing = self._parse_version_dirs()
        if not existing:
            return 1
        return existing[-1]["version"] + 1

    def create_version(self, description: str) -> Path:
        """Create a new numbered version directory and write ``version.json``.

        Returns the path to the new version directory.
        """
        version_num = self._next_version_number()
        slug = _slugify(description)
        dir_name = f"v{version_num}_{slug}"
        version_dir = self.root / dir_name
        version_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "version": version_num,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(version_dir / "version.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Created dataset version: {version_dir}")
        return version_dir

    def list_versions(self) -> list[dict]:
        """Return all versions, sorted ascending."""
        return self._parse_version_dirs()

    def latest_version(self) -> Path | None:
        """Return the path of the latest version, or ``None``."""
        versions = self._parse_version_dirs()
        if not versions:
            return None
        return versions[-1]["path"]

    def get_version(self, version: int) -> Path:
        """Return the path for a specific version number.

        Raises ``FileNotFoundError`` if it does not exist.
        """
        for v in self._parse_version_dirs():
            if v["version"] == version:
                return v["path"]
        raise FileNotFoundError(
            f"Version {version} not found in '{self.root}'"
        )


class ModelRegistry:
    """Manages model artefacts under ``<root>/<model_name>/``.

    Supports two workflows:

    1. **YOLO-native** (preferred): call ``prepare_run()`` to get ``project``
       and ``name`` values for YOLO, then ``finalize_run()`` after training to
       write ``run.json``.  YOLO writes weights, metrics, plots, etc. directly
       into the registry.

    2. **Manual import**: call ``register_run()`` to copy external weights /
       metrics into a new run directory.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _model_root(self, model_name: str) -> Path:
        return self.root / model_name

    def prepare_run(self, model_name: str) -> tuple[str, str]:
        """Return ``(project, name)`` strings to pass to ``model.train()``.

        YOLO will create ``<project>/<name>/`` and write all artefacts there.
        The directory is inside the model registry so no copying is needed.
        """
        project = str(self._model_root(model_name))
        name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return project, name

    def finalize_run(
        self,
        model_name: str,
        run_dir: Path,
        metadata: dict | None = None,
    ) -> Path:
        """Write ``run.json`` into an existing run directory.

        Call this after ``model.train()`` completes to attach metadata to the
        run.  Returns the run directory path.
        """
        run_dir = Path(run_dir)
        run_meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        with open(run_dir / "run.json", "w") as f:
            json.dump(run_meta, f, indent=2)

        print(f"Finalized run: {run_dir}")
        return run_dir

    def set_default_weights(self, model_name: str, weights_path: Path) -> Path:
        """Copy an external weights file as the canonical default for a model.

        The file is always stored as ``<model_root>/default.pt`` regardless of
        its original name.  This is intentionally separate from trained runs —
        it represents weights you brought in from outside, not a training
        artefact produced by this project.

        Returns the path to the stored ``default.pt``.
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        model_root = self._model_root(model_name)
        model_root.mkdir(parents=True, exist_ok=True)

        dst = model_root / "default.pt"
        shutil.copy2(weights_path, dst)
        print(f"Set default weights: {dst}")
        return dst

    def get_default_weights(self, model_name: str) -> Path | None:
        """Return the path to the default weights, or ``None`` if not set."""
        dst = self._model_root(model_name) / "default.pt"
        return dst if dst.exists() else None

    def list_runs(self, model_name: str) -> list[dict]:
        """List all registered runs for a model, sorted by directory name."""
        model_dir = self._model_root(model_name)
        if not model_dir.exists():
            return []

        runs = []
        for d in sorted(model_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith("run_"):
                continue
            info = {"name": d.name, "path": d}
            meta_file = d / "run.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    info.update(json.load(f))
            runs.append(info)
        return runs

    def latest_run(self, model_name: str) -> Path | None:
        """Return the path of the most recent run, or ``None``."""
        runs = self.list_runs(model_name)
        if not runs:
            return None
        return runs[-1]["path"]

    def get_weights(
        self,
        model_name: str,
        run: int | None = None,
        filename: str = "best.pt",
    ) -> Path:
        """Return the path to a weights file.

        Resolution order when ``run`` is not specified:
        1. ``<model_root>/default.pt``  — imported / pre-trained weights
        2. Latest ``run_*/weights/best.pt``  — most recent trained run

        When ``run`` is specified (1-indexed), only trained runs are searched:
        1. ``<run_dir>/weights/<filename>``  (YOLO-native layout)
        2. ``<run_dir>/<filename>``           (flat layout)

        Raises:
            FileNotFoundError: If nothing is found.
        """
        model_root = self._model_root(model_name)

        if run is None:
            # Prefer default weights (imported)
            default = model_root / "default.pt"
            if default.exists():
                return default

            # Fall back to the latest trained run
            all_runs = self.list_runs(model_name)
            if not all_runs:
                raise FileNotFoundError(
                    f"No weights found for model '{model_name}': "
                    f"no default.pt and no trained runs."
                )
            run_dir = Path(all_runs[-1]["path"])
        else:
            all_runs = self.list_runs(model_name)
            if not all_runs:
                raise FileNotFoundError(f"No trained runs for model '{model_name}'")
            if run < 1 or run > len(all_runs):
                raise FileNotFoundError(
                    f"Run {run} not found for model '{model_name}' "
                    f"(have {len(all_runs)} run(s))"
                )
            run_dir = Path(all_runs[run - 1]["path"])

        # YOLO-native layout first, then flat
        yolo_path = run_dir / "weights" / filename
        if yolo_path.exists():
            return yolo_path
        flat_path = run_dir / filename
        if flat_path.exists():
            return flat_path

        raise FileNotFoundError(
            f"Weights file '{filename}' not found in {run_dir} "
            f"(checked weights/{filename} and {filename})"
        )


class MLStorage:
    """Single entry-point for all managed ML storage.

    Usage::

        storage = MLStorage("ml_storage")
        version_dir = storage.datasets.create_version("baseline")

        # Detection model (YOLO tracker)
        project, name = storage.detection_models.prepare_run("goldfish_yolo")

        # Classifier model
        project, name = storage.classifier_models.prepare_run("fish_position_classifier")
    """

    def __init__(self, root: str | Path = "ml_storage"):
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.datasets = DatasetManager(self.root / "datasets")
        registry_root = self.root / "model_registry"
        self.detection_models = ModelRegistry(registry_root / "detection")
        self.classifier_models = ModelRegistry(registry_root / "classifier")

    def __repr__(self) -> str:
        return f"MLStorage(root={self.root!r})"
