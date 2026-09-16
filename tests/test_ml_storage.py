import json
import pytest
from pathlib import Path
from core.ml_storage import _slugify, DatasetManager, ModelRegistry, MLStorage

def test_slugify():
    assert _slugify("Hello World") == "hello_world"
    assert _slugify("  Spaces  And !@# Special  Chars  ") == "spaces_and_special_chars"
    assert _slugify("") == ""
    assert _slugify("version 1.0") == "version_1_0"
    assert _slugify("unicode 🐟 fish") == "unicode_fish"

def test_dataset_manager(tmp_path):
    dm = DatasetManager(tmp_path / "datasets")
    
    # Empty state
    assert dm.list_versions() == []
    assert dm.latest_version() is None
    with pytest.raises(FileNotFoundError):
        dm.get_version(1)
        
    # Create versions
    v1_path = dm.create_version("Baseline Model")
    assert v1_path.exists()
    assert v1_path.name.startswith("v1_baseline_model")
    assert (v1_path / "version.json").exists()
    
    v2_path = dm.create_version("Improved Data")
    assert v2_path.exists()
    assert v2_path.name.startswith("v2_improved_data")
    
    # List versions
    versions = dm.list_versions()
    assert len(versions) == 2
    assert versions[0]["version"] == 1
    assert versions[0]["slug"] == "baseline_model"
    assert versions[0]["description"] == "Baseline Model"
    assert versions[1]["version"] == 2
    
    # Latest version
    assert dm.latest_version() == v2_path
    
    # Get version
    assert dm.get_version(1) == v1_path
    assert dm.get_version(2) == v2_path
    with pytest.raises(FileNotFoundError):
        dm.get_version(3)

def test_model_registry_runs(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    model_name = "test_model"
    
    # Empty state
    assert reg.list_runs(model_name) == []
    assert reg.latest_run(model_name) is None
    
    # Prepare run
    project, name = reg.prepare_run(model_name)
    assert model_name in project
    assert name.startswith("run_")
    
    run_dir = Path(project) / name
    run_dir.mkdir(parents=True)
    
    # Finalize run
    final_dir = reg.finalize_run(model_name, run_dir, metadata={"metrics": 0.95})
    assert final_dir == run_dir
    assert (run_dir / "run.json").exists()
    
    with open(run_dir / "run.json") as f:
        meta = json.load(f)
        assert meta["metrics"] == 0.95
        assert "created_at" in meta
        
    runs = reg.list_runs(model_name)
    assert len(runs) == 1
    assert runs[0]["name"] == name
    assert runs[0]["path"] == run_dir
    assert runs[0]["metrics"] == 0.95
    
    assert reg.latest_run(model_name) == run_dir

def test_model_registry_default_weights(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    model_name = "test_model"
    
    assert reg.get_default_weights(model_name) is None
    
    # Create fake weights
    ext_weights = tmp_path / "external.pt"
    ext_weights.write_text("fake weights")
    
    dst = reg.set_default_weights(model_name, ext_weights)
    assert dst.name == "default.pt"
    assert dst.parent.name == model_name
    assert dst.read_text() == "fake weights"
    
    assert reg.get_default_weights(model_name) == dst

def test_model_registry_get_weights_resolution(tmp_path):
    reg = ModelRegistry(tmp_path / "registry")
    model_name = "test_model"
    
    with pytest.raises(FileNotFoundError, match="no default.pt and no trained runs"):
        reg.get_weights(model_name)
        
    # Setup runs
    r1 = reg._model_root(model_name) / "run_01"
    r1.mkdir(parents=True)
    (r1 / "best.pt").write_text("r1 flat") # flat layout
    reg.finalize_run(model_name, r1)
    
    r2 = reg._model_root(model_name) / "run_02"
    r2.mkdir(parents=True)
    (r2 / "weights").mkdir()
    (r2 / "weights" / "best.pt").write_text("r2 yolo") # yolo layout
    reg.finalize_run(model_name, r2)
    
    # 1. Latest trained run (no default.pt) -> should be r2 yolo layout
    assert reg.get_weights(model_name).read_text() == "r2 yolo"
    
    # 2. Add default weights -> should prefer default.pt
    ext = tmp_path / "ext.pt"
    ext.write_text("default weights")
    reg.set_default_weights(model_name, ext)
    assert reg.get_weights(model_name).read_text() == "default weights"
    
    # 3. Specify run explicitly (1-indexed)
    assert reg.get_weights(model_name, run=1).read_text() == "r1 flat"
    assert reg.get_weights(model_name, run=2).read_text() == "r2 yolo"
    
    # 4. Custom filename flat
    (r1 / "last.pt").write_text("r1 last")
    assert reg.get_weights(model_name, run=1, filename="last.pt").read_text() == "r1 last"
    
    # 5. Custom filename yolo
    (r2 / "weights" / "last.pt").write_text("r2 last")
    assert reg.get_weights(model_name, run=2, filename="last.pt").read_text() == "r2 last"
    
    # 6. Errors
    with pytest.raises(FileNotFoundError, match="Run 3 not found"):
        reg.get_weights(model_name, run=3)
    
    with pytest.raises(FileNotFoundError, match="Weights file 'missing.pt' not found"):
        reg.get_weights(model_name, run=1, filename="missing.pt")

def test_ml_storage_initialization(tmp_path):
    storage = MLStorage(tmp_path / "ml_storage")
    
    assert storage.root == (tmp_path / "ml_storage").absolute()
    assert storage.root.exists()
    
    assert isinstance(storage.datasets, DatasetManager)
    assert storage.datasets.root == storage.root / "datasets"
    
    assert isinstance(storage.detection_models, ModelRegistry)
    assert storage.detection_models.root == storage.root / "model_registry" / "detection"
    
    assert isinstance(storage.classifier_models, ModelRegistry)
    assert storage.classifier_models.root == storage.root / "model_registry" / "classifier"
    
    assert repr(storage).startswith("MLStorage(root=")
