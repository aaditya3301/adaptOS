from pathlib import Path

import pytest
import yaml

from adaptos.config import REPO_ROOT, load_config

CONFIG_DIR = REPO_ROOT / "configs"


def test_repo_configs_are_valid():
    config = load_config(CONFIG_DIR)
    assert set(config.adapters) == {"sql", "summarize", "math"}
    assert config.backends.training == "lora"
    assert config.backends.inference == "peft"


def test_every_adapter_has_monitor_thresholds():
    config = load_config(CONFIG_DIR)
    assert set(config.monitor.thresholds) == set(config.adapters)


def _copy_configs(tmp_path: Path) -> Path:
    dst = tmp_path / "configs"
    (dst / "adapters").mkdir(parents=True)
    for src in CONFIG_DIR.rglob("*.yaml"):
        target = dst / src.relative_to(CONFIG_DIR)
        target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def test_adapter_name_must_match_file_name(tmp_path):
    cfg = _copy_configs(tmp_path)
    path = cfg / "adapters" / "sql.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["name"] = "not_sql"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="must match file name"):
        load_config(cfg)


def test_missing_monitor_threshold_is_rejected(tmp_path):
    cfg = _copy_configs(tmp_path)
    path = cfg / "monitor.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["thresholds"]["math"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="thresholds missing"):
        load_config(cfg)


def test_duplicate_registered_model_is_rejected(tmp_path):
    cfg = _copy_configs(tmp_path)
    path = cfg / "adapters" / "math.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["registered_model"] = "adaptos-adapter-sql"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="must be unique"):
        load_config(cfg)
