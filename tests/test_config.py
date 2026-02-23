from llm_trainer.config import load_config


def test_load_default_config() -> None:
    config = load_config("configs/default.toml")

    assert "model" in config
    assert "training" in config
    assert "data" in config
    assert "device" in config

    assert config["training"]["batch_size"] > 0
    assert config["data"]["dataset_path"]
