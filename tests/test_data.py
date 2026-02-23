from __future__ import annotations

import pytest

from llm_trainer.data import prepare_wikitext2


def test_prepare_wikitext2_writes_non_empty_train_and_validation(tmp_path) -> None:
    def fake_loader():
        return {
            "train": [{"text": "hello"}, {"text": ""}, {"text": "world"}],
            "validation": [{"text": "validate me"}],
        }

    result = prepare_wikitext2(data_root=tmp_path, loader=fake_loader)

    assert result.dataset_name == "wikitext-2"
    assert result.train_samples == 2
    assert result.validation_samples == 1
    assert result.train_path.exists()
    assert result.validation_path.exists()
    assert result.train_path.read_text(encoding="utf-8").strip() == "hello\nworld"
    assert result.validation_path.read_text(encoding="utf-8").strip() == "validate me"


def test_prepare_wikitext2_rejects_empty_splits(tmp_path) -> None:
    def fake_loader():
        return {
            "train": [{"text": "   "}],
            "validation": [{"text": "ok"}],
        }

    with pytest.raises(ValueError, match="train split is empty"):
        prepare_wikitext2(data_root=tmp_path, loader=fake_loader)
