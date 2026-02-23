from __future__ import annotations

from llm_trainer.dataloader import SequenceDataLoader, load_token_ids, tokenize_wikitext2


def test_tokenize_wikitext2_persists_tokenizer_and_ids(tmp_path) -> None:
    raw_dir = tmp_path / "wikitext-2" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "train.txt").write_text("hello world\nhello again\n", encoding="utf-8")
    (raw_dir / "validation.txt").write_text("world again\n", encoding="utf-8")

    result = tokenize_wikitext2(data_root=tmp_path, seed=7)

    assert result.tokenizer_path.exists()
    assert result.train_ids_path.exists()
    assert result.validation_ids_path.exists()
    assert result.train_tokens > 0
    assert result.validation_tokens > 0


def test_sequence_dataloader_shapes_and_seed_determinism() -> None:
    token_ids = list(range(200))
    loader_a = SequenceDataLoader(token_ids, seq_length=8, batch_size=4, shuffle=True, seed=123)
    loader_b = SequenceDataLoader(token_ids, seq_length=8, batch_size=4, shuffle=True, seed=123)

    batch_a = next(iter(loader_a))
    batch_b = next(iter(loader_b))

    assert batch_a.shape == (4, 8)
    assert batch_b.shape == (4, 8)
    assert batch_a.input_ids == batch_b.input_ids
    assert batch_a.labels == batch_b.labels


def test_load_token_ids_roundtrip(tmp_path) -> None:
    path = tmp_path / "ids.json"
    path.write_text("[1,2,3]", encoding="utf-8")
    assert load_token_ids(path) == [1, 2, 3]
