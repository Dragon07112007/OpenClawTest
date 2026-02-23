from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)


def test_model_forward_shapes_and_loss() -> None:
    torch = importlib.import_module("torch")
    GPTLanguageModel = importlib.import_module("llm_trainer.model").GPTLanguageModel

    model = GPTLanguageModel(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_length=16,
    )
    input_ids = torch.randint(0, 64, (3, 12), dtype=torch.long)
    labels = torch.randint(0, 64, (3, 12), dtype=torch.long)

    logits, loss = model(input_ids, labels)

    assert logits.shape == (3, 12, 64)
    assert loss is not None
    assert loss.ndim == 0


def test_single_train_step_executes() -> None:
    torch = importlib.import_module("torch")
    GPTLanguageModel = importlib.import_module("llm_trainer.model").GPTLanguageModel

    model = GPTLanguageModel(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        max_seq_length=8,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
    labels = torch.randint(0, 32, (2, 8), dtype=torch.long)

    optimizer.zero_grad(set_to_none=True)
    _, loss = model(input_ids, labels)
    assert loss is not None
    loss.backward()
    optimizer.step()
