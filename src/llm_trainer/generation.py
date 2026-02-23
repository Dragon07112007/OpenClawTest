from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

from .dataloader import load_tokenizer
from .model import GPTLanguageModel
from .run_metadata import load_meta


def _torch():
    return importlib.import_module("torch")


def _resolve_tokenizer_from_checkpoint(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    run_id = checkpoint_path.parent.name
    meta_path = Path("runs") / run_id / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find run metadata for checkpoint: {checkpoint_path}")
    meta = load_meta(meta_path)
    return Path(meta["tokenizer_path"])


def _sample_next_token(logits, *, temperature: float, top_k: int):
    torch = _torch()
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        min_value = values[-1]
        logits = torch.where(
            logits < min_value,
            torch.tensor(-math.inf, device=logits.device),
            logits,
        )
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def generate_from_checkpoint(
    *,
    checkpoint_path: str | Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> str:
    torch = _torch()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config: dict[str, Any] = checkpoint["config"]

    tokenizer = load_tokenizer(_resolve_tokenizer_from_checkpoint(checkpoint_path))
    model = GPTLanguageModel(
        vocab_size=len(tokenizer.vocab),
        d_model=int(config["model"]["d_model"]),
        n_heads=int(config["model"]["n_heads"]),
        n_layers=int(config["model"]["n_layers"]),
        d_ff=int(config["model"]["d_ff"]),
        max_seq_length=int(config["training"]["seq_length"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generated = tokenizer.encode(prompt)
    if not generated:
        generated = [tokenizer.unk_id]

    for _ in range(max_new_tokens):
        context = generated[-int(config["training"]["seq_length"]) :]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(input_ids)
        next_token = _sample_next_token(
            logits[0, -1],
            temperature=temperature,
            top_k=top_k,
        )
        generated.append(int(next_token))

    return tokenizer.decode(generated)
