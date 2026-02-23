from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BasicTokenizer:
    vocab: dict[str, int]

    @property
    def pad_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.vocab["<unk>"]

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(token, self.unk_id) for token in text.split()]

    def decode(self, ids: list[int]) -> str:
        inv_vocab = {idx: token for token, idx in self.vocab.items()}
        return " ".join(inv_vocab.get(idx, "<unk>") for idx in ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"vocab": self.vocab}, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BasicTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab={k: int(v) for k, v in payload["vocab"].items()})


def build_basic_tokenizer(
    *,
    texts: list[str],
    tokenizer_path: str | Path,
    seed: int = 42,
    force_rebuild: bool = False,
) -> BasicTokenizer:
    tokenizer_path = Path(tokenizer_path)
    if tokenizer_path.exists() and not force_rebuild:
        return BasicTokenizer.load(tokenizer_path)

    tokens = set()
    for text in texts:
        tokens.update(text.split())

    rng = random.Random(seed)
    sorted_tokens = sorted(tokens)
    rng.shuffle(sorted_tokens)

    vocab = {"<pad>": 0, "<unk>": 1}
    for token in sorted_tokens:
        vocab[token] = len(vocab)

    tokenizer = BasicTokenizer(vocab=vocab)
    tokenizer.save(tokenizer_path)
    return tokenizer

