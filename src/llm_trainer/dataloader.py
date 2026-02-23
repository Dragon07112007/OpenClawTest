from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .tokenization import BasicTokenizer, build_basic_tokenizer


@dataclass(frozen=True)
class TokenizedData:
    tokenizer_path: Path
    train_ids_path: Path
    validation_ids_path: Path
    train_tokens: int
    validation_tokens: int


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize_wikitext2(
    *,
    data_root: str | Path = "data",
    seed: int = 42,
) -> TokenizedData:
    dataset_root = Path(data_root) / "wikitext-2"
    raw_root = dataset_root / "raw"
    tokenized_root = dataset_root / "tokenized"
    tokenizer_path = tokenized_root / "tokenizer.json"
    train_ids_path = tokenized_root / "train_ids.json"
    validation_ids_path = tokenized_root / "validation_ids.json"

    train_lines = _read_lines(raw_root / "train.txt")
    validation_lines = _read_lines(raw_root / "validation.txt")
    if not train_lines or not validation_lines:
        raise ValueError("Raw dataset files are empty. Run dataset preparation first.")

    tokenizer = build_basic_tokenizer(
        texts=train_lines + validation_lines,
        tokenizer_path=tokenizer_path,
        seed=seed,
    )

    train_ids = [token_id for line in train_lines for token_id in tokenizer.encode(line)]
    validation_ids = [token_id for line in validation_lines for token_id in tokenizer.encode(line)]
    if not train_ids or not validation_ids:
        raise ValueError("Tokenized dataset is empty.")

    tokenized_root.mkdir(parents=True, exist_ok=True)
    train_ids_path.write_text(json.dumps(train_ids), encoding="utf-8")
    validation_ids_path.write_text(json.dumps(validation_ids), encoding="utf-8")

    return TokenizedData(
        tokenizer_path=tokenizer_path,
        train_ids_path=train_ids_path,
        validation_ids_path=validation_ids_path,
        train_tokens=len(train_ids),
        validation_tokens=len(validation_ids),
    )


@dataclass
class Batch:
    input_ids: list[list[int]]
    labels: list[list[int]]

    @property
    def shape(self) -> tuple[int, int]:
        if not self.input_ids:
            return (0, 0)
        return (len(self.input_ids), len(self.input_ids[0]))


class SequenceDataLoader:
    def __init__(
        self,
        token_ids: list[int],
        *,
        seq_length: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if seq_length <= 0 or batch_size <= 0:
            raise ValueError("seq_length and batch_size must be > 0")
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        segment = seq_length + 1
        total_segments = len(token_ids) // segment
        self._samples = [
            token_ids[i * segment : (i + 1) * segment] for i in range(total_segments) if segment > 1
        ]

    def __iter__(self) -> Iterator[Batch]:
        indices = list(range(len(self._samples)))
        if self.shuffle:
            random.Random(self.seed).shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if len(batch_indices) < self.batch_size:
                continue
            input_ids = [self._samples[idx][:-1] for idx in batch_indices]
            labels = [self._samples[idx][1:] for idx in batch_indices]
            yield Batch(input_ids=input_ids, labels=labels)


def load_token_ids(path: str | Path) -> list[int]:
    return [int(v) for v in json.loads(Path(path).read_text(encoding="utf-8"))]


def load_tokenizer(path: str | Path) -> BasicTokenizer:
    return BasicTokenizer.load(path)

