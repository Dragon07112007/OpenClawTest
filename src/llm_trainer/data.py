from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class DatasetPrepResult:
    dataset_name: str
    data_dir: Path
    train_path: Path
    validation_path: Path
    train_samples: int
    validation_samples: int


def _non_empty_lines(rows: list[str]) -> list[str]:
    return [row.strip() for row in rows if row.strip()]


def prepare_wikitext2(
    *,
    data_root: str | Path,
    loader: Callable[..., dict[str, list[dict[str, str]]]] | None = None,
) -> DatasetPrepResult:
    """Download/load WikiText-2 and persist local train/validation artifacts."""
    data_root_path = Path(data_root)
    dataset_dir = data_root_path / "wikitext-2"
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if loader is None:
        from datasets import load_dataset

        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            cache_dir=str(dataset_dir / "cache"),
        )
    else:
        dataset = loader()

    train_rows = _non_empty_lines([row["text"] for row in dataset["train"]])
    validation_rows = _non_empty_lines([row["text"] for row in dataset["validation"]])

    if not train_rows:
        raise ValueError("WikiText-2 train split is empty after filtering.")
    if not validation_rows:
        raise ValueError("WikiText-2 validation split is empty after filtering.")

    train_path = raw_dir / "train.txt"
    validation_path = raw_dir / "validation.txt"
    train_path.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    validation_path.write_text("\n".join(validation_rows) + "\n", encoding="utf-8")

    return DatasetPrepResult(
        dataset_name="wikitext-2",
        data_dir=dataset_dir,
        train_path=train_path,
        validation_path=validation_path,
        train_samples=len(train_rows),
        validation_samples=len(validation_rows),
    )
