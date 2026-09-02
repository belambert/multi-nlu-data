"""Loading of the Mix* multi-intent NLU datasets.

MixSNIPS and MixATIS share a schema and a construction method, so both load
through the same path.
"""

from dataclasses import dataclass
from enum import StrEnum

from datasets import load_dataset


class Dataset(StrEnum):
    """Corpora we can load; also the choices Typer offers for --dataset."""

    MIXSNIPS = "mixsnips"
    MIXATIS = "mixatis"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


DATASETS = {
    Dataset.MIXSNIPS: "nahyeon00/mixsnips_clean",
    Dataset.MIXATIS: "gamy0315/mixatis_clean",
}


@dataclass
class Slot:
    name: str
    value: str


@dataclass
class Example:
    """One utterance with its intents and BIO-derived slots."""

    tokens: list[str]
    tags: list[str]
    intents: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def slots(self) -> list[Slot]:
        slots: list[Slot] = []
        for tok, tag in zip(self.tokens, self.tags):
            if tag.startswith("B-"):
                slots.append(Slot(tag[2:], tok))
            elif tag.startswith("I-") and slots:
                slots[-1].value += f" {tok}"
        return slots


def load(split: str = Split.TRAIN, dataset: str = Dataset.MIXSNIPS) -> list[Example]:
    """Load a split; intents are stored as a single '#'-joined string."""
    rows = load_dataset(DATASETS[Dataset(dataset)], split=Split(split))
    return [Example(r["token"], r["tag"], r["intent"][0].split("#")) for r in rows]
