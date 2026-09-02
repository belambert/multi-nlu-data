"""Loading of the Mix* multi-intent NLU datasets.

MixSNIPS and MixATIS share a schema and a construction method, so both load
through the same path.
"""

from dataclasses import dataclass

from datasets import load_dataset

DATASETS = {
    "mixsnips": "nahyeon00/mixsnips_clean",
    "mixatis": "gamy0315/mixatis_clean",
}
SPLITS = ("train", "validation", "test")


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


def load(split: str = "train", dataset: str = "mixsnips") -> list[Example]:
    """Load a split; intents are stored as a single '#'-joined string."""
    rows = load_dataset(DATASETS[dataset], split=split)
    return [Example(r["token"], r["tag"], r["intent"][0].split("#")) for r in rows]
