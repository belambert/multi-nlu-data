"""Approximate per-intent segmentation of Mix* utterances.

MixSNIPS and MixATIS were built by concatenating single-intent utterances but do
not preserve the boundaries, so intent spans have to be recovered. Two free
signals carry most of the work: single-intent rows label their own slots
unambiguously, and the concatenation connectives ("and also", "and then") mark
the seams. Neither signal is dataset-specific — the slot map is learned from
whichever corpus is loaded. Rows the heuristics cannot settle go to an LLM.
"""

from dataclasses import dataclass
from xml.sax.saxutils import escape

from multi_nlu_data.data import Dataset, Example, Split, load

# boundary markers, strongest first; the strongest tier that matches a gap wins
TIERS = [
    ["and also", "and then", "as well as", "and after that"],
    ["also", "then", "next"],
    [","],
    ["and"],
]


@dataclass
class Segment:
    """A token range [start, end) carrying one intent."""

    intent: str
    start: int
    end: int


@dataclass
class Segmentation:
    example: Example
    segments: list[Segment]
    source: str  # "single", "heuristic", or "llm"


def segment_split(
    split: str = Split.TRAIN, dataset: str = Dataset.MIXSNIPS
) -> tuple[list[Segmentation], list[Example]]:
    """Segment a split heuristically, returning what resolved and what needs an LLM."""
    examples = load(split, dataset)
    slot_intents = learn_slot_intents(examples)

    done, todo = [], []
    for ex in examples:
        seg = segment(ex, slot_intents)
        (done if seg else todo).append(seg or ex)
    return done, todo


def segment(ex: Example, slot_intents: dict[str, set[str]]) -> Segmentation | None:
    """Segment one utterance, or None if the heuristics are not decisive."""
    if len(ex.intents) == 1:
        return Segmentation(ex, [Segment(ex.intents[0], 0, len(ex.tokens))], "single")

    blocks = anchor_blocks(ex, slot_intents)
    order = [b[2] for b in blocks]
    if len(set(order)) != len(order) or set(order) != set(ex.intents):
        return None  # anchors contradict, or some intent has no anchor at all

    bounds = []
    for a, b in zip(blocks, blocks[1:]):
        cuts = cuts_in_gap(ex, a[1], b[0])
        if len(cuts) != 1:
            return None
        bounds.append(cuts[0])

    segments, start = [], 0
    for (cut, width), intent in zip(bounds, order):
        segments.append(Segment(intent, start, cut))
        start = cut + width
    segments.append(Segment(order[-1], start, len(ex.tokens)))
    return Segmentation(ex, segments, "heuristic")


def learn_slot_intents(examples: list[Example]) -> dict[str, set[str]]:
    """Map each slot type to the intents it occurs with, learned from single-intent rows."""
    out: dict[str, set[str]] = {}
    for ex in examples:
        if len(ex.intents) == 1:
            for s in ex.slots:
                out.setdefault(s.name, set()).add(ex.intents[0])
    return out


def anchor_blocks(
    ex: Example, slot_intents: dict[str, set[str]]
) -> list[tuple[int, int, str]]:
    """Runs of consecutive slot spans that each pin down a single intent.

    A slot type ambiguous in general is often unambiguous within a row, since
    only the row's own intents are candidates.
    """
    intents = set(ex.intents)
    blocks: list[list] = []
    for st, en, name in spans(ex):
        c = slot_intents.get(name, set()) & intents
        if len(c) != 1:
            continue
        intent = next(iter(c))
        if blocks and blocks[-1][2] == intent:
            blocks[-1][1] = en
        else:
            blocks.append([st, en, intent])
    return [tuple(b) for b in blocks]


def cuts_in_gap(ex: Example, lo: int, hi: int) -> list[tuple[int, int]]:
    """Boundary candidates (position, width) in [lo, hi] from the strongest matching tier."""
    for tier in TIERS:
        found = [
            (i, len(c.split()))
            for i in range(lo, hi + 1)
            for c in tier
            if ex.tokens[i : i + len(c.split())] == c.split()
        ]
        if found:
            return found
    return []


def spans(ex: Example) -> list[tuple[int, int, str]]:
    """BIO spans as (start, end, slot name)."""
    out: list[list] = []
    for i, tag in enumerate(ex.tags):
        if tag.startswith("B-"):
            out.append([i, i + 1, tag[2:]])
        elif tag.startswith("I-") and out:
            out[-1][1] = i + 1
    return [tuple(s) for s in out]


def to_xml(seg: Segmentation) -> str:
    """Render as nested intent/slot XML; connective tokens sit between segments."""
    slot_at = {st: (en, name) for st, en, name in spans(seg.example)}
    toks = seg.example.tokens

    out, pos = [], 0
    for s in seg.segments:
        out.extend(escape(t) for t in toks[pos : s.start])  # connective
        inner, i = [], s.start
        while i < s.end:
            if i in slot_at:
                end, name = slot_at[i]
                inner.append(f"<{name}>{escape(' '.join(toks[i:end]))}</{name}>")
                i = end
            else:
                inner.append(escape(toks[i]))
                i += 1
        out.append(f"<{s.intent}>{' '.join(inner)}</{s.intent}>")
        pos = s.end
    out.extend(escape(t) for t in toks[pos:])
    return " ".join(out)
