"""LLM fallback for the utterances the segmentation heuristics cannot settle.

The model is asked only for the two things the heuristics failed to pin down —
the order the intents appear in and where the seams fall — as token indices
rather than free text, so every answer can be validated against the BIO spans
before it is accepted.
"""

import json
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fireworks import Fireworks

from multi_nlu_data.data import Example
from multi_nlu_data.segment import TIERS, Segment, Segmentation, spans

# list alternatives with `multi-nlu-convert models`
DEFAULT_MODEL = "accounts/fireworks/models/kimi-k3"

SYSTEM = """\
You segment multi-intent utterances. The utterance is a concatenation of \
several single-intent requests, in some order. Given the numbered tokens and \
the set of intents present, return the order the intents appear in and the \
token index where each later segment begins. Boundaries must be strictly \
increasing, and must never fall inside a marked slot. Answer with JSON only."""

SCHEMA = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "order": {"type": "array", "items": {"type": "string"}},
            "boundaries": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["order", "boundaries"],
    },
}


def label_all(
    examples: list[Example],
    model: str = DEFAULT_MODEL,
    cache: Path | None = None,
    workers: int = 8,
    on_done: Callable[[], None] | None = None,
) -> tuple[list[Segmentation], list[Example]]:
    """Label examples with the LLM, reusing any cached answers; returns (done, failed).

    `on_done` fires once per finished example, for progress reporting.
    """
    client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])
    seen = read_cache(cache) if cache else {}

    def run(ex: Example) -> Segmentation | None:
        answer = seen.get(ex.text) or ask(client, ex, model)
        seg = build(ex, answer) if answer else None
        if seg and cache and ex.text not in seen:
            append_cache(cache, ex.text, answer)
        return seg

    results: list[Segmentation | None] = [None] * len(examples)
    with ThreadPoolExecutor(workers) as pool:
        pending = {pool.submit(run, ex): i for i, ex in enumerate(examples)}
        for future in as_completed(pending):
            results[pending[future]] = future.result()
            if on_done:
                on_done()

    done = [s for s in results if s]
    failed = [ex for ex, s in zip(examples, results) if not s]
    return done, failed


def ask(client: Fireworks, ex: Example, model: str) -> dict | None:
    """Ask the model to order the intents and place the seams."""
    reply = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt(ex)},
        ],
        response_format=SCHEMA,
        temperature=0,
        # not every model honours this; on one that does not, thinking eats the
        # completion budget and the answer comes back empty
        reasoning_effort="none",
        max_tokens=200,
        prompt_cache_key="mixsnips-segment",
    )
    content = reply.choices[0].message.content
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def prompt(ex: Example) -> str:
    """Numbered tokens, marked slots, and the intents to place."""
    marks = {st: (en, name) for st, en, name in spans(ex)}
    lines, i = [], 0
    while i < len(ex.tokens):
        if i in marks:
            end, name = marks[i]
            lines.append(f"{i}: {' '.join(ex.tokens[i:end])}  [{name}]")
            i = end
        else:
            lines.append(f"{i}: {ex.tokens[i]}")
            i += 1
    return (
        "\n".join(lines)
        + f"\n\nintents present ({len(ex.intents)}): {', '.join(ex.intents)}\n"
        f"Return {len(ex.intents) - 1} boundaries."
    )


def build(ex: Example, answer: dict) -> Segmentation | None:
    """Turn an answer into a Segmentation, or None if it violates the constraints."""
    order, bounds = answer.get("order"), answer.get("boundaries")
    if (
        sorted(order or []) != sorted(ex.intents)
        or len(bounds or []) != len(ex.intents) - 1
    ):
        return None
    if bounds != sorted(set(bounds)) or not all(0 < b < len(ex.tokens) for b in bounds):
        return None
    if any(st < b < en for b in bounds for st, en, _ in spans(ex)):
        return None  # a boundary inside a slot span

    segments, start = [], 0
    for cut, intent in zip(bounds, order):
        segments.append(Segment(intent, start, cut))
        start = cut
    segments.append(Segment(order[-1], start, len(ex.tokens)))
    return Segmentation(ex, [trim(ex, s) for s in segments], "llm")


def trim(ex: Example, seg: Segment) -> Segment:
    """Push connectives out of both ends so they sit between segments, as heuristics do.

    The model may cut anywhere inside a multi-word connective ("and | then"),
    which would otherwise strand half of it in the preceding segment.
    """
    for tier in TIERS:
        for c in tier:
            words = c.split()
            n = len(words)
            if (
                ex.tokens[seg.start : seg.start + n] == words
                and seg.start + n < seg.end
            ):
                return trim(ex, Segment(seg.intent, seg.start + n, seg.end))
            if ex.tokens[seg.end - n : seg.end] == words and seg.start < seg.end - n:
                return trim(ex, Segment(seg.intent, seg.start, seg.end - n))
    return seg


def read_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {r["text"]: r["answer"] for r in map(json.loads, f)}


_cache_lock = threading.Lock()


def append_cache(path: Path, text: str, answer: dict) -> None:
    with _cache_lock, path.open("a") as f:
        f.write(json.dumps({"text": text, "answer": answer}) + "\n")
