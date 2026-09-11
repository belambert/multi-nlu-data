# {name}

{title} with per-intent segmentation, as character spans.

The source corpus marks slots with BIO tags and lists the intents of an
utterance as an unordered set, but does not say which span of the utterance
belongs to which intent. This dataset adds that missing structure: each
utterance is segmented into contiguous per-intent spans, given as character
offsets into `text`, with slot spans nested under the intent that owns them.

```json
{{
  "text": "book a diner for 1 in green isle and then will it be warmer at 15 o clock",
  "intents": [
    {{"intent": "BookRestaurant", "start": 0, "end": 32,
     "slots": [{{"name": "restaurant_type", "start": 7, "end": 12}},
               {{"name": "party_size_number", "start": 17, "end": 18}},
               {{"name": "city", "start": 22, "end": 32}}]}},
    {{"intent": "GetWeather", "start": 42, "end": 73,
     "slots": [{{"name": "condition_temperature", "start": 53, "end": 59}},
               {{"name": "timeRange", "start": 63, "end": 73}}]}}
  ],
  "source": "heuristic"
}}
```

Offsets count Unicode codepoints into `text` — Python characters, not UTF-8
bytes or UTF-16 units — and are half-open, so `text[start:end]` is the span.
Connective tokens ("and then") sit between segments, covered by no intent span.

## Fields

| Field     | Description                                                          |
| --------- | -------------------------------------------------------------------- |
| `text`    | The utterance, whitespace-joined source tokens                        |
| `intents` | Intent spans over `text`, each with the slot spans it contains        |
| `source`  | How the segmentation was derived: `single`, `heuristic` or `llm`      |

## How The Segmentation Was Derived

The boundaries are **reconstructed, not gold** — the source corpus does not
record them. `source` says how each row was settled, in decreasing order of
confidence:

- `single` — a single-intent utterance, so the whole span is unambiguous.
- `heuristic` — slot types that pin down one intent fix the order, and the
  strongest connective in the gap between them places the seam.
- `llm` — the heuristics were not decisive, so a model chose the intent order
  and seam positions. Every answer was validated against the BIO spans before
  being accepted, but these rows are the least corroborated.

Treat the segmentation as approximate and spot-check before relying on it.
Where the two methods overlap they agree closely, but there is no gold standard
to measure either against.

## Provenance

{provenance}

Produced by [multi-nlu-data](https://github.com/belambert/multi-nlu-data) at
revision `{rev}`.
