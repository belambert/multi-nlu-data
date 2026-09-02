# {name}

{title} with per-intent segmentation, as nested XML.

The source corpus marks slots with BIO tags and lists the intents of an
utterance as an unordered set, but does not say which span of the utterance
belongs to which intent. This dataset adds that missing structure: each
utterance is segmented into contiguous per-intent spans and rendered as XML,
with slots nested inside the intent that owns them.

```xml
<BookRestaurant>book a <restaurant_type>diner</restaurant_type> for <party_size_number>1</party_size_number> in <city>green isle</city></BookRestaurant>
and then
<GetWeather>will it be <condition_temperature>warmer</condition_temperature> at <timeRange>15 o clock</timeRange></GetWeather>
```

Connective tokens ("and then") sit between segments, outside both tags.

## Fields

| Field    | Description                                                      |
| -------- | ---------------------------------------------------------------- |
| `text`   | The utterance, whitespace-joined source tokens                    |
| `xml`    | The same utterance with nested intent and slot tags               |
| `source` | How the segmentation was derived: `single`, `heuristic` or `llm`  |

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
