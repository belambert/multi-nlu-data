# multi-nlu

Multi-intent natural language understanding with open language models, using the
[MixSNIPS](https://huggingface.co/datasets/nahyeon00/mixsnips_clean) dataset.

MixSNIPS composes SNIPS utterances into single utterances carrying one to three
intents (e.g. "play anything by george formby jr, give zero points out of 6 to
the devil in a forest and also can i get the showtimes for the jade faced
assassin cinema"). Each example has BIO slot tags alongside its intent set.

| Split      | Examples |
| ---------- | -------- |
| train      | 39,776   |
| validation | 2,198    |
| test       | 2,199    |

## Native Dataset Format

Useful to know before converting the data into a modeling format. Each row has
exactly three fields, all lists of strings — there are no `ClassLabel` features
or label-id mappings, so nothing needs a vocabulary file to interpret.

```python
{
    "token":  ["play", "isham", "jones", "and", "swine", "not", "deserves", "four", "points"],
    "tag":    ["O", "B-artist", "I-artist", "O", "B-object_name", "I-object_name", "O", "B-rating_value", "B-rating_unit"],
    "intent": ["PlayMusic#RateBook"],
}
```

| Field    | Type           | Notes                                                       |
| -------- | -------------- | ----------------------------------------------------------- |
| `token`  | `list[str]`    | Word-level tokens, one per position; 11,409 distinct in train |
| `tag`    | `list[str]`    | BIO slot tags, aligned 1:1 with `token`; 39 slot types       |
| `intent` | `list[str]`    | Always one element: the intents joined by `#`                |

### Tokens

Whitespace-level natural words, not subwords — there are no `##`, `Ġ`, or `▁`
markers and no token contains a space. Everything is lowercased. Punctuation is
split inconsistently: `,` `&` `/` `-` appear as standalone tokens, but some
tokens keep punctuation attached (`tsuihou:`, `philosopher's`) and hyphenated
entity names stay whole (`maliau-basin-conservation-area`). Utterances average
19.7 tokens and range from 2 to 54. The train vocabulary contains one mojibake
token (`������`) worth normalizing away before exact-match scoring.

Because the tags are aligned to *these* word tokens, an encoder tagger needs the
usual word-to-subword alignment: tokenize with `is_split_into_words=True`, then
project each word's tag onto its first subword via `word_ids()` and mask the
rest with `-100`.

### Tags

Standard BIO. Verified across all 39,776 train rows: `tag` is the same length as
`token` in every row, only `O`/`B-`/`I-` prefixes occur, and every `I-x` follows
a `B-x` or `I-x` of the same type — there are no orphan `I-` tags, so a
single-pass span decoder is safe. `Example.slots` in `data.py` does this decoding.

Note that spans are **not** attributed to a particular intent. A three-intent
utterance yields one flat span list, and recovering which intent owns which slot
requires inferring it from the slot type or token position.

### Intents

Seven intents, inherited from SNIPS: `AddToPlaylist`, `BookRestaurant`,
`GetWeather`, `PlayMusic`, `RateBook`, `SearchCreativeWork`,
`SearchScreeningEvent`. One to three per utterance (450 / 1,249 / 500 of the
2,199 test rows).

The `#`-joined list is alphabetically sorted in all 39,776 train rows, so its
order says nothing about the order the intents appear in the utterance. Treat
intents as a **set**: score with set-based exact match, and if a generative model
is asked to emit them, either sort its output before comparing or compare as
sets. The same intent never repeats within a row.

## Setup

    uv sync

## Usage

Print examples from a split:

    uv run multi-nlu show --split test --n 5

Sample randomly instead of taking the first `n`:

    uv run multi-nlu show --seed 0

Show intent counts and how many intents appear per utterance:

    uv run multi-nlu stats --split train

## Converting To XML

`multi-nlu-convert` recovers per-intent segmentation and emits nested XML:

    <PlayMusic>play <artist>isham jones</artist></PlayMusic>
    and
    <RateBook><object_name>swine not</object_name> deserves
      <rating_value>four</rating_value> <rating_unit>points</rating_unit></RateBook>

Connective tokens sit between segments, outside both tags, so the segments stay
a clean partition of the content.

### How Segmentation Is Recovered

The dataset has no intent spans (see above), so they are approximated from two
free signals:

1.  **Slot-to-intent anchors.** Single-intent rows label their own slots
    unambiguously, giving a slot type to intent map for free. Of 39 slot types,
    28 are globally unambiguous — and the other 11 usually resolve *within a
    row*, since only that row's intents are candidates (`artist` is
    PlayMusic-or-AddToPlaylist in general, but forced in a
    `{PlayMusic, GetWeather}` row).
2.  **Connectives.** The seam is marked by the phrase used to join the source
    utterances, ranked strongest-first: `and also`/`and then` beat `also`/`then`,
    which beat `,`, which beats a bare `and`. The ranking matters — bare `and`
    also occurs inside entity names, and using it indiscriminately drops
    resolution from 88% to 43%.

Anchors give the intent order and bracket where each seam must lie; the
strongest connective in that gap places it. This settles **88.0% of train**
(8,277 single-intent rows plus 26,713 multi-intent rows), leaving 4,786 for the
LLM. Reassuringly, anchors never interleave in any of the 31,499 multi-intent
train rows, which is what you would expect if the concatenation structure holds.

### LLM Fallback

Unresolved rows go to an open model on Fireworks. The model is asked only for
what the heuristics could not pin down — the intent order and the seam
positions, as token indices rather than free text — so every answer is validated
against the BIO spans (boundaries in range, strictly increasing, never inside a
slot span, intents a permutation of the row's set) before being accepted.
Rejected answers are reported, never silently patched.

    export FIREWORKS_API_KEY=...
    uv run multi-nlu-convert models --filter llama     # pick a real model id
    uv run multi-nlu-convert xml --split test --llm --model <id> --out test.jsonl

Answers are cached to `llm-cache.jsonl` and reused, so reruns cost nothing.
`--limit` caps the number of calls for a trial run.

Heuristic coverage on a split, without spending anything:

    uv run multi-nlu-convert coverage --split train

### Caveat

The segmentation is **approximate and unvalidated**. MixSNIPS ships no gold
boundaries, so the 88% figure is the rate at which the heuristics reach a
confident answer, not a measured accuracy. Spot-check a sample before treating
the output as training data.

## Development

Format and check:

    uv run black .
    uv run isort .
    uv run pytest
