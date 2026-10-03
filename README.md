# multi-nlu-data

Data tooling for multi-intent NLU: inspect the Mix\* datasets and convert them
into per-intent character spans. Modelling lives elsewhere.

Two datasets are supported, selected with `--dataset` (default `mixsnips`):

| `--dataset` | Source                                                                               | Train  | Validation | Test  | Intents          | Slot types |
| ----------- | ------------------------------------------------------------------------------------ | ------ | ---------- | ----- | ---------------- | ---------- |
| `mixsnips`  | [nahyeon00/mixsnips_clean](https://huggingface.co/datasets/nahyeon00/mixsnips_clean) | 39,776 | 2,198      | 2,199 | 7                | 39         |
| `mixatis`   | [gamy0315/mixatis_clean](https://huggingface.co/datasets/gamy0315/mixatis_clean)     | 13,162 | 759        | 828   | 18 (17 in train) | 74         |

Both join single-intent utterances into one carrying one to three intents (e.g.
"play anything by george formby jr, give zero points out of 6 to the devil in a
forest and also can i get the showtimes for the jade faced assassin cinema"),
with BIO slot tags. They share a schema, so everything here works on either.

## Native Dataset Format

Each row has three fields, all lists of strings, with no `ClassLabel` features
or label-id mappings. Figures are for MixSNIPS; MixATIS has the same shape.

```python
{
    "token":  ["play", "isham", "jones", "and", "swine", "not", "deserves", "four", "points"],
    "tag":    ["O", "B-artist", "I-artist", "O", "B-object_name", "I-object_name", "O", "B-rating_value", "B-rating_unit"],
    "intent": ["PlayMusic#RateBook"],
}
```

| Field    | Type        | Notes                                                         |
| -------- | ----------- | ------------------------------------------------------------- |
| `token`  | `list[str]` | Word-level tokens, one per position; 11,409 distinct in train |
| `tag`    | `list[str]` | BIO slot tags, aligned 1:1 with `token`; 39 slot types        |
| `intent` | `list[str]` | Always one element: the intents joined by `#`                 |

**Tokens** are lowercased whole words, not subwords, averaging 19.7 per
utterance (range 2–54). Punctuation splitting is inconsistent (`,` stands alone
but `tsuihou:` and `maliau-basin-conservation-area` stay whole), and train has
one mojibake token (`������`) worth normalizing before exact-match scoring. An
encoder tagger needs the usual word-to-subword alignment via `word_ids()`.

**Tags** are clean BIO: lengths always match, and every `I-x` follows a `B-x` or
`I-x`, so a single-pass decoder is safe (`Example.slots` in `data.py`). Slots
are **not** attributed to an intent.

**Intents** are a sorted, `#`-joined set, so their order says nothing about
utterance order; score with set-based exact match. MixSNIPS has the seven SNIPS
intents, with 450 / 1,249 / 500 test rows carrying one / two / three. MixATIS
has 18 `atis_*` intents, unevenly split: `atis_day_name` appears in test but
**never in train**.

## Setup

    uv sync

## Usage

    uv run multi-nlu show --split test --n 5           # print examples
    uv run multi-nlu show --seed 0                     # sample randomly
    uv run multi-nlu stats --split train               # intent counts
    uv run multi-nlu show --dataset mixatis --split test

## Converting To Spans

`multi-nlu-convert` recovers per-intent segmentation as half-open character
offsets, with each intent carrying its slots:

    text     play isham jones and swine not deserves four points
    spans    [0, 16)  PlayMusic   -> artist       [5, 16)
             [21, 51) RateBook    -> object_name  [21, 30)
                                     rating_value [40, 44)
                                     rating_unit  [45, 51)

Offsets count Unicode codepoints, so `text[start:end]` works in Python.
Connectives between segments belong to no span.

### How Segmentation Is Recovered

Two signals, both learned from whichever corpus is loaded:

1.  **Slot-to-intent anchors.** Single-intent rows map slot types to intents.
    Ambiguous types often resolve within a row, since only that row's intents
    are candidates.
2.  **Connectives.** Seams are placed at the strongest joining phrase between
    anchors: `and also`/`and then` > `also`/`then` > `,` > bare `and`. The
    ranking matters, as bare `and` also occurs inside entity names.

| Train split | Single | Heuristic | Unresolved | Resolved free |
| ----------- | ------ | --------- | ---------- | ------------- |
| `mixsnips`  | 8,277  | 26,713    | 4,786      | **88.0%**     |
| `mixatis`   | 1,118  | 4,763     | 7,281      | **44.7%**     |

MixATIS resolves far less because its 74 slot types are widely shared across
intents. Check coverage for free:

    uv run multi-nlu-convert coverage --dataset mixatis --split train

### LLM Fallback

Unresolved rows go to an open model on Fireworks, which returns only the intent
order and seam positions as token indices. Answers are validated against the BIO
spans and rejected ones are reported, never patched.

    export FIREWORKS_API_KEY=...    # in ~/.zshenv or ~/.zprofile, not ~/.profile
    uv run multi-nlu-convert spans --split test --llm --out test.jsonl

Answers are cached in `llm-cache.jsonl`, keyed by utterance text alone, so
**delete the cache when changing model**. `--limit` caps calls for a trial run.

### Running The Full Conversion

    mkdir -p data/mixsnips
    uv run multi-nlu-convert spans --split train      --llm --workers 32 --out data/mixsnips/train.jsonl
    uv run multi-nlu-convert spans --split validation --llm --workers 32 --out data/mixsnips/validation.jsonl
    uv run multi-nlu-convert spans --split test       --llm --workers 32 --out data/mixsnips/test.jsonl

MixSNIPS train takes about 7 minutes at `--workers 32`; the other splits under a
minute. Give each dataset its own cache:

    mkdir -p data/mixatis
    uv run multi-nlu-convert spans --dataset mixatis --split train --llm --workers 32 \
        --cache mixatis-cache.jsonl --out data/mixatis/train.jsonl

Each line of output is one row:

```json
{
  "text": "i want to eat close to bowlegs and then show me the movie ...",
  "intents": [
    {"intent": "BookRestaurant", "start": 0, "end": 30,
     "slots": [{"name": "spatial_relation", "start": 14, "end": 19}]},
    {"intent": "SearchCreativeWork", "start": 40, "end": 61,
     "slots": [{"name": "object_type", "start": 48, "end": 53}]}
  ],
  "source": "heuristic"
}
```

`source` is `single`, `heuristic` or `llm`; `llm` rows are the least
corroborated. Rows that could not be segmented go to a `.failed.jsonl` sidecar
beside the output, so output plus sidecar should equal the split size.

### Choosing A Model

The default is Kimi K3 with thinking disabled, chosen by agreement with 40
heuristically settled rows:

| Model          | Thinking | Agrees | Completion tok |
| -------------- | -------- | ------ | -------------- |
| `kimi-k3`      | off      | 40/40  | 23             |
| `glm-5p3`      | forced   | 40/40  | 35             |
| `glm-5p2`      | off      | 39/40  | 21             |
| `kimi-k2p6`    | off      | 39/40  | 22             |
| `qwen3p8-max`  | off      | 39/40  | 22             |
| `qwen3p7-plus` | off      | 38/40  | 26             |

Accuracy differences at n=40 are noise. Whether thinking can be disabled is
per-model: on a model that ignores the flag, reasoning eats the token budget and
returns empty content, so test one call before changing `DEFAULT_MODEL`. About
half the catalogue isn't deployed serverless and returns 404.

    uv run multi-nlu-convert models --filter kimi

## Publishing To Hugging Face

    uv run multi-nlu-publish --dry-run
    uv run multi-nlu-publish

This reads `data/<dataset>/` and pushes to `<user>/<dataset>-intent-spans`
(override with `--data` and `--repo`). Re-pushing adds a revision to the same
dataset, with a commit message naming this repo's git revision.

The card is regenerated on every push, overwriting web edits. Its text lives in
`src/multi_nlu_data/cards/`; its metadata is set in `publish.py`. To update only
the card:

    uv run multi-nlu-publish --card-only

Datasets are **private** by default; see [License](#license) before passing
`--no-private`.

### Caveat

The segmentation is **approximate and unvalidated**: there are no gold
boundaries, so coverage is not accuracy. Spot-check before training on it,
especially MixATIS, where over half of train relies on the LLM.

## License

**Unsettled, and none of this is legal advice.** The card's `cc-by-4.0` default
(`--license`) is a placeholder.

| Layer                      | Source                  | License                 |
| -------------------------- | ----------------------- | ----------------------- |
| SNIPS utterances           | `sonos/nlu-benchmark`   | CC0-1.0 (public domain) |
| ATIS utterances            | LDC93S4B / LDC94S19     | copyright LDC           |
| Mix*_clean construction    | `LooperXX/AGIF`         | GPL-2.0, repo-wide      |
| The mirrors we load        | `nahyeon00`, `gamy0315` | none declared           |
| Our segmentation and spans | this repo + Kimi K3     | ours                    |

- **MixSNIPS** looks safe to publish openly: the text is CC0, the spans are ours
  under Fireworks' terms (§3.2, §7), and AGIF's GPL-2.0 cannot relicense CC0
  text. Cite the Snips and AGIF papers.
- **MixATIS**: the ATIS text is copyright LDC.

References: [nlu-benchmark](https://github.com/snipsco/nlu-benchmark),
[ATIS at LDC](https://catalog.ldc.upenn.edu/LDC93S4B),
[AGIF](https://github.com/LooperXX/AGIF),
[AGIF paper](https://arxiv.org/abs/2004.10087),
[Snips paper](https://arxiv.org/abs/1805.10190),
[Fireworks terms](https://fireworks.ai/terms-of-service).

## Development

    uv run black .
    uv run isort .
    uv run pytest
