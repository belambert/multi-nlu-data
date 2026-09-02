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

## Setup

    uv sync

## Usage

Print examples from a split:

    uv run multi-nlu show --split test --n 5

Sample randomly instead of taking the first `n`:

    uv run multi-nlu show --seed 0

Show intent counts and how many intents appear per utterance:

    uv run multi-nlu stats --split train

## Development

Format and check:

    uv run black .
    uv run isort .
    uv run pytest
