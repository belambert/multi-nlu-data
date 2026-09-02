"""CLI for publishing converted splits to the Hugging Face Hub.

Pushing to the same repo id updates that dataset in place: the Hub keeps every
push as a commit, so each run adds a revision rather than creating a new
dataset. Commits record the source revision of this repo, so a dataset revision
can be traced back to the code that produced it.
"""

import json
import subprocess
from pathlib import Path

import typer
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from huggingface_hub import DatasetCard, whoami

from multi_nlu_data.data import Dataset, Split

app = typer.Typer(help="Publish converted splits to the Hugging Face Hub.")

CARD = """\
# {name}

{title} with per-intent segmentation, as nested XML.

The source corpus marks slots with BIO tags and lists the intents of an
utterance as an unordered set, but does not say which span of the utterance
belongs to which intent. This dataset adds that missing structure: each
utterance is segmented into contiguous per-intent spans and rendered as XML,
with slots nested inside the intent that owns them.

```xml
<BookRestaurant>book a <restaurant_type>diner</restaurant_type> for \
<party_size_number>1</party_size_number> in <city>green isle</city></BookRestaurant>
and then
<GetWeather>will it be <condition_temperature>warmer</condition_temperature> at \
<timeRange>15 o clock</timeRange></GetWeather>
```

Connective tokens ("and then") sit between segments, outside both tags.

## Fields

| Field    | Description                                                    |
| -------- | -------------------------------------------------------------- |
| `text`   | The utterance, whitespace-joined source tokens                  |
| `xml`    | The same utterance with nested intent and slot tags             |
| `source` | How the segmentation was derived: `single`, `heuristic` or `llm` |

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
"""

TITLES = {
    Dataset.MIXSNIPS: "[MixSNIPS](https://huggingface.co/datasets/nahyeon00/mixsnips_clean)",
    Dataset.MIXATIS: "[MixATIS](https://huggingface.co/datasets/gamy0315/mixatis_clean)",
}

PROVENANCE = {
    Dataset.MIXSNIPS: """\
Utterances come from the SNIPS benchmark, released as CC0-1.0 in
[snipsco/nlu-benchmark](https://github.com/snipsco/nlu-benchmark), composed into
multi-intent utterances by [AGIF](https://github.com/LooperXX/AGIF). Please cite
the [Snips](https://arxiv.org/abs/1805.10190) and
[AGIF](https://arxiv.org/abs/2004.10087) papers. The segmentation added here was
produced with an open model served by Fireworks.""",
    Dataset.MIXATIS: """\
Utterances derive from ATIS, which is distributed by the Linguistic Data
Consortium under licence and is **not** public domain, composed into
multi-intent utterances by [AGIF](https://github.com/LooperXX/AGIF). Confirm your
ATIS licence position before redistributing this data. Please cite the
[AGIF](https://arxiv.org/abs/2004.10087) paper.""",
}


@app.command()
def push(
    dataset: Dataset = typer.Option(
        Dataset.MIXSNIPS, help="Which corpus was converted."
    ),
    data: Path = typer.Option(
        Path("data"), help="Directory holding the converted JSONL."
    ),
    repo: str = typer.Option(
        None, help="Target repo id; defaults to <user>/<dataset>-xml."
    ),
    private: bool = typer.Option(
        True, help="Create the dataset private (first push only)."
    ),
    license: str = typer.Option("cc-by-4.0", help="License tag for the dataset card."),
    message: str = typer.Option(None, help="Commit message for this revision."),
    dry_run: bool = typer.Option(False, help="Report what would be pushed, then stop."),
) -> None:
    """Push the converted splits, adding a revision to an existing dataset."""
    splits = read_splits(data)
    repo = repo or f"{whoami()['name']}/{dataset}-xml"
    rev = source_revision()
    message = message or f"Update from multi-nlu-data@{rev}"

    for name, rows in splits.items():
        typer.echo(f"  {name:<12} {len(rows):>6} rows", err=True)
    typer.echo(f"-> {repo}  ({message})", err=True)
    if dry_run:
        return

    DatasetDict(
        {name: HFDataset.from_list(rows) for name, rows in splits.items()}
    ).push_to_hub(repo, private=private, commit_message=message)
    push_card(repo, dataset, license, rev, message)

    typer.secho(
        f"published https://huggingface.co/datasets/{repo}", fg="green", err=True
    )


def read_splits(data: Path) -> dict[str, list[dict]]:
    """Load whichever converted splits are present, failing if none are."""
    splits = {}
    for split in Split:
        path = data / f"{split}.jsonl"
        if path.exists():
            splits[str(split)] = [json.loads(line) for line in path.open()]
    if not splits:
        raise typer.BadParameter(f"no {'/'.join(Split)} .jsonl files under {data}")
    return splits


def push_card(
    repo: str, dataset: Dataset, license: str, rev: str, message: str
) -> None:
    """Replace the card prose, keeping the split metadata push_to_hub just wrote."""
    card = DatasetCard.load(repo)
    card.text = CARD.format(
        name=repo.split("/")[-1],
        title=TITLES[dataset],
        provenance=PROVENANCE[dataset],
        rev=rev,
    )
    card.data.license = license
    card.data.language = ["en"]
    card.data.task_categories = ["token-classification", "text2text-generation"]
    card.data.tags = ["multi-intent", "nlu", "slot-filling", "intent-detection"]
    card.push_to_hub(repo, commit_message=f"{message} (card)")


def source_revision() -> str:
    """Short git revision of this repo, so a dataset revision maps back to code."""
    rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    )
    if rev.returncode != 0:
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


if __name__ == "__main__":
    app()
