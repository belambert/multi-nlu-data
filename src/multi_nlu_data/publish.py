"""CLI for publishing converted splits to the Hugging Face Hub.

Pushing to the same repo id updates that dataset in place: the Hub keeps every
push as a commit, so each run adds a revision rather than creating a new
dataset. Commits record the source revision of this repo, so a dataset revision
can be traced back to the code that produced it.
"""

import json
import subprocess
from importlib.resources import files
from pathlib import Path

import typer
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from huggingface_hub import DatasetCard, whoami

from multi_nlu_data.data import Dataset, Split

app = typer.Typer(help="Publish converted splits to the Hugging Face Hub.")

CARDS = files("multi_nlu_data") / "cards"

# one-line links rather than prose, so they stay next to the code that uses them
TITLES = {
    Dataset.MIXSNIPS: "[MixSNIPS](https://huggingface.co/datasets/nahyeon00/mixsnips_clean)",
    Dataset.MIXATIS: "[MixATIS](https://huggingface.co/datasets/gamy0315/mixatis_clean)",
}


@app.command()
def push(
    dataset: Dataset = typer.Option(
        Dataset.MIXSNIPS, help="Which corpus was converted."
    ),
    data: Path = typer.Option(
        None, help="Converted JSONL directory; defaults to data/<dataset>."
    ),
    repo: str = typer.Option(
        None, help="Target repo id; defaults to <user>/<dataset>-intent-spans."
    ),
    private: bool = typer.Option(
        True, help="Create the dataset private (first push only)."
    ),
    license: str = typer.Option("cc-by-4.0", help="License tag for the dataset card."),
    message: str = typer.Option(None, help="Commit message for this revision."),
    card_only: bool = typer.Option(
        False, help="Update the card, leaving the data alone."
    ),
    dry_run: bool = typer.Option(False, help="Report what would be pushed, then stop."),
) -> None:
    """Push the converted splits, adding a revision to an existing dataset."""
    # tie the default to --dataset, so one corpus cannot be published as another
    data = data or Path("data") / dataset
    splits = {} if card_only else read_splits(data)
    repo = repo or f"{whoami()['name']}/{dataset}-intent-spans"
    rev = source_revision()
    message = message or f"Update from multi-nlu-data@{rev}"

    if not card_only:
        typer.echo(f"{data}/", err=True)
    for name, rows in splits.items():
        typer.echo(f"  {name:<12} {len(rows):>6} rows", err=True)
    typer.echo(f"-> {repo}  ({message}){' [card only]' if card_only else ''}", err=True)
    if dry_run:
        return

    if splits:
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
    card.text = render_card(dataset, repo.split("/")[-1], rev)
    card.data.license = license
    card.data.language = ["en"]
    # both must come from the Hub's official list, or the card fails validation
    card.data.task_categories = ["token-classification", "text-generation"]
    card.data.tags = ["multi-intent", "nlu", "slot-filling", "intent-detection"]
    card.push_to_hub(repo, commit_message=f"{message} (card)")


def render_card(dataset: Dataset, name: str, rev: str) -> str:
    """Fill cards/body.md with the provenance for this dataset."""
    return (
        (CARDS / "body.md")
        .read_text()
        .format(
            name=name,
            title=TITLES[dataset],
            provenance=(CARDS / f"{dataset}.md").read_text().strip(),
            rev=rev,
        )
    )


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
