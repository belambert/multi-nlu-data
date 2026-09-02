"""CLI for inspecting the Mix* multi-intent NLU datasets."""

import random

import typer

from multi_nlu_data.data import DATASETS, SPLITS, Example, load

app = typer.Typer(help="Inspect the Mix* multi-intent NLU datasets.")

DATASET = typer.Option("mixsnips", help=f"One of {', '.join(DATASETS)}.")
SPLIT = typer.Option("train", help=f"One of {', '.join(SPLITS)}.")


@app.command()
def show(
    split: str = SPLIT,
    dataset: str = DATASET,
    n: int = typer.Option(5, help="Number of examples to print."),
    seed: int | None = typer.Option(None, help="Sample randomly with this seed."),
) -> None:
    """Print examples from a split."""
    examples = load(split, dataset)
    chosen = (
        random.Random(seed).sample(examples, n) if seed is not None else examples[:n]
    )

    typer.echo(f"{dataset}/{split}: {len(examples)} examples\n")
    for i, ex in enumerate(chosen):
        print_example(i, ex)


@app.command()
def stats(split: str = SPLIT, dataset: str = DATASET) -> None:
    """Print intent counts and utterance-length distribution for a split."""
    examples = load(split, dataset)
    counts: dict[str, int] = {}
    per_utt: dict[int, int] = {}
    for ex in examples:
        per_utt[len(ex.intents)] = per_utt.get(len(ex.intents), 0) + 1
        for intent in ex.intents:
            counts[intent] = counts.get(intent, 0) + 1

    typer.echo(f"{dataset}/{split}: {len(examples)} examples\n")
    typer.echo("intents per utterance:")
    for k in sorted(per_utt):
        typer.echo(f"  {k}: {per_utt[k]}")
    typer.echo("\nintent frequency:")
    for intent, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {intent:<24} {c}")


def print_example(i: int, ex: Example) -> None:
    typer.secho(f"[{i}] {ex.text}", bold=True)
    typer.echo(f"    intents: {', '.join(ex.intents)}")
    for slot in ex.slots:
        typer.echo(f"    {slot.name:<20} {slot.value}")
    typer.echo("")


if __name__ == "__main__":
    app()
