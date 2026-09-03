"""CLI for converting the Mix* datasets into segmented, span-annotated data."""

import collections
import json
import os
import sys
from pathlib import Path

import typer

from multi_nlu_data.data import Dataset, Split
from multi_nlu_data.segment import segment_split, to_spans

app = typer.Typer(help="Convert the Mix* datasets into per-intent spans.")

DATASET = typer.Option(Dataset.MIXSNIPS, help="Which corpus to convert.")
SPLIT = typer.Option(Split.TRAIN, help="Which split to convert.")


@app.command()
def spans(
    split: Split = SPLIT,
    dataset: Dataset = DATASET,
    out: Path = typer.Option(None, help="Write JSONL here instead of stdout."),
    llm: bool = typer.Option(False, help="Send unresolved rows to Fireworks."),
    model: str = typer.Option(
        None, help="Fireworks model; defaults to label.DEFAULT_MODEL."
    ),
    cache: Path = typer.Option(
        "llm-cache.jsonl", help="Reuse and record LLM answers here."
    ),
    limit: int = typer.Option(0, help="Cap the number of LLM calls (0 = no cap)."),
    workers: int = typer.Option(8, help="Concurrent LLM requests."),
) -> None:
    """Segment a split and emit intent/slot spans as character offsets."""
    done, todo = segment_split(split, dataset)
    typer.echo(
        f"{dataset}/{split}: {len(done)} resolved by heuristics, {len(todo)} unresolved",
        err=True,
    )

    failed = todo
    if llm and todo:
        from multi_nlu_data.label import DEFAULT_MODEL, label_all

        batch = todo[:limit] if limit else todo
        # on stderr, so a piped JSONL stream stays clean
        with typer.progressbar(
            length=len(batch), label="labelling", file=sys.stderr
        ) as bar:
            labelled, failed = label_all(
                batch, model or DEFAULT_MODEL, cache, workers, lambda: bar.update(1)
            )
        done += labelled
        failed += todo[len(batch) :]
        typer.echo(
            f"llm: {len(labelled)} labelled, {len(failed)} still unresolved", err=True
        )

    lines = [
        json.dumps({"text": s.example.text, "intents": to_spans(s), "source": s.source})
        for s in done
    ]
    if out:
        out.write_text("\n".join(lines) + "\n")
        typer.echo(f"wrote {len(lines)} rows to {out}", err=True)
    else:
        for line in lines:
            typer.echo(line)

    if failed:
        report_failures(failed, out)


def report_failures(failed: list, out: Path | None) -> None:
    """Write dropped rows beside the output so a long run cannot lose them silently."""
    if not out:
        typer.echo(
            f"warning: {len(failed)} rows dropped (pass --out to record them)", err=True
        )
        return

    path = out.with_suffix(".failed.jsonl")
    path.write_text(
        "\n".join(
            json.dumps({"text": ex.text, "intents": ex.intents, "tokens": ex.tokens})
            for ex in failed
        )
        + "\n"
    )
    typer.secho(
        f"warning: {len(failed)} rows dropped, listed in {path}", err=True, fg="yellow"
    )


@app.command()
def coverage(split: Split = SPLIT, dataset: Dataset = DATASET) -> None:
    """Report how much of a split the heuristics settle without an LLM."""
    done, todo = segment_split(split, dataset)
    by_source = collections.Counter(s.source for s in done)
    total = len(done) + len(todo)

    for source, n in by_source.most_common():
        typer.echo(f"  {source:<12} {n:>6}  {n/total:.1%}")
    typer.echo(f"  {'unresolved':<12} {len(todo):>6}  {len(todo)/total:.1%}")
    typer.echo(f"\n{len(done)}/{total} resolved without an LLM ({len(done)/total:.1%})")


@app.command()
def models(
    filter: str = typer.Option("", help="Substring to match against model ids.")
) -> None:
    """List Fireworks models, to pick a real id for `spans --llm`."""
    from fireworks import Fireworks

    client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id="fireworks")
    for m in client.models.list(page_size=200):
        if filter in m.name and m.state == "READY":
            typer.echo(f"{m.name:<48} {m.display_name}")


if __name__ == "__main__":
    app()
