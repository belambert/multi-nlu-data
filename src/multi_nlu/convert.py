"""CLI for converting MixSNIPS into segmented, XML-tagged training data."""

import collections
import json
import os
from pathlib import Path

import typer

from multi_nlu.data import SPLITS
from multi_nlu.segment import segment_split, to_xml

app = typer.Typer(help="Convert MixSNIPS into per-intent XML.")


@app.command()
def xml(
    split: str = typer.Option("train", help=f"One of {', '.join(SPLITS)}."),
    out: Path = typer.Option(None, help="Write JSONL here instead of stdout."),
    llm: bool = typer.Option(False, help="Send unresolved rows to Fireworks."),
    model: str = typer.Option(
        None, help="Fireworks model; defaults to label.DEFAULT_MODEL."
    ),
    cache: Path = typer.Option(
        "llm-cache.jsonl", help="Reuse and record LLM answers here."
    ),
    limit: int = typer.Option(0, help="Cap the number of LLM calls (0 = no cap)."),
) -> None:
    """Segment a split and emit nested intent/slot XML."""
    done, todo = segment_split(split)
    typer.echo(
        f"{split}: {len(done)} resolved by heuristics, {len(todo)} unresolved", err=True
    )

    failed = todo
    if llm and todo:
        from multi_nlu.label import DEFAULT_MODEL, label_all

        batch = todo[:limit] if limit else todo
        labelled, failed = label_all(batch, model or DEFAULT_MODEL, cache)
        done += labelled
        failed += todo[len(batch) :]
        typer.echo(
            f"llm: {len(labelled)} labelled, {len(failed)} still unresolved", err=True
        )

    lines = [
        json.dumps({"text": s.example.text, "xml": to_xml(s), "source": s.source})
        for s in done
    ]
    if out:
        out.write_text("\n".join(lines) + "\n")
        typer.echo(f"wrote {len(lines)} rows to {out}", err=True)
    else:
        for line in lines:
            typer.echo(line)


@app.command()
def coverage(
    split: str = typer.Option("train", help=f"One of {', '.join(SPLITS)}.")
) -> None:
    """Report how much of a split the heuristics settle without an LLM."""
    done, todo = segment_split(split)
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
    """List Fireworks models, to pick a real id for `xml --llm`."""
    from fireworks import Fireworks

    client = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id="fireworks")
    for m in client.models.list(page_size=200):
        if filter in m.name and m.state == "READY":
            typer.echo(f"{m.name:<48} {m.display_name}")


if __name__ == "__main__":
    app()
