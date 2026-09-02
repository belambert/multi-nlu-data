import json

from multi_nlu_data.convert import report_failures
from multi_nlu_data.data import Example
from multi_nlu_data.label import build, trim
from multi_nlu_data.segment import (
    Segment,
    Segmentation,
    learn_slot_intents,
    segment,
    to_xml,
)

PLAY_RATE = Example(
    tokens="play isham jones and swine not deserves four points".split(),
    tags=[
        "O",
        "B-artist",
        "I-artist",
        "O",
        "B-object_name",
        "I-object_name",
        "O",
        "B-rating_value",
        "B-rating_unit",
    ],
    intents=["PlayMusic", "RateBook"],
)

SLOT_INTENTS = {
    "artist": {"PlayMusic", "AddToPlaylist"},
    "object_name": {"RateBook", "SearchCreativeWork"},
    "rating_value": {"RateBook"},
    "rating_unit": {"RateBook"},
}


def test_learn_slot_intents_uses_only_single_intent_rows():
    rows = [
        Example(["play", "x"], ["O", "B-artist"], ["PlayMusic"]),
        Example(["rate", "y"], ["O", "B-artist"], ["RateBook", "GetWeather"]),
    ]
    assert learn_slot_intents(rows) == {"artist": {"PlayMusic"}}


def test_single_intent_covers_whole_utterance():
    ex = Example(["play", "x"], ["O", "B-artist"], ["PlayMusic"])
    seg = segment(ex, SLOT_INTENTS)
    assert seg.source == "single"
    assert seg.segments == [Segment("PlayMusic", 0, 2)]


def test_ambiguous_slot_resolves_within_the_intent_set():
    """`artist` is ambiguous globally but not in a row without AddToPlaylist."""
    seg = segment(PLAY_RATE, SLOT_INTENTS)
    assert [s.intent for s in seg.segments] == ["PlayMusic", "RateBook"]
    assert seg.segments[0] == Segment("PlayMusic", 0, 3)


def test_connective_falls_outside_both_segments():
    seg = segment(PLAY_RATE, SLOT_INTENTS)
    assert PLAY_RATE.tokens[seg.segments[0].end] == "and"
    assert seg.segments[1].start == 4


def test_unresolvable_when_an_intent_has_no_anchor():
    ex = Example(["play", "x"], ["O", "B-artist"], ["PlayMusic", "GetWeather"])
    assert segment(ex, SLOT_INTENTS) is None


def test_to_xml_nests_slots_inside_intents_and_escapes():
    ex = Example(["find", "b&b"], ["O", "B-object_name"], ["SearchCreativeWork"])
    seg = Segmentation(ex, [Segment("SearchCreativeWork", 0, 2)], "single")
    assert to_xml(seg) == (
        "<SearchCreativeWork>find <object_name>b&amp;b</object_name></SearchCreativeWork>"
    )


def test_build_rejects_boundary_inside_a_slot_span():
    answer = {"order": ["PlayMusic", "RateBook"], "boundaries": [2]}
    assert build(PLAY_RATE, answer) is None


def test_build_rejects_wrong_intent_set():
    assert (
        build(PLAY_RATE, {"order": ["PlayMusic", "GetWeather"], "boundaries": [4]})
        is None
    )


def test_build_accepts_and_trims_the_connective():
    seg = build(PLAY_RATE, {"order": ["PlayMusic", "RateBook"], "boundaries": [3]})
    assert seg.source == "llm"
    assert [(s.intent, s.start, s.end) for s in seg.segments] == [
        ("PlayMusic", 0, 3),
        ("RateBook", 4, 9),
    ]


def test_trim_keeps_a_segment_that_is_only_a_connective():
    ex = Example(["and"], ["O"], ["PlayMusic"])
    assert trim(ex, Segment("PlayMusic", 0, 1)) == Segment("PlayMusic", 0, 1)


def test_trim_strips_a_stranded_connective_from_the_segment_end():
    """A cut inside "and then" must not leave "and" in the previous segment."""
    ex = Example(
        "play x and then rate y".split(),
        ["O", "B-artist", "O", "O", "O", "B-object_name"],
        ["PlayMusic", "RateBook"],
    )
    seg = build(ex, {"order": ["PlayMusic", "RateBook"], "boundaries": [3]})
    assert [(s.start, s.end) for s in seg.segments] == [(0, 2), (4, 6)]


def test_report_failures_writes_a_sidecar_beside_the_output(tmp_path):
    out = tmp_path / "train.jsonl"
    report_failures([PLAY_RATE], out)

    sidecar = tmp_path / "train.failed.jsonl"
    row = json.loads(sidecar.read_text())
    assert row["text"] == PLAY_RATE.text
    assert row["intents"] == PLAY_RATE.intents


def test_report_failures_without_an_output_path_only_warns(tmp_path):
    report_failures([PLAY_RATE], None)
    assert list(tmp_path.iterdir()) == []
