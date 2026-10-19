import pytest

from goldpass.perturb import (
    DEFAULT_SENTENCES,
    add_sentence,
    build,
    numbers_in,
    perturb_number,
)
from goldpass.record import Passage, Record

CTX = ["The stationary noise limit for EMUs is 65 dB, measured at 7,5 m from the track."]


@pytest.mark.parametrize(
    ("text", "keys"),
    [
        ("65 dB at 7,5 m", {"65", "7.5"}),
        ("0.80 and 0,8", {"0.8"}),
        ("section 4.2.3", {"4.2.3"}),
        ("1 000 parts", {"1000"}),
        ("within four months", {"4"}),
    ],
)
def test_numbers_are_normalised(text: str, keys: set[str]) -> None:
    assert numbers_in(text) == keys


def test_number_is_changed_to_a_value_the_contexts_never_mention() -> None:
    p = perturb_number("q1", "The limit is 65 dB [1].", CTX)
    assert p is not None
    assert (p.kind, p.original, p.planted) == ("number", "65", "66")
    assert p.answer == "The limit is 66 dB [1]."
    assert p.answer[p.position : p.position + 2] == "66"


def test_citation_markers_and_unsupported_numbers_are_left_alone() -> None:
    assert perturb_number("q1", "See [1] and [2].", CTX) is None
    assert perturb_number("q1", "It is 99 dB.", CTX) is None  # 99 is not in the contexts


def test_new_value_must_not_appear_anywhere_in_the_contexts() -> None:
    ctx = ["Limits: 65 dB for EMUs, 66 dB for DMUs, 67 dB for locomotives."]
    p = perturb_number("q1", "The EMU limit is 65 dB.", ctx)
    assert p is not None and p.planted == "68"


def test_range_endpoints_are_skipped_and_upper_bounds_move_down() -> None:
    ctx = ["Curves of 205 to 300 m radius. Reply within four months. Speed 80 km/h."]
    p = perturb_number("q1", "Curves of 205 to 300 m; reply within four months.", ctx)
    assert p is not None and (p.original, p.planted) == ("four", "three")  # stricter, not implied


def test_sentence_is_appended_and_never_one_the_contexts_support() -> None:
    p = add_sentence("q1", "The limit is 65 dB [1].", CTX, DEFAULT_SENTENCES)
    assert p.kind == "sentence" and p.answer.startswith("The limit is 65 dB [1]. ")
    assert p.answer.endswith(" [1]") and p.planted in p.answer
    marker, _ = next(s for s in DEFAULT_SENTENCES if s[1] in p.planted)
    assert marker not in CTX[0].lower()


def test_sentence_choice_is_deterministic_and_skips_supported_markers() -> None:
    a = add_sentence("q7", "x", CTX, DEFAULT_SENTENCES)
    assert a == add_sentence("q7", "x", CTX, DEFAULT_SENTENCES)
    markers = [m for m, _ in DEFAULT_SENTENCES]
    every_but_last = [" ".join(markers[:-1])]
    p = add_sentence("q7", "x", every_but_last, DEFAULT_SENTENCES)
    assert DEFAULT_SENTENCES[-1][1] in p.planted


def test_no_usable_sentence_is_an_error() -> None:
    with pytest.raises(ValueError, match="every planted sentence"):
        add_sentence("q1", "x", [" ".join(m for m, _ in DEFAULT_SENTENCES)], DEFAULT_SENTENCES)


def test_build_skips_refusals_and_answers_without_context_text() -> None:
    def rec(i: str, answer: str, refused: bool = False, text: str | None = CTX[0]) -> Record:
        return Record(i, answer, refused, contexts=(Passage("p1", "d1", text),))

    records = [
        rec("a", "The limit is 65 dB [1]."),
        rec("b", "No number here [1]."),
        rec("r", "I cannot answer.", refused=True),
        rec("t", "65 dB", text=None),
    ]
    got = [(p.id, p.kind) for p in build(records, DEFAULT_SENTENCES)]
    assert got == [("a", "number"), ("a", "sentence"), ("b", "sentence")]


def test_numbers_after_a_lower_bound_condition_are_skipped() -> None:
    # "greater than 260" is implied by "greater than 250" for a condition like this one
    ctx = ["New lines with a speed greater than 250 km/h shall use the systems in 4.2.3."]
    p = perturb_number("q8", "Lines with a speed greater than 250 km/h use 4.2.3 [1].", ctx)
    assert p is not None and (p.original, p.planted) == ("4.2.3", "4.2.4")
