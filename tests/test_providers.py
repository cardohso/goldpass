import pytest

from goldpass.providers import SameFamilyError, check_judge_family, model_family


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("openai/gpt-oss-120b", "openai"),
        ("gpt-4.1-mini", "openai"),
        ("o3-mini", "openai"),
        ("gemini-2.5-flash", "google"),
        ("google/gemma-3-27b-it", "google"),
        ("claude-sonnet-4-5", "anthropic"),
        ("llama-3.3-70b-versatile", "meta"),
        ("meta-llama/llama-4-scout-17b-16e-instruct", "meta"),
        ("mistral-small-latest", "mistral"),
        ("qwen/qwen3-32b", "alibaba"),
        ("deepseek-r1-distill-llama-70b", "deepseek"),  # a DeepSeek distill of Llama: maker wins
        ("moonshotai/kimi-k2-instruct", "moonshot"),
        ("my-finetune-v2", None),
        (None, None),
    ],
)
def test_model_family(model: str | None, family: str | None) -> None:
    assert model_family(model) == family


def test_different_families_pass() -> None:
    check_judge_family("groq:openai/gpt-oss-120b", "openai", "google", allow_same=False)


def test_same_family_is_refused() -> None:
    with pytest.raises(SameFamilyError, match=r"same model family \(google\).*self-preference"):
        check_judge_family("gemini:gemini-2.5-pro", "google", "google", allow_same=False)


def test_unknown_families_are_refused() -> None:
    with pytest.raises(SameFamilyError, match="generator's model family is unknown"):
        check_judge_family("groq:x", "openai", None, allow_same=False)
    with pytest.raises(SameFamilyError, match="unrecognised model family"):
        check_judge_family("custom:x", None, "google", allow_same=False)


def test_override() -> None:
    check_judge_family("gemini:gemini-2.5-pro", "google", "google", allow_same=True)
    check_judge_family("x", None, None, allow_same=True)
