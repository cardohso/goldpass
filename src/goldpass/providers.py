"""Judge providers (OpenAI-compatible endpoints) and model families.

The same-family guard compares model *families* (who trained the model), not API hosts:
`openai/gpt-oss-120b` served by Groq is an OpenAI model, and a Llama judge on OpenRouter
shares a family with a Llama generator on Groq. LLM judges tend to prefer outputs from their
own family (self-preference bias), so a same-family judge inflates the scores it gates on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    base_url: str
    api_key_env: str | None  # None: no key needed (local server)


PROVIDERS: dict[str, Provider] = {
    "groq": Provider("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": Provider("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openai": Provider("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": Provider("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "mistral": Provider("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "ollama": Provider("http://localhost:11434/v1", None),
}

# First match wins, so order matters: "gpt-oss" is OpenAI whoever serves it, and a distill
# such as deepseek-r1-distill-llama counts as its maker's (DeepSeek). Families of distills and
# fine-tunes are a judgement call; set `family` explicitly when it matters.
FAMILIES: list[tuple[str, str]] = [
    (r"deepseek", "deepseek"),
    (r"gpt-oss|\bgpt-|(^|/)o[1-9]\b|(^|/)openai/", "openai"),
    (r"gemini|gemma", "google"),
    (r"claude", "anthropic"),
    (r"llama", "meta"),
    (r"mistral|mixtral|codestral|ministral|magistral|devstral", "mistral"),
    (r"qwen|qwq", "alibaba"),
    (r"\bphi-", "microsoft"),
    (r"command", "cohere"),
    (r"grok", "xai"),
    (r"kimi|moonshot", "moonshot"),
]


def model_family(model: str | None) -> str | None:
    """Best-effort family from a model name; None if unrecognised (then set `family`)."""
    if not model:
        return None
    name = model.lower()
    return next((family for pattern, family in FAMILIES if re.search(pattern, name)), None)


class SameFamilyError(ValueError):
    pass


def check_judge_family(
    judge: str, judge_family: str | None, generator_family: str | None, allow_same: bool
) -> None:
    """Refuse to gate on a judge that cannot be shown to differ from the generator's family."""
    if allow_same:
        return
    hint = "or pass --allow-same-family (results will be marked as such)"
    if generator_family is None:
        raise SameFamilyError(
            "the generator's model family is unknown, so goldpass cannot check that the gating "
            "judge is from a different family: set generator.model or generator.family in "
            f"goldpass.yaml (or --generator MODEL), {hint}"
        )
    if judge_family is None:
        raise SameFamilyError(
            f"the gating judge {judge} has an unrecognised model family: set `family` on it, {hint}"
        )
    if judge_family == generator_family:
        raise SameFamilyError(
            f"the gating judge {judge} is from the same model family ({judge_family}) as the "
            "generator. LLM judges favour outputs from their own family (self-preference bias), "
            f"which inflates the scores being gated on. Use a judge from another family, {hint}"
        )
