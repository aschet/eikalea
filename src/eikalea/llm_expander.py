# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""
LLM-driven prompt generation: seed -> finished image prompt.
----------------------------------------------------------------
Used by cli.py. The six priming axes (medium, composition, subject,
palette, mood, art movement) live as wildcard text files under wildcards/,
resolved through `dynamicprompts` -- the same templating library behind the
sd-dynamic-prompts A1111/ComfyUI extensions -- via template.md, which also
carries the axis-specific instructions for how to unify them
(see that file; kept as .md rather than .txt so WildcardManager's directory
scan, which picks up every .txt/.json/.yaml file as a wildcard collection,
never mistakes it for one if it ends up sitting next to the wildcard
files). Both are packaged defaults, but fully overridable per call (see
build_user_message), so someone can restructure the axes entirely without
touching this module.

expander_system_prompt.txt, by contrast, stays axis-independent: output
format only (full sentences not tags, no camera/quality boilerplate, no
narrative padding) -- true regardless of what template/wildcards are in use.

The seed drives dynamicprompts' own sampler (for axis selection) and the
LLM's request-level "seed" field (for the synthesis call), so the same seed
reproduces the same prompt end to end.

Talks to any OpenAI-compatible chat completions server (Ollama by default,
via `ollama serve`, but also LM Studio/vLLM/etc. if pointed at one) over
the standard /v1/chat/completions endpoint.
"""

import random
import re
import shutil
from pathlib import Path

from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards import WildcardManager
from openai import OpenAI

SYSTEM_PROMPT_PATH = Path(__file__).parent / "expander_system_prompt.txt"
TEMPLATE_PATH = Path(__file__).parent / "template.md"
WILDCARDS_DIR = Path(__file__).parent / "wildcards"

_WILDCARD_TOKEN_RE = re.compile(r"__([\w/]+)__")

# Offset so the model pick isn't derived from the same draw as the template's
# own wildcards (dynamicprompts already decorrelates wildcards drawn together
# within one generate() call, but the model choice happens outside of that).
_MODEL_SEED_OFFSET = 5_999_999_789


def pick_model(seed: int, models: list[str]) -> str:
    """When more than one model is given, pick one per seed (same seed ->
    same model, like every other axis) rather than always using the first."""
    return random.Random(seed + _MODEL_SEED_OFFSET).choice(models)


def build_user_message(
    seed: int,
    template_path: Path | str | None = None,
    wildcards_dir: Path | str | None = None,
) -> str:
    """Resolves the template (default: the packaged template.md)
    against the wildcard files (default: the packaged wildcards/ directory)
    via dynamicprompts, seeded for reproducibility."""
    template = Path(template_path or TEMPLATE_PATH).read_text(encoding="utf-8")
    wildcard_manager = WildcardManager(path=wildcards_dir or WILDCARDS_DIR)
    generator = RandomPromptGenerator(wildcard_manager=wildcard_manager, seed=seed)
    return generator.generate(template, num_images=1)[0].strip()


def validate_template(
    template_path: Path | str | None = None,
    wildcards_dir: Path | str | None = None,
) -> list[str]:
    """Returns the names of any wildcards the template references (as
    __name__) that aren't defined in the wildcards directory -- a typo in
    a custom --template otherwise fails silently, leaving the literal
    "__name__" token unresolved in what gets sent to the LLM instead of
    raising a clear error."""
    template = Path(template_path or TEMPLATE_PATH).read_text(encoding="utf-8")
    wildcard_manager = WildcardManager(path=wildcards_dir or WILDCARDS_DIR)
    referenced = set(_WILDCARD_TOKEN_RE.findall(template))
    known = wildcard_manager.get_collection_names()
    return sorted(referenced - known)


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def export_templates(dest_dir: Path | str) -> Path:
    """Copy the packaged default template.txt + wildcards/*.txt into
    `dest_dir`, so it can be edited and pointed back at via --template /
    --wildcards-dir instead of hunting for the files inside the installed
    package."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE_PATH, dest / TEMPLATE_PATH.name)
    wildcards_dest = dest / "wildcards"
    wildcards_dest.mkdir(exist_ok=True)
    for wildcard_file in sorted(WILDCARDS_DIR.glob("*.txt")):
        shutil.copy2(wildcard_file, wildcards_dest / wildcard_file.name)
    return dest


def generate_with_llm(
    seed: int,
    system_prompt: str,
    model: str,
    host: str = "http://localhost:11434",
    template_path: Path | str | None = None,
    wildcards_dir: Path | str | None = None,
    reasoning_effort: str = "none",
) -> str:
    """Uses the standard OpenAI-compatible /v1/chat/completions surface, via
    the official `openai` client rather than a hand-rolled request, so this
    works unmodified against any OpenAI-compatible server (Ollama, LM
    Studio, vLLM, etc.) pointed at via `host`. "reasoning_effort" (passed
    via extra_body -- it's not part of the openai SDK's own request schema)
    is the field that actually controls hidden chain-of-thought on this
    endpoint -- Ollama's own native "think": false is not honored there
    (confirmed empirically: qwen3.5/qwen3.6/gemma4 all kept reasoning with
    "think": false, all stopped with "reasoning_effort": "none", going from
    ~20-30s/prompt to <1s). Defaults to "none" since reasoning adds latency
    without improving this particular task (short synthesis, not multi-step
    problem solving); pass "low"/"medium"/"high" to re-enable it."""
    client = OpenAI(base_url=f"{host}/v1", api_key="not-needed", timeout=120)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(seed, template_path, wildcards_dir)},
        ],
        seed=seed,
        extra_body={"reasoning_effort": reasoning_effort},
    )
    return response.choices[0].message.content.strip()


def unload_ollama_model(model: str, host: str = "http://localhost:11434") -> None:
    """Evict a model from GPU memory immediately (keep_alive: 0). Call this
    before any other VRAM-heavy step (e.g. ComfyUI/Krea2 image generation)
    run in the same process or on the same machine -- Ollama otherwise keeps
    the model resident for its default keep-alive window, which starves the
    next GPU consumer of memory."""
    import requests

    requests.post(f"{host}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)


def generate(
    seed: int,
    models: list[str],
    host: str = "http://localhost:11434",
    template_path: Path | str | None = None,
    wildcards_dir: Path | str | None = None,
    reasoning_effort: str = "none",
) -> str:
    system_prompt = load_system_prompt()
    model = pick_model(seed, models)
    return generate_with_llm(
        seed, system_prompt, model=model, host=host,
        template_path=template_path, wildcards_dir=wildcards_dir,
        reasoning_effort=reasoning_effort,
    )
