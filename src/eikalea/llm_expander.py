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

By default, a run draws each of the six axes without repeats (see
no_repeat_message_generator): every value in a pool is used once before any
repeats, instead of plain independent per-seed sampling letting ordinary
variance over-represent a few values across a long run. --repeat opts back
into that plain per-seed dynamicprompts resolution (build_user_message),
needed to hold axis draws fixed while comparing --model or --template on
the same seed. Either way, the LLM's own request-level "seed" field always
reproduces the synthesis call itself.

Talks to any OpenAI-compatible chat completions server (Ollama by default,
via `ollama serve`, but also LM Studio/vLLM/etc. if pointed at one) over
the standard /v1/chat/completions endpoint.
"""

import random
import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import openai
from dynamicprompts.enums import SamplingMethod
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.sampling_context import SamplingContext
from dynamicprompts.wildcards import WildcardManager
from openai import OpenAI

SYSTEM_PROMPT_PATH = Path(__file__).parent / "expander_system_prompt.txt"
TEMPLATE_PATH = Path(__file__).parent / "template.md"
WILDCARDS_DIR = Path(__file__).parent / "wildcards"

_WILDCARD_TOKEN_RE = re.compile(r"__([\w/*]+)__")

# Offset so the model pick isn't derived from the same draw as the template's
# own wildcards (dynamicprompts already decorrelates wildcards drawn together
# within one generate() call, but the model choice happens outside of that).
_MODEL_SEED_OFFSET = 5_999_999_789

# The six axes the packaged template/wildcards define. Distinct offsets so
# each axis's shuffle order doesn't move in lockstep with the others.
AXIS_NAMES = ("medium", "composition", "subject", "palette", "mood", "movement")
_AXIS_SEED_OFFSETS = dict(
    zip(AXIS_NAMES, range(3_100_000_001, 3_100_000_001 + len(AXIS_NAMES)), strict=True)
)


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


def no_repeat_message_generator(
    start_seed: int,
    template_path: Path | str | None = None,
    wildcards_dir: Path | str | None = None,
) -> Iterator[str]:
    """One persistent user-message generator for a whole run, drawing the
    six named axes without repeats (dynamicprompts' own CyclicalSampler --
    it cycles through a pool once before repeating any value) instead of
    plain per-seed independent sampling. We supply the pool ourselves,
    shuffled once from `start_seed`, via WildcardManager's own root_map
    argument (in-memory, no files written) rather than the default
    directory scan, so a given starting seed still reproduces the whole
    run's draw order. Real wildcard files (default or --wildcards-dir)
    still back everything else a custom template might reference.
    dynamicprompts does the actual template resolution throughout; nothing
    here reimplements any part of it. Yields indefinitely -- also covers
    --count -1 (run until interrupted), which has no upfront image count."""
    template = Path(template_path or TEMPLATE_PATH).read_text(encoding="utf-8")
    wildcards_dir = Path(wildcards_dir or WILDCARDS_DIR)

    # get_values() resolves through dynamicprompts' own collection matching,
    # so this reads a flat axis.txt and a grouped axis.yaml's subgroups the
    # same way -- unlike the old direct-file-read, this doesn't care which
    # form an axis is in. Whether an axis is grouped is detected per call
    # from the actual wildcards_dir (try the "name/*" glob first, fall back
    # to the flat name) rather than hardcoded per axis name -- a fixed
    # "medium is always grouped" assumption would silently stop reshuffling
    # medium for anyone still pointing --wildcards-dir at a flat
    # medium.txt (confirmed: it wouldn't error, just quietly fall back to
    # unshuffled file-order cycling). Weights on individual entries (a
    # .yaml/.json-only feature) don't survive into this shuffled/no-repeat
    # pool, same as before this axis loader supported .yaml at all -- the
    # old direct-file-read never had a notion of weights either.
    source_wildcard_manager = WildcardManager(path=wildcards_dir)
    shuffled_axes = {}
    for name in AXIS_NAMES:
        grouped_pattern = f"{name}/*"
        values = list(source_wildcard_manager.get_values(grouped_pattern).string_values)
        if values:
            random.Random(start_seed + _AXIS_SEED_OFFSETS[name]).shuffle(values)
            # The real subgroup collections this glob matches (e.g.
            # medium/prints, medium/drawing) are still visible through
            # wildcards_dir below, so without this they'd be drawn from
            # *in addition to* this shuffled pool -- doubling up on the
            # same values instead of replacing them. Overriding each real
            # subgroup down to empty, then adding the one combined pool as
            # a fresh subgroup under the same glob, is what makes the
            # override a true replacement. (Confirmed empirically:
            # skipping this step doubled/tripled draws for a grouped axis.)
            for real_name in source_wildcard_manager.get_collection_names():
                if real_name.startswith(f"{name}/"):
                    shuffled_axes[real_name] = []
            shuffled_axes[f"{name}/__no_repeat_pool__"] = values
            continue

        values = list(source_wildcard_manager.get_values(name).string_values)
        if not values:
            continue
        random.Random(start_seed + _AXIS_SEED_OFFSETS[name]).shuffle(values)
        shuffled_axes[name] = values

    wildcard_manager = WildcardManager(root_map={"": [wildcards_dir, shuffled_axes]})
    wildcard_manager.sort_wildcards = False
    context = SamplingContext(wildcard_manager=wildcard_manager, default_sampling_method=SamplingMethod.CYCLICAL)
    for result in context.sample_prompts(template, None):
        yield str(result).strip()


def validate_template(
    template_path: Path | str | None = None,
    wildcards_dir: Path | str | None = None,
) -> list[str]:
    """Returns the names of any wildcards the template references (as
    __name__) that aren't defined in the wildcards directory -- a typo in
    a custom --template otherwise fails silently, leaving the literal
    "__name__" token unresolved in what gets sent to the LLM instead of
    raising a clear error. Checked via match_collections (glob matching)
    rather than an exact-name set membership check, so a grouped axis's
    __medium/*__-style reference validates correctly too -- "medium/*"
    is never itself a real collection name, only a pattern that matches
    one or more (medium/prints, medium/drawing, ...)."""
    template = Path(template_path or TEMPLATE_PATH).read_text(encoding="utf-8")
    wildcard_manager = WildcardManager(path=wildcards_dir or WILDCARDS_DIR)
    referenced = set(_WILDCARD_TOKEN_RE.findall(template))
    missing = [name for name in referenced if not any(wildcard_manager.match_collections(name))]
    return sorted(missing)


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def export_templates(dest_dir: Path | str) -> Path:
    """Copy the packaged default template.md + wildcards/*.{txt,yaml,json}
    into `dest_dir`, so it can be edited and pointed back at via --template
    / --wildcards-dir instead of hunting for the files inside the installed
    package. Every extension WildcardManager's own directory scan treats as
    a wildcard collection -- wildcards.yaml (the grouped medium axis plus
    the five flat ones) needs to be exported too, or --wildcards-dir points
    at a directory missing every axis entirely."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE_PATH, dest / TEMPLATE_PATH.name)
    wildcards_dest = dest / "wildcards"
    wildcards_dest.mkdir(exist_ok=True)
    wildcard_files = (f for ext in ("*.txt", "*.yaml", "*.json") for f in WILDCARDS_DIR.glob(ext))
    for wildcard_file in sorted(wildcard_files):
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
    user_message: str | None = None,
) -> str:
    """Uses the standard OpenAI-compatible /v1/chat/completions surface, via
    the official `openai` client rather than a hand-rolled request, so this
    works unmodified against any OpenAI-compatible server (Ollama, LM
    Studio, vLLM, etc.) pointed at via `host`. "reasoning_effort" is the
    field that actually controls hidden chain-of-thought on this endpoint --
    Ollama's own native "think": false is not honored there (confirmed
    empirically: qwen3.5/qwen3.6/gemma4 all kept reasoning with "think":
    false, all stopped with "reasoning_effort": "none", going from ~20-30s/
    prompt to <1s). Defaults to "none" since reasoning adds latency without
    improving this particular task (short synthesis, not multi-step problem
    solving); pass "low"/"medium"/"high" to re-enable it. Not every
    reasoning-capable model supports this override though -- some reject the
    request outright (a BadRequestError whose `param` is "reasoning_effort")
    rather than ignoring it, so that specific failure is retried once
    without the parameter rather than surfaced as a crash; any other failure
    propagates normally. `user_message`, when given (from
    no_repeat_message_generator), is used as-is instead of resolving the
    template here -- the seed still drives the request itself."""
    if user_message is None:
        user_message = build_user_message(seed, template_path, wildcards_dir)
    client = OpenAI(base_url=f"{host}/v1", api_key="not-needed", timeout=120)
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "seed": seed,
    }
    try:
        response = client.chat.completions.create(reasoning_effort=reasoning_effort, **kwargs)
    except openai.BadRequestError as exc:
        if exc.param != "reasoning_effort":
            raise
        response = client.chat.completions.create(**kwargs)
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
    user_message: str | None = None,
) -> str:
    system_prompt = load_system_prompt()
    model = pick_model(seed, models)
    return generate_with_llm(
        seed, system_prompt, model=model, host=host,
        template_path=template_path, wildcards_dir=wildcards_dir,
        reasoning_effort=reasoning_effort, user_message=user_message,
    )
