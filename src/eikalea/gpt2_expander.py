# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""
Alternate seed -> prompt path: a GPT-2 fine-tune trained on old-style
Stable Diffusion tag prompts (default: Gustavosta/MagicPrompt-Stable-
Diffusion) free-associates a rough draft from scratch, which then goes
through the same LLM synthesis call as the six-axis path (see
llm_expander.generate_with_llm's user_message parameter) to be rewritten
into one coherent paragraph -- via its own template (template_gpt2.md)
and instructions, kept deliberately separate from template.md/the six
wildcard axes rather than folded into them.

Reuses dynamicprompts' own MagicPromptGenerator (dynamicprompts.generators.
magicprompt) for the GPT-2 call itself rather than a hand-rolled
transformers wrapper -- it already handles model loading/caching, seeding
(via transformers.set_seed), and tag-soup clean-up. Requires the
`dynamicprompts[magicprompt]` extra (transformers + torch), only imported
when this mode is actually used -- a plain eikalea install doesn't need to
pull those in.

llm_expander's expander_system_prompt.txt already tells the LLM to
ignore embedded tag-soup boilerplate and defer to any axes given
separately (see that file) -- covers the messy GPT-2 drafts here too, so
this module doesn't carry a system prompt of its own; a near-duplicate
gpt2-specific one existed briefly before being folded into the shared
file instead.

build_axis_message_with_gpt2_subject offers a second, narrower way to use
the GPT-2 draft: instead of replacing the whole six-axis template, it
swaps out just the subject axis for a fresh draft, leaving
medium/composition/palette/mood/movement drawing from their curated pools
as usual -- the six-axis path's own style variety was what pure gpt2-seed
mode was missing, this keeps it and only lets GPT-2 supply the subject.
Confirmed empirically (Gustavosta/MagicPrompt-Stable-Diffusion, seed 100)
that a raw draft can bundle its own style/artist/lighting language in
with the subject matter (e.g. "...beautifully lit, artgerm, joshua
middleton comic cover art"), which would otherwise compete with the axes
drawn separately -- handled by the shared system prompt rather than
trying to strip it out programmatically.

generate_gpt2_expansion_of_axes offers a third direction: instead of GPT-2
supplying material that flows into the six-axis template, the resolved
six-axis line flows into GPT-2 -- it becomes the seed text GPT-2
continues from, and the combined (axis text + GPT-2 continuation) result
goes through the gpt2-seed template for the final LLM pass, same as
--gpt2-seed.
"""

from pathlib import Path

from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.generators.magicprompt import DEFAULT_MODEL_NAME, MagicPromptGenerator
from dynamicprompts.wildcards import WildcardManager

GPT2_TEMPLATE_PATH = Path(__file__).parent / "template_gpt2.md"
DEFAULT_GPT2_MODEL = DEFAULT_MODEL_NAME


def generate_gpt2_seed_text(
    seed: int,
    model_name: str | None = None,
    seed_text: str = "",
    max_prompt_length: int = 100,
    device: str | None = None,
) -> str:
    """One raw draft prompt, seeded for reproducibility (same seed -> same
    draft). device=None (default) leaves MagicPromptGenerator's own choice
    in place -- CUDA when available, since this is a small model (a
    couple hundred MB) that fits alongside most LLM backends. Pass
    device="cpu" (see --gpt2-cpu) on tighter VRAM budgets, where it would
    otherwise compete with --model (and any ComfyUI rendering afterward)
    for the same GPU memory.

    `seed_text`, when given, is what GPT-2 continues from instead of
    free-associating from nothing -- MagicPromptGenerator's own clean-up
    keeps it as the result's prefix, so it survives intact with GPT-2's
    continuation appended after it (see generate_gpt2_expansion_of_axes).
    `max_prompt_length` caps the *total* sequence length (seed_text plus
    whatever GPT-2 adds), not just the addition -- confirmed empirically
    that a long seed_text left at the 100-token default leaves no room to
    actually generate anything (the result comes back unchanged), so a
    non-trivial seed_text needs a correspondingly larger budget."""
    generator = MagicPromptGenerator(
        model_name=model_name or DEFAULT_GPT2_MODEL, seed=seed, device=device, max_prompt_length=max_prompt_length
    )
    return generator.generate(seed_text, num_images=1)[0].strip()


def build_gpt2_user_message(gpt2_seed_text: str, template_path: Path | str | None = None) -> str:
    """Substitutes the GPT-2 draft into the gpt2-seed template (default:
    the packaged template_gpt2.md). Plain str.format(), not dynamicprompts
    template resolution -- the draft is dynamically generated text, not a
    value drawn from a static wildcard pool, so there's nothing for
    dynamicprompts to resolve here."""
    template = Path(template_path or GPT2_TEMPLATE_PATH).read_text(encoding="utf-8")
    return template.format(gpt2_seed=gpt2_seed_text.strip())


def build_axis_message_with_gpt2_subject(
    seed: int,
    gpt2_model_name: str | None = None,
    template_path: Path | str | None = None,
    wildcards_dir: Path | str | None = None,
    gpt2_device: str | None = None,
) -> str:
    """Resolves the six-axis template (default: llm_expander.TEMPLATE_PATH)
    exactly like llm_expander.build_user_message, except the subject axis
    is a fresh GPT-2 draft instead of a draw from subject.txt --
    medium/composition/palette/mood/movement still come from their curated
    wildcard pools. The draft is injected as a single-value wildcard
    collection via WildcardManager's own root_map argument (the same
    technique llm_expander.no_repeat_message_generator uses for its
    shuffled pools -- items later in a root_map list override earlier ones
    with the same collection name), not a custom substitution.
    dynamicprompts still does all the actual template resolution.

    Confirmed empirically that dynamicprompts' RandomSampler hangs forever
    (reproduced with a 10s timeout) if the drawn wildcard value is an
    empty string -- SamplingContext.sample_prompts("") short-circuits to
    an empty result instead of yielding one, so
    RandomSampler._get_wildcard's `while True` retry loop spins without
    ever producing output. Gustavosta/MagicPrompt-Stable-Diffusion does
    occasionally return an empty draft (confirmed at more than one seed),
    so an empty draft here skips the override entirely and falls back to
    a normal draw from subject.txt for this seed, rather than injecting
    the empty string and triggering that hang."""
    from .llm_expander import TEMPLATE_PATH, WILDCARDS_DIR

    template = Path(template_path or TEMPLATE_PATH).read_text(encoding="utf-8")
    wildcards_dir = Path(wildcards_dir or WILDCARDS_DIR)
    gpt2_draft = generate_gpt2_seed_text(seed, gpt2_model_name, device=gpt2_device)
    overrides = {"subject": [gpt2_draft]} if gpt2_draft else {}

    wildcard_manager = WildcardManager(root_map={"": [wildcards_dir, overrides]})
    generator = RandomPromptGenerator(wildcard_manager=wildcard_manager, seed=seed)
    return generator.generate(template, num_images=1)[0].strip()


def generate_gpt2_expansion_of_axes(
    seed: int,
    gpt2_model_name: str | None = None,
    template_path: Path | str | None = None,
    wildcards_dir: Path | str | None = None,
    gpt2_device: str | None = None,
) -> str:
    """The reverse direction from build_axis_message_with_gpt2_subject:
    resolves the six-axis template via our own llm_expander.build_user_message
    first, then lets GPT-2 continue/elaborate on the axis line (just
    "Medium: X. Composition: Y. ..." -- the template's first paragraph,
    not the synthesis instructions that follow it in template.md) as its
    seed text. Deliberately done as two explicit steps in our own code,
    not via dynamicprompts' own MagicPromptGenerator(prompt_generator=
    RandomPromptGenerator(...)) chaining (which the library's README shows
    as the standard way to do this) -- that would hand the six-axis
    resolution over to a generator instance MagicPromptGenerator
    constructs and owns internally, where here build_user_message (and
    whatever no-repeat/override behavior it may gain later) stays fully
    ours.

    Only the axis line goes to GPT-2, not the trailing instructions --
    confirmed empirically that the instructions alone push the resolved
    template well past 1000 characters, and GPT-2 isn't instruction-tuned
    anyway; feeding it instructions meant for the synthesis LLM would just
    waste its (small) token budget and risk that text getting embedded
    twice once template_gpt2.md's own instructions wrap the result.
    GPT-2's own clean-up keeps the axis line as the result's prefix, so it
    survives intact for the LLM's synthesis pass, with GPT-2's own
    continuation appended after it."""
    from .llm_expander import build_user_message

    axis_text = build_user_message(seed, template_path, wildcards_dir)
    axis_line = axis_text.split("\n\n", 1)[0].strip()
    max_prompt_length = len(axis_line) // 3 + 80
    return generate_gpt2_seed_text(
        seed, gpt2_model_name, seed_text=axis_line, max_prompt_length=max_prompt_length, device=gpt2_device
    )
