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

generate_gpt2_nudged_draft offers a third direction: instead of GPT-2
supplying material that flows into the six-axis template, a resolved
template (dynamicprompts syntax, same mechanism as the six-axis
template.md) flows into GPT-2 as its seed_text -- --gpt2-mode expand's
packaged default (template_gpt2_nudge.md) is a subset of the six axes,
just medium/palette/mood ("Medium: X. Palette: Y. Mood: Z."), but any
dynamicprompts template can stand in via --gpt2-nudge-template, including
the full six-axis line or one drawing from entirely different wildcards.
The combined (nudge text + GPT-2 continuation) result then goes through
the gpt2-seed template for the final LLM pass, same as --gpt2-mode seed
-- both modes share generate_gpt2_draft underneath, differing only in
whether a nudge template is resolved first.
"""

from pathlib import Path

from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.generators.magicprompt import DEFAULT_MODEL_NAME, MagicPromptGenerator
from dynamicprompts.wildcards import WildcardManager

GPT2_TEMPLATE_PATH = Path(__file__).parent / "template_gpt2.md"
GPT2_NUDGE_TEMPLATE_PATH = Path(__file__).parent / "template_gpt2_nudge.md"
DEFAULT_GPT2_MODEL = DEFAULT_MODEL_NAME


class Gpt2DraftUnavailable(RuntimeError):
    """Raised by generate_gpt2_draft once it gives up retrying an empty
    draft -- callers should skip this seed (no prompt, no output) rather
    than let it propagate as a generic crash."""


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
    continuation appended after it (see generate_gpt2_nudged_draft).
    `max_prompt_length` caps the *total* sequence length (seed_text plus
    whatever GPT-2 adds), not just the addition -- confirmed empirically
    that a long seed_text left at the 100-token default leaves no room to
    actually generate anything (the result comes back unchanged), so a
    non-trivial seed_text needs a correspondingly larger budget."""
    _quiet_transformers_logging()
    generator = MagicPromptGenerator(
        model_name=model_name or DEFAULT_GPT2_MODEL, seed=seed, device=device, max_prompt_length=max_prompt_length
    )
    _disable_conflicting_max_new_tokens_default(generator)
    return generator.generate(seed_text, num_images=1)[0].strip()


def _quiet_transformers_logging() -> None:
    """Silences transformers' own logger (a generation_config/max_length
    deprecation notice, a pad_token_id deprecation notice, a BPE
    clean_up_tokenization_spaces notice, an attention-mask/truncation
    notice that per its own source only matters for batch_size>1, and a
    "pipelines sequentially on GPU" throughput hint) down to ERROR -- all
    noise here, since every call is a single seed (batch_size=1).

    Deliberately imports transformers itself rather than waiting for
    MagicPromptGenerator to do it lazily: transformers.utils.logging only
    configures the "transformers" logger's level the *first* time
    transformers is imported (guarded, so later calls are no-ops), and
    resets it to its own default regardless of any level set on that
    logger beforehand -- so setting the level has to happen as part of
    (or after) that first import, not before it. Doing it here, before
    MagicPromptGenerator's first pipeline construction, is what actually
    catches the pad_token_id notice that construction itself emits (a
    fixup applied *after* construction, like the max_new_tokens one below,
    would already be too late for that one). Safe to import unconditionally
    here since this function only ever runs once --gpt2-mode is already in
    use, at which point the optional transformers/torch dependency is
    already required."""
    import transformers

    transformers.utils.logging.set_verbosity_error()


def _disable_conflicting_max_new_tokens_default(generator: MagicPromptGenerator) -> None:
    """Works around a dynamicprompts/transformers interaction: the
    underlying HF text-generation pipeline sets its own max_new_tokens=256
    default at construction time, and dynamicprompts never clears it
    before passing max_length per call -- transformers then warns "Both
    max_new_tokens and max_length seem to have been set" and
    max_new_tokens wins. Confirmed empirically: with the default
    max_prompt_length=100, drafts still ran 56-60 words (well past 100
    tokens), i.e. max_prompt_length wasn't actually capping total length.
    Clearing the pipeline's own default lets max_length -- and so
    max_prompt_length/generate_gpt2_nudged_draft's length budgeting --
    actually govern generation length, as documented."""
    pipeline = getattr(generator, "_generator", None)
    if pipeline is not None:
        pipeline.generation_config.max_new_tokens = None


def generate_gpt2_draft(
    seed: int,
    model_name: str | None = None,
    seed_text: str = "",
    max_prompt_length: int | None = None,
    device: str | None = None,
    max_attempts: int = 10,
) -> str:
    """The shared draft step behind both --gpt2-mode seed (bare, seed_text=
    "") and --gpt2-mode expand (nudged by a resolved template, see
    generate_gpt2_nudged_draft) -- they differ only in whether a nudge
    template gets resolved into seed_text first.

    A *non-empty* seed_text can never come back empty: MagicPromptGenerator's
    own clean-up keeps it as the result's prefix regardless of whether GPT-2
    generates a continuation on top of it, so callers with a real nudge
    don't need the retry below. A bare/empty seed_text has nothing to fall
    back on that way -- an empty draft substitutes into template_gpt2.md as
    "Raw draft: " with nothing after it, and the synthesis LLM predictably
    responds asking for the draft instead of writing a prompt. Confirmed
    empirically at a ~33% empty rate for Gustavosta/MagicPrompt-Stable-
    Diffusion on GPU (CPU gives different, and much rarer, empties --
    transformers' RNG stream differs by device) -- far too common to leave
    unhandled.

    Retries generate_gpt2_seed_text with a perturbed seed -- offset by a
    large prime each attempt, wrapped to stay a valid transformers/numpy
    seed -- until a non-empty draft comes back, bounded so a persistently
    empty model doesn't retry forever. The perturbation is deterministic,
    so a given outer seed always retries the same way. Raises rather than
    returning the empty draft once attempts run out -- silently passing
    it through would reproduce the exact bug this exists to avoid, just
    at a lower (roughly 1-in-60000, at the measured 33% empty rate) rate."""
    kwargs = {} if max_prompt_length is None else {"max_prompt_length": max_prompt_length}
    if seed_text:
        return generate_gpt2_seed_text(seed, model_name, seed_text=seed_text, device=device, **kwargs)

    for attempt in range(max_attempts):
        draft = generate_gpt2_seed_text((seed + attempt * 999_999_937) % 2**32, model_name, device=device, **kwargs)
        if draft:
            return draft
    raise Gpt2DraftUnavailable(
        f"GPT-2 draft came back empty {max_attempts} times in a row for seed {seed} -- "
        "try a different --gpt2-model, or rerun with a different --seed"
    )


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
    is a fresh GPT-2 draft instead of a draw from its own wildcard pool --
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
    a normal draw from the subject pool for this seed, rather than injecting
    the empty string and triggering that hang."""
    from .llm_expander import TEMPLATE_PATH, WILDCARDS_DIR

    template = Path(template_path or TEMPLATE_PATH).read_text(encoding="utf-8")
    wildcards_dir = Path(wildcards_dir or WILDCARDS_DIR)
    gpt2_draft = generate_gpt2_seed_text(seed, gpt2_model_name, device=gpt2_device)
    overrides = {"subject": [gpt2_draft]} if gpt2_draft else {}

    wildcard_manager = WildcardManager(root_map={"": [wildcards_dir, overrides]})
    generator = RandomPromptGenerator(wildcard_manager=wildcard_manager, seed=seed)
    return generator.generate(template, num_images=1)[0].strip()


def generate_gpt2_nudged_draft(
    seed: int,
    gpt2_model_name: str | None = None,
    nudge_template_path: Path | str | None = None,
    wildcards_dir: Path | str | None = None,
    gpt2_device: str | None = None,
) -> str:
    """The reverse direction from build_axis_message_with_gpt2_subject:
    resolves a dynamicprompts template (default: GPT2_NUDGE_TEMPLATE_PATH,
    the medium/palette/mood line -- "Medium: X. Palette: Y. Mood: Z." --
    resolved via our own llm_expander.build_user_message, same mechanism
    template.md itself uses) and lets GPT-2 continue/elaborate on the
    *entire* result as its seed text.

    Deliberately a dedicated template file, not a slice of template.md --
    an earlier version resolved the full six-axis template.md and sliced
    out what it assumed was the first paragraph (axis_text.split("\\n\\n",
    1)[0]) to drop the trailing synthesis instructions before handing the
    rest to GPT-2. Confirmed broken for any --template that isn't shaped
    exactly like the packaged default: a custom template with no blank
    line between axis material and instructions (or instructions first)
    got that split wrong, leaking synthesis instructions into GPT-2's
    seed_text. A dedicated nudge template has no instructions paragraph
    mixed into the same file to slice around in the first place, so
    nothing needs guessing -- the whole resolved result is nudge material,
    by construction.

    Deliberately two explicit steps in our own code (resolve, then feed
    to GPT-2), not via dynamicprompts' own MagicPromptGenerator(prompt_
    generator=RandomPromptGenerator(...)) chaining (which the library's
    README shows as the standard way to do this) -- that would hand
    resolution over to a generator instance MagicPromptGenerator
    constructs and owns internally, where here build_user_message (and
    whatever no-repeat/override behavior it may gain later) stays fully
    ours.

    `max_prompt_length` is sized off the resolved nudge text's own length
    (len(text)//3 + 80 -- a rough chars-per-token estimate, plus a fixed
    continuation budget) rather than generate_gpt2_seed_text's 100-token
    default, which leaves no room for GPT-2 to add anything once the
    nudge text alone is non-trivial -- confirmed empirically."""
    from .llm_expander import build_user_message

    nudge_text = build_user_message(seed, nudge_template_path or GPT2_NUDGE_TEMPLATE_PATH, wildcards_dir)
    max_prompt_length = len(nudge_text) // 3 + 80
    return generate_gpt2_draft(
        seed, gpt2_model_name, seed_text=nudge_text, max_prompt_length=max_prompt_length, device=gpt2_device
    )
