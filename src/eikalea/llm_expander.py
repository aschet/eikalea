# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""
LLM-driven prompt generation: seed -> finished image prompt.
----------------------------------------------------------------
Shared by random_prompt_generator.py and krea2_pipeline.py. No hardcoded
subject pool -- the model invents the subject itself per call, steered by
the anti-bias rules in expander_system_prompt.txt (no tag lists, no artist
name-dropping, no default portrait). The seed drives the model's own
sampler so the same seed reproduces the same prompt.

Medium and composition get one exception each: every call is stateless (no
memory of what a previous call picked), and a vague instruction like "vary
the medium" or "vary the composition" gets ignored in practice -- the model
just settles into a habitual default (oil painting; a centered, tightly
arranged still life) across a whole batch. So MEDIA_TECHNIQUES and
COMPOSITIONS below each name one specific option per seed and state it
directly in the user turn. The model still invents everything else --
subject, lighting, palette -- itself.

Talks to any OpenAI-compatible chat completions server (Ollama by default,
via `ollama serve`, but also LM Studio/vLLM/etc. if pointed at one) over
the standard /v1/chat/completions endpoint, using the request-level "seed"
field for reproducibility.
"""

import random
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT_PATH = Path(__file__).parent / "expander_system_prompt.txt"

# Specific named techniques, not vague categories -- a bucket like "painting"
# just lets the model default to oil painting every time. All media where the
# artwork itself IS a flat 2D image (paint/ink/toner directly on a page or
# canvas). No photography, 3D/CGI rendering, sculpture, or physical/textured
# craft media (embroidery, stained glass, mosaic, tapestry, etc.) -- a picture
# "of" any of those collapses back into a photo of a real physical object.
MEDIA_TECHNIQUES: list[str] = [
    "oil painting",
    "watercolor painting",
    "gouache painting",
    "acrylic painting",
    "ink wash painting",
    "woodcut print",
    "linocut print",
    "etching",
    "engraving",
    "lithograph",
    "screen print",
    "risograph print",
    "charcoal drawing",
    "graphite pencil drawing",
    "pastel drawing",
    "colored pencil drawing",
    "pen and ink drawing",
    "batik textile art",
    "paper collage",
    "mixed-media assemblage",
    "vector illustration",
    "editorial illustration",
    "technical or scientific illustration",
]


# Specific named compositions/framings, for the same reason as MEDIA_TECHNIQUES
# above -- "vary composition" as a vague instruction just gets ignored, and
# the model keeps defaulting to a centered, tightly-arranged still life.
COMPOSITIONS: list[str] = [
    # Grounded in the standard canon of composition/design principles (rule
    # of thirds, golden ratio, balance, rhythm, perspective types) rather
    # than invented from scratch.
    "a composition built on the rule of thirds, with the focal point at one of the four intersection points",
    "a composition built on the golden spiral, with visual weight concentrated at its inner curl",
    "radial balance, with every element arranged symmetrically around a central point",
    "asymmetrical balance, where a large simple area offsets a small complex one",
    "strong diagonal leading lines drawing the eye toward a single focal point",
    "a natural frame-within-a-frame surrounding the subject, such as a doorway, arch, or overhanging branches",
    "distinct layered depth of foreground, midground, and background",
    "dominant negative space around one small, isolated focal point",
    "a regular, rhythmic repetition of a single motif across the frame",
    "strict one-point linear perspective converging on a single vanishing point",
    "two-point perspective with two converging vanishing points",
    "isometric projection with no vanishing point at all",
    "atmospheric perspective, with distant elements fading in contrast and clarity as they recede",
    "an extreme close-up crop isolating texture and detail",
    "an extremely wide establishing view",
    "a top-down, bird's-eye view",
    "an extreme low-angle, worm's-eye view",
    "a tilted, off-kilter horizon that destabilizes the whole scene",
    "a cutaway or cross-section view",
    "a cluttered, maximalist composition filling the entire frame",
    "a sparse, minimal composition with a single small element",
    "a composition juxtaposing two vastly different scales in the same frame, forcing the eye to reconcile them",
    "a multi-panel composition showing the same subject at several different moments in sequence",
    "a composition built on deliberately conflicting, multiple vanishing points that create spatial disorientation",
    "a nested composition, with a smaller, more detailed scene embedded within a larger, simpler one",
    "a composition where the primary subject is mostly obscured, its presence only inferred from fragments, shadows, or reflections",
    "a composition split by a strong diagonal divide, contrasting two completely different visual densities on either side",
    "a composition built from several overlapping, translucent or transparent planes receding in depth",
    "a composition that mirrors or rhymes two distant elements across the frame, inviting comparison between them",
    "a composition organized as a spiral, drawing the eye from the outer edge inward through several distinct zones",
    "a composition split by an internal threshold or frame, showing two connected but distinct spaces at once",
    "a composition with a deliberately unstable, off-kilter horizon that destabilizes the whole scene",
    "a composition built from a dense, irregular grid of small vignettes, each showing a fragment of a larger scene",
    "a composition where converging forms from multiple directions collide at a single focal point",
    "a composition using extreme foreshortening to compress or distort spatial depth dramatically",
    "a composition where background and foreground are given equal visual weight, read as two competing subjects",
]

# Scenery/subject matter -- the thing actually depicted, as opposed to
# COMPOSITIONS above (which is about framing/viewing angle, not content).
# Without this, the model has a strong attractor toward "subterranean
# library/marketplace, forgotten knowledge, bioluminescent moss" almost
# regardless of seed or medium -- confirmed directly by testing the same
# seeds with and without this prime.
#
# Organized around the two real classification systems this maps to: the
# academic hierarchy of genres (Félibien, 1667 -- history/narrative painting,
# portrait, genre scenes of everyday life, landscape, still life, animal
# painting) and Iconclass's top-level divisions (nature; human being;
# society/civilization/culture; religion and magic; classical mythology;
# abstract/non-representational) for subject matter the genre hierarchy
# alone doesn't cover.
SCENERY: list[str] = [
    # History/narrative painting -- myth, religion, epic, allegory
    "a mythological or folkloric scene, mid-story",
    "a religious or spiritual scene",
    "an allegorical scene where figures personify abstract ideas",
    "a historical or period scene",
    # Portrait -- the human figure. "a single figure or portrait" is repeated
    # here (a lightweight way to weight a flat random.choice pool without a
    # separate weighting mechanism) -- at a flat 1-in-45 it was landing in
    # under 40% of any given 20-generation batch purely from pool growth
    # (SCENERY has grown from 15 to 45 entries), when the actual intent was
    # "occasional, not the default", not "almost never".
    "a single figure or portrait",
    "a single figure or portrait",
    "a single figure or portrait",
    "a single figure or portrait",
    "two figures caught mid-interaction or conversation",
    "a pair of hands engaged in a craft or task",
    "a crowd or group scene going about an activity",
    # Genre scenes -- everyday life, work, society
    "a quiet domestic interior mid-routine",
    "a bustling marketplace or bazaar",
    "a festival, procession, or ceremony",
    "figures in motion during a sport or physical activity",
    "a workbench of tools and instruments",
    "an industrial facility or factory interior",
    "a laboratory or workshop mid-process",
    "a library, archive, or dense shelf of objects",
    "a transportation hub such as a harbor, station, or airfield",
    "a monument, bridge, or piece of civic infrastructure",
    # Landscape -- nature and built environment
    "a sweeping natural landscape",
    "an architectural interior",
    "a dense urban streetscape",
    "a nighttime cityscape",
    "ruins or an abandoned structure reclaimed by nature",
    "a geological formation like a canyon, cave, or mountain range",
    "an arid desert or dune landscape",
    "a coastal shoreline or tidepool scene",
    "the interior of a dense forest",
    "a snow-covered or arctic landscape",
    "a volcanic or geothermal landscape",
    "a cultivated garden or agricultural landscape",
    "a celestial scene of planets, stars, or nebulae",
    "a weather or atmospheric phenomenon",
    "an underwater scene",
    # Still life
    "a still life arrangement of everyday objects",
    "a food or culinary arrangement",
    "a close study of a textile or repeating pattern",
    "a microscopic or crystalline structure magnified large",
    # Animal/botanical painting
    "an animal or creature in its environment",
    "insects or small creatures at macro scale",
    "birds in flight or roosting",
    "a herd or flock of animals moving together",
    "a botanical study of plants or fungi",
    # Abstract/non-representational (Iconclass division 0)
    "an abstract composition of shapes and forms",
    "a surreal, dreamlike juxtaposition of unrelated objects",
    "a machine, vehicle, or piece of engineered equipment",
]

# Color and mood register. Without this, medium/composition/subject can all
# vary while the output still feels repetitive, because the color palette
# and overall mood collapse onto the same "muted, faded, antique" register
# almost every time (sepia, ochre, verdigris, dusty patina) regardless of
# what else is primed -- confirmed directly by the recurring vocabulary
# across otherwise-varied samples.
PALETTES: list[str] = [
    # The first seven are the standard color-harmony schemes from color
    # theory; the rest are concrete, named applications rather than vague
    # mood words.
    "a monochromatic palette in a single hue, varied only by value from near-black to near-white",
    "an analogous palette of three neighboring hues on the color wheel",
    "a complementary palette of two opposing hues pushed to full saturation for maximum contrast",
    "a split-complementary palette pairing one hue with the two neighbors of its opposite",
    "a triadic palette of three hues evenly spaced around the color wheel",
    "a tetradic palette built from two complementary pairs",
    "an achromatic palette of pure black, white, and gray with no hue at all",
    "a warm, golden late-afternoon palette",
    "a cold, blue-toned nocturnal palette",
    "a bright, cheerful, candy-colored palette",
    "a high-contrast palette of pure black with one sharp accent color",
    "a palette of unnatural, fluorescent, artificial colors",
    "a clean, clinical palette of whites and cool grays",
    "a richly saturated palette of deep jewel tones",
    "a bold, flat, poster-like palette with hard color boundaries",
    "a fresh, springlike palette of bright greens and light pinks",
    "an overexposed, sun-bleached palette washed almost to white",
    "a rich, dark palette dominated by deep burgundy and black",
    "a soft, low-saturation pastel palette",
    "an electric neon palette of magenta and cyan",
]

# Emotional/atmospheric register -- distinct from PALETTES above, which is
# color only. Without this, "mood" was just a word tacked onto the palette
# label in the prompt text with nothing actually controlling it, so it
# defaulted to whatever the model's habitual register was regardless of seed.
MOODS: list[str] = [
    "serene and calm",
    "eerie and unsettling",
    "chaotic and frenetic",
    "whimsical and playful",
    "melancholic and wistful",
    "oppressive and claustrophobic",
    "triumphant and heroic",
    "tense and foreboding",
    "dreamlike and disorienting",
    "austere and solemn",
    "joyful and exuberant",
    "quiet and contemplative",
    "menacing and predatory",
    "nostalgic and tender",
    "absurd and comedic",
    "reverent and ceremonial",
    "restless and anxious",
    "detached and clinical",
]

# Real, named art-historical movements and traditions -- not individual
# artists (most current models respond weakly and inconsistently to those,
# and it reintroduces the name-dropping problem), but broadly recognized
# collective styles with well-documented visual conventions. Spans multiple
# eras and cultures rather than just the Western canon.
ART_MOVEMENTS: list[str] = [
    "Art Nouveau, with flowing organic curves and flat decorative color fields",
    "Art Deco, with geometric ornamentation and streamlined symmetry",
    "Bauhaus, with primary colors, geometric abstraction, and functionalist clarity",
    "Constructivism, with bold diagonals, industrial motifs, and a restrained palette",
    "De Stijl, with strict horizontal and vertical lines and primary colors on white",
    "Cubism, with fragmented, multi-angle simultaneous views of the same form",
    "Futurism, with dynamic lines suggesting speed, motion, and mechanical energy",
    "Suprematism, with pure geometric forms floating in undefined space",
    "Fauvism, with wildly non-naturalistic, high-intensity color",
    "1960s psychedelic poster art, with swirling, saturated, hand-lettered forms",
    "the Arts and Crafts movement, with stylized natural motifs and handcrafted texture",
    "Pointillism, built entirely from small distinct dots of unmixed color",
    "the Vienna Secession, with ornamental gilded patterning and elongated figures",
    "Byzantine icon painting, with gold ground, frontal figures, and no cast shadow",
    "Ancient Egyptian frieze convention, with profile heads and frontal shoulders",
    "Scandinavian rosemaling folk-painting tradition, with flowing floral scrollwork",
    "Impressionism, with loose visible brushstrokes capturing fleeting light and atmosphere",
    "Post-Impressionism, with structured brushwork and symbolic, non-naturalistic color",
    "German Expressionism, with jagged distorted forms and raw, unnatural color",
    "Surrealism, with dreamlike juxtaposition rendered in precise, illusionistic detail",
    "Dada, with absurdist collage and a deliberate rejection of conventional beauty",
    "Pop Art, with bold flat color, hard outlines, and mass-media imagery",
    "Abstract Expressionism, with large-scale gestural, non-representational mark-making",
    "Minimalism, with radically reduced form and industrial materials",
    "Op Art, with optical illusions built from precise repeating geometric patterns",
    "Precisionism, with crisp geometric clarity applied to industrial and architectural subjects",
    "American Regionalism, with stylized, idealized rural scenes and smooth modeling",
    "Social Realism, with unidealized depictions of labor and everyday hardship",
    "Naive art, with flattened perspective, bright color, and untrained directness",
    "Romanticism, with dramatic lighting and turbulent, emotionally charged scenes",
    "Neoclassicism, with clean line, idealized anatomy, and restrained composition",
    "Baroque, with dramatic chiaroscuro and dynamic diagonal composition",
    "Rococo, with pastel colors, ornate curves, and playful lightness",
    "Northern Renaissance, with meticulous detail and symbolic realism",
    "Pre-Raphaelite, with jewel-toned color and intricate naturalistic detail",
    "Symbolism, with dreamlike imagery expressing inner psychological states",
    "Soviet socialist realism, with heroic idealized figures serving a collective narrative",
    "Memphis Design, with clashing pastel geometry and playful postmodern pattern",
    "Celtic illumination, with interlacing knotwork and dense decorative borders",
]

# Offsets so medium/composition/scenery/palette/mood/movement/model aren't
# derived from the same draw (would otherwise correlate the picks for a
# given seed).
_COMPOSITION_SEED_OFFSET = 999_999_937
_SCENERY_SEED_OFFSET = 1_999_999_943
_PALETTE_SEED_OFFSET = 2_999_999_753
_MOOD_SEED_OFFSET = 3_999_999_121
_MOVEMENT_SEED_OFFSET = 4_999_999_811
_MODEL_SEED_OFFSET = 5_999_999_789


def pick_medium(seed: int) -> str:
    return random.Random(seed).choice(MEDIA_TECHNIQUES)


def pick_composition(seed: int) -> str:
    return random.Random(seed + _COMPOSITION_SEED_OFFSET).choice(COMPOSITIONS)


def pick_scenery(seed: int) -> str:
    return random.Random(seed + _SCENERY_SEED_OFFSET).choice(SCENERY)


def pick_palette(seed: int) -> str:
    return random.Random(seed + _PALETTE_SEED_OFFSET).choice(PALETTES)


def pick_mood(seed: int) -> str:
    return random.Random(seed + _MOOD_SEED_OFFSET).choice(MOODS)


def pick_art_movement(seed: int) -> str:
    return random.Random(seed + _MOVEMENT_SEED_OFFSET).choice(ART_MOVEMENTS)


def pick_model(seed: int, models: list[str]) -> str:
    """When more than one model is given, pick one per seed (same seed ->
    same model, like every other axis) rather than always using the first."""
    return random.Random(seed + _MODEL_SEED_OFFSET).choice(models)


def build_user_message(seed: int) -> str:
    medium = pick_medium(seed)
    composition = pick_composition(seed)
    scenery = pick_scenery(seed)
    palette = pick_palette(seed)
    mood = pick_mood(seed)
    movement = pick_art_movement(seed)
    return (
        f"Medium: {medium}. Composition: {composition}. Subject: {scenery}. "
        f"Palette: {palette}. Mood: {mood}. Art movement/tradition: {movement}. "
        f"Invent one deliberate concept that unifies all six."
    )


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def generate_with_llm(seed: int, system_prompt: str, model: str, host: str = "http://localhost:11434") -> str:
    """Uses the standard OpenAI-compatible /v1/chat/completions surface, via
    the official `openai` client rather than a hand-rolled request, so this
    works unmodified against any OpenAI-compatible server (Ollama, LM
    Studio, vLLM, etc.) pointed at via `host`. "reasoning_effort": "none"
    (passed via extra_body -- it's not part of the openai SDK's own request
    schema) is the field that actually disables hidden chain-of-thought --
    Ollama's own native "think": false is not honored on this endpoint
    (confirmed empirically: qwen3.5/qwen3.6/gemma4 all kept reasoning with
    "think": false, all stopped with "reasoning_effort": "none", going from
    ~20-30s/prompt to <1s)."""
    client = OpenAI(base_url=f"{host}/v1", api_key="not-needed", timeout=120)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(seed)},
        ],
        seed=seed,
        extra_body={"reasoning_effort": "none"},
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


def generate(seed: int, models: list[str], host: str = "http://localhost:11434") -> str:
    system_prompt = load_system_prompt()
    model = pick_model(seed, models)
    return generate_with_llm(seed, system_prompt, model=model, host=host)
