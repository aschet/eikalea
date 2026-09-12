# eikalea

[![AI-SLOP](https://img.shields.io/badge/AI-SLOP-yellow?style=flat-square)](https://en.wikipedia.org/wiki/AI_slop)

eikalea is an experimental, autonomous art generator inspired by the
infinite monkey theorem. It draws randomly from a set of predefined pools,
synthesizes the result into a cohesive prompt via an LLM, and optionally
renders it through an existing ComfyUI workflow. The longer-term vision is
an installation: viewers experience a continuous stream of unique,
generated artworks at intervals, and the growing archive of images can be
browsed, displayed at random, or rotated across multiple screens.

*The name "eikalea" blends Greek εἰκών (eikōn — image, likeness; root of "icon") with Latin alea (chance, dice; root of "aleatoric" — fittingly, "governed by chance").*

<table align="center">
  <tr>
    <td><img src=".github/images/example_1.webp" width="300"></td>
    <td><img src=".github/images/example_2.webp" width="300"></td>
  </tr>
  <tr>
    <td><img src=".github/images/example_3.webp" width="300"></td>
    <td><img src=".github/images/example_4.webp" width="300"></td>
  </tr>
</table>

## Requirements

- Python 3.10+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running, with the workflow you want to render already saved (only needed if you want images, not just prompts). Rendering goes through [uncomfymcp](https://github.com/aschet/uncomfymcp)'s own workflow-patching logic, so its limitations apply here too.
- [Ollama](https://ollama.com) running with a model pulled (e.g. `nemotron-3.5-lightning:30b`), or any other OpenAI-compatible chat completions endpoint -- larger models synthesize noticeably more coherent, specific prompts; small models tend toward generic or muddled results

## Installation

Linux:

```bash
git clone https://github.com/aschet/eikalea.git
cd eikalea
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Windows:

```bash
git clone https://github.com/aschet/eikalea.git
cd eikalea
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

`pip install -e ".[gpt2]"` instead, needed only for `--gpt2-mode` (see [below](#seeding-from-a-gpt-2-fine-tune)).

## Usage

Generate prompts only, printed to stdout:

```bash
eikalea --count 20 --model nemotron-3.5-lightning:30b
```

Save them to a file too (always written as JSONL):

```bash
eikalea --count 20 --model nemotron-3.5-lightning:30b --out prompts.jsonl
```

Also render each prompt into an image, via a workflow already saved in ComfyUI:

```bash
eikalea --count 20 --model nemotron-3.5-lightning:30b --comfy-workflow Krea2
```

`generate` is the default command, so the examples above also work as `eikalea generate --count 20 ...`. Replaying a saved prompt list uses its own `replay` command instead — see `eikalea replay --help`. See `eikalea generate --help` for everything else — multiple models, run-until-interrupted mode, custom Ollama/ComfyUI hosts, and more.

## How prompts are built

Each seed primes six axes — medium, composition, subject, palette, mood, and art movement — then asks the LLM to invent one concept that unifies all six into a single prompt. The axes live in `wildcards.yaml` (medium is grouped into subgroups like painting/prints/drawing; the rest are flat lists), assembled by a template and resolved via [dynamicprompts](https://github.com/adieyal/dynamicprompts), the same templating library behind the `sd-dynamic-prompts` extension for AUTOMATIC1111/ComfyUI. Both are packaged defaults, but fully replaceable:

```bash
# Get an editable copy of the packaged template + wildcard files
eikalea templates export ./my-templates

# edit ./my-templates/template.md and the files under
# ./my-templates/wildcards/ -- add, remove, or rename axes freely

# Check it resolves cleanly and see an example output, before spending an LLM call on it
eikalea templates validate --template ./my-templates/template.md --wildcards-dir ./my-templates/wildcards

eikalea --count 20 --model nemotron-3.5-lightning:30b \
    --template ./my-templates/template.md \
    --wildcards-dir ./my-templates/wildcards
```

The system prompt (which governs output format — full sentences rather than tags, no camera/quality boilerplate) stays fixed and axis-independent, so a template override never desyncs from it.

Keep the template file out of the wildcards directory: `dynamicprompts` treats every `.txt`/`.json`/`.yaml` file under `--wildcards-dir` as its own wildcard collection, so a template file dropped in there would show up as a spurious, unused axis. `template.md`'s `.md` extension is deliberately outside that set, but it still shouldn't live inside `wildcards/`.

### Seeding from a GPT-2 fine-tune

`--gpt2-mode {seed,subject}` brings a small GPT-2 model fine-tuned on old-style Stable Diffusion tag prompts (default: [`Gustavosta/MagicPrompt-Stable-Diffusion`](https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion)) into the pipeline: `seed` replaces the six-axis template entirely with a GPT-2 draft, continued from a resolved nudge template; `subject` keeps the six-axis template but lets GPT-2 supply just the subject axis.

`seed`'s nudge (`--gpt2-nudge-template`) is a dynamicprompts template, resolved against `--wildcards-dir` like `--template`, that GPT-2 continues from — default: a packaged medium/palette/mood line (`template_gpt2_nudge.md`, see `eikalea templates export`). Confirmed empirically to matter: a bare/empty nudge gives GPT-2 far less varied drafts (19 duplicates out of 100 consecutive seeds) than a real one (0 duplicates over the same range). Pass `--gpt2-nudge-template` to use a different template instead — a lighter one, the full six-axis line, or (pointed at an empty file) no nudge at all for pure free-association.

```bash
eikalea --count 20 --model nemotron-3.5-lightning:30b --gpt2-mode seed
```

Requires `pip install -e ".[gpt2]"`. Runs on GPU by default; pass `--gpt2-cpu` on tight VRAM budgets. See `eikalea generate --help` for `--gpt2-model`/`--gpt2-template`/`--gpt2-nudge-template`.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e . --group dev
pytest
ruff check .
mypy
```
