# eikalea

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
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running, with the workflow you want to render already saved (only needed if you want images, not just prompts)
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

Each seed primes six axes — medium, composition, subject, palette, mood, and art movement — then asks the LLM to invent one concept that unifies all six into a single prompt. The axes are wildcard text files (one option per line) assembled by a template, resolved via [dynamicprompts](https://github.com/adieyal/dynamicprompts), the same templating library behind the `sd-dynamic-prompts` extension for AUTOMATIC1111/ComfyUI. Both are packaged defaults, but fully replaceable:

```bash
# Get an editable copy of the packaged template + wildcard files
eikalea templates export ./my-templates

# edit ./my-templates/template.md and the .txt files under
# ./my-templates/wildcards/ -- add, remove, or rename axes freely

# Check it resolves cleanly and see an example output, before spending an LLM call on it
eikalea templates validate --template ./my-templates/template.md --wildcards-dir ./my-templates/wildcards

eikalea --count 20 --model nemotron-3.5-lightning:30b \
    --template ./my-templates/template.md \
    --wildcards-dir ./my-templates/wildcards
```

The system prompt (which governs output format — full sentences rather than tags, no camera/quality boilerplate) stays fixed and axis-independent, so a template override never desyncs from it.

Keep the template file out of the wildcards directory: `dynamicprompts` treats every `.txt`/`.json`/`.yaml` file under `--wildcards-dir` as its own wildcard collection, so a template file dropped in there would show up as a spurious, unused axis. `template.md`'s `.md` extension is deliberately outside that set, but it still shouldn't live inside `wildcards/`.
