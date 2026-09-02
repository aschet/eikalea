# eikalea

eikalea is an experimental, autonomous art generator inspired by the
infinite monkey theorem. It draws randomly from a set of predefined pools,
synthesizes the result into a cohesive prompt via an LLM, and optionally
renders it through an existing ComfyUI workflow. The longer-term vision is
an installation: viewers experience a continuous stream of unique,
generated artworks at intervals, and the growing archive of images can be
browsed, displayed at random, or rotated across multiple screens.

<table align="center">
  <tr>
    <td><img src=".github/images/exampe_1.webp" width="300"></td>
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
- [Ollama](https://ollama.com) running with a model pulled (e.g. `qwen3.6:35b`), or any other OpenAI-compatible chat completions endpoint

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
eikalea --count 20 --model qwen3.6:35b
```

Save them to a file too (always written as JSONL):

```bash
eikalea --count 20 --model qwen3.6:35b --out prompts.jsonl
```

Also render each prompt into an image, via a workflow already saved in ComfyUI:

```bash
eikalea --count 20 --model qwen3.6:35b --generate-image Krea2
```

See `eikalea --help` for everything else — multiple models, run-until-interrupted mode, replaying a saved prompt list, custom Ollama/ComfyUI hosts, and more.
