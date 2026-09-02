# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""
eikalea: seed -> LLM-invented image prompt -> optional image via ComfyUI
--------------------------------------------------------------------------
1. A seed asks an LLM, over any OpenAI-compatible chat completions server
   (Ollama by default, via `ollama serve`, but LM Studio/vLLM/etc. work too),
   to invent a complete image concept and write it as one natural-language
   paragraph (see llm_expander.py / expander_system_prompt.txt).
2. Optionally (--generate-image), the prompt is fed into a saved ComfyUI
   workflow using uncomfymcp's own client/patcher
   (https://github.com/aschet/uncomfymcp) -- the same library behind the
   `uncomfymcp` MCP server, reused here directly rather than reimplemented.
   It resolves the workflow ComfyUI has saved, converts it from UI to API
   format, patches in this generation's prompt and seed, submits it, and
   downloads the resulting image. ComfyUI must already be running (default
   http://127.0.0.1:8188) with the named workflow saved and its models
   installed.

Setup:
    pip install -r requirements.txt   # pins uncomfymcp to a release tag, not main
    ollama pull qwen3.6:35b           # needs Ollama installed and running -- https://ollama.com

Run:
    eikalea --count 20 --model qwen3.6:35b                          # prompts only, printed to stdout
    eikalea --count 20 --model qwen3.6:35b --out prompts.jsonl      # ...and saved as JSONL
    eikalea --count 3 --model qwen3.6:35b --generate-image Krea2    # each prompt rendered right
    eikalea --seed 42 --model qwen3.6:35b \\                         # after it's generated, one
        --generate-image "Krea2+Upscale.app"                        # seed at a time
    eikalea --count 20 --model qwen3.6:35b gemma4:26b               # model picked at random per seed
    eikalea --count -1 --model qwen3.6:35b                          # run until interrupted (Ctrl+C)
    eikalea --count -1 --model qwen3.6:35b --generate-image Krea2   # ...same, but rendering each too
    eikalea --count 20 --model qwen3.6:35b --json | jq .prompt       # pipeable JSONL on stdout

    # Replay a saved prompt list (as written by --out) instead of generating
    # fresh via the LLM -- the backend is never touched in this mode, so
    # --model is not needed, but --generate-image is required (otherwise
    # this would just print back what's already in the file):
    eikalea --in prompts.jsonl --generate-image Krea2
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from .llm_expander import generate, pick_model, unload_ollama_model

# Matches uncomfymcp's own ComfyClient/server defaults exactly (comfy.py,
# server.py) -- no default workflow name, since uncomfymcp itself never
# assumes one either; the caller must always name it explicitly.
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_TIMEOUT = 300.0


async def generate_image(
    prompt: str, seed: int, workflow_name: str, comfy_url: str, timeout: float, out_path: str
) -> str:
    """Run `workflow_name` on ComfyUI with this prompt/seed patched in, save the first image."""
    from uncomfymcp import sources
    from uncomfymcp import workflow as wf
    from uncomfymcp.comfy import ComfyClient, ComfyError

    client = ComfyClient(comfy_url, timeout=timeout)
    found = await sources.resolve(workflow_name, client)
    patched = wf.patch(found.graph, prompt=prompt, seed=seed)
    # Embeds the workflow into the PNG's `workflow` chunk so it can be
    # dragged back into ComfyUI -- same as the uncomfymcp MCP server itself
    # (returns None when it can't be proven to describe this exact
    # generation, in which case the PNG just carries no workflow chunk).
    embed = await sources.patched_ui(client, found, prompt, seed)
    refs = await client.generate(patched, embed)
    if not refs:
        raise ComfyError("The workflow ran but produced no images.")

    data = await client.fetch(refs[0])
    Path(out_path).write_bytes(data)
    return out_path


def load_prompts_jsonl(path: str) -> list[tuple[int, str]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append((obj["seed"], obj["prompt"]))
    return records


def save_prompts(records: list[tuple[int, str]], out_path: str) -> None:
    with open(out_path, "a", encoding="utf-8") as f:
        for seed, prompt in records:
            f.write(json.dumps({"seed": seed, "prompt": prompt}) + "\n")


def print_status(message: str, *, as_json: bool) -> None:
    """Progress/status messages -- kept off stdout when --json is set, so
    stdout stays pure, pipeable JSONL (one object per generated prompt)."""
    print(message, file=sys.stderr if as_json else sys.stdout)


def print_prompt_header(seed: int, *, as_json: bool, progress: str, model: str | None = None) -> None:
    """Printed before the prompt itself is known -- generation can take a
    while, so this gives immediate feedback that a new seed has started
    rather than going silent until it finishes. Not applicable to --json:
    there, the whole record is a single line emitted once it's complete."""
    if as_json:
        return
    header = f"\n{progress} with seed {seed}"
    if model is not None:
        header += f" and model {model}"
    print(header + ":")


def print_prompt_body(seed: int, prompt: str, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"seed": seed, "prompt": prompt}))
    else:
        print()
        print(prompt)


def unique_output_path(outdir: str, seed: int) -> str:
    """seed_{seed}.png, or seed_{seed}_2.png / _3.png / ... if that's
    already taken -- across separate runs the same seed can come up more
    than once (same --seed passed twice, or two random starting seeds
    happening to land on it), and overwriting an earlier render of that
    seed with no warning would silently lose it."""
    base = Path(outdir) / f"seed_{seed}.png"
    if not base.exists():
        return str(base)
    n = 2
    while True:
        candidate = Path(outdir) / f"seed_{seed}_{n}.png"
        if not candidate.exists():
            return str(candidate)
        n += 1


def run_streaming(args: argparse.Namespace, limit: int | None) -> None:
    """Generate (and, if --generate-image is set, render) one seed at a
    time. `limit=None` runs until interrupted (Ctrl+C); otherwise stops
    after `limit` seeds.

    Used whenever --generate-image is set, regardless of --count: without
    interleaving, a batch-then-render design means a large --count renders
    no images at all until every prompt in the whole batch has finished
    generating first, which for hundreds of prompts is indistinguishable
    from image generation being broken. The tradeoff is that (unless
    --no-unload is set) the backend is unloaded before every render instead
    of once at the end, since generation and rendering interleave here --
    there's no single point where it's done being needed."""
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    if limit is None:
        print_status("Generating until interrupted -- press Ctrl+C to stop.", as_json=args.json)
    i = 0
    try:
        while limit is None or i < limit:
            model = pick_model(seed, args.model)
            progress = f"{i + 1}/{limit}" if limit is not None else str(i + 1)
            print_prompt_header(
                seed, as_json=args.json, progress=progress,
                model=model if len(args.model) > 1 else None,
            )

            prompt = generate(seed, models=args.model, host=args.api_host)
            print_prompt_body(seed, prompt, as_json=args.json)

            if args.out:
                save_prompts([(seed, prompt)], args.out)

            if args.generate_image:
                if not args.no_unload:
                    unload_ollama_model(model, host=args.api_host)
                out_path = unique_output_path(args.outdir, seed)
                asyncio.run(generate_image(
                    prompt, seed, args.generate_image, args.comfy_url, args.timeout, out_path
                ))
                print_status("", as_json=args.json)
                print_status(f"saved: {out_path}", as_json=args.json)

            seed += 1
            i += 1
    except KeyboardInterrupt:
        print_status("Stopped.", as_json=args.json)


def main():
    parser = argparse.ArgumentParser(
        description="Seed -> LLM-invented image prompt -> optional image via ComfyUI.",
        epilog="Requires an OpenAI-compatible chat completions server (e.g. Ollama, via `ollama "
               "serve`) reachable at --api-host. --generate-image additionally requires a running "
               "ComfyUI instance (at --comfy-url) with the named workflow already saved.",
    )
    parser.add_argument(
        "--count", type=int, default=1,
        help="How many prompts to generate. A negative value (e.g. -1) runs until interrupted "
             "(Ctrl+C) instead.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Fixed starting seed (omit for random each run).")
    parser.add_argument(
        "--model", type=str, nargs="+", default=None,
        help="One or more model names. Required unless --in is used (the backend is never "
             "touched in that mode). With more than one, a model is picked at random per seed "
             "(same seed -> same model, like every other axis).",
    )
    parser.add_argument(
        "--api-host", type=str, default="http://localhost:11434",
        help="Base URL of any OpenAI-compatible chat completions server (Ollama by default).",
    )
    parser.add_argument(
        "--in", dest="in_path", type=str, default=None, metavar="FILE",
        help="Read {\"seed\": ..., \"prompt\": ...} records from this JSONL file (as written by "
             "--out) instead of generating fresh via the LLM. --count/--seed/--model/--api-host "
             "are ignored in this mode; the LLM backend is never touched. Requires "
             "--generate-image -- without it there's nothing to do with the loaded prompts.",
    )
    parser.add_argument(
        "--out", type=str, default=None, metavar="FILE",
        help="Append generated prompts to this file, as JSONL ({\"seed\": ..., \"prompt\": ...} "
             "per line).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print each prompt to stdout as a JSON line ({\"seed\": ..., \"prompt\": ...}) "
             "instead of the human-readable block. Status/progress messages move to stderr so "
             "stdout stays pure, pipeable JSONL.",
    )

    parser.add_argument(
        "--generate-image", type=str, default=None, metavar="WORKFLOW",
        help="Also render each prompt via ComfyUI, using this workflow name (as ComfyUI's "
             "workflow list shows it).",
    )
    parser.add_argument(
        "--no-unload", action="store_true",
        help="Don't evict the model from VRAM before each image render. Skip this only if your "
             "GPU has enough VRAM to hold both the LLM and ComfyUI's models at once -- it avoids "
             "the reload cost between prompts.",
    )
    parser.add_argument("--comfy-url", type=str, default=DEFAULT_COMFY_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                         help="Seconds to wait for a generation before giving up.")
    parser.add_argument("--outdir", type=str, default="./eikalea_outputs")

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if not args.in_path and args.model is None:
        parser.error("--model is required unless --in is used")

    if args.in_path and not args.generate_image:
        parser.error("--generate-image is required when using --in (otherwise there's nothing "
                      "to do -- the prompts are already generated)")

    if args.generate_image:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if not args.in_path and (args.count < 0 or args.generate_image):
        run_streaming(args, limit=None if args.count < 0 else args.count)
        return

    if args.in_path:
        records = load_prompts_jsonl(args.in_path)
        for i, (seed, final_prompt) in enumerate(records):
            progress = f"{i + 1}/{len(records)}"
            print_prompt_header(seed, as_json=args.json, progress=progress)
            print_prompt_body(seed, final_prompt, as_json=args.json)
    else:
        start_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
        # Generate every prompt first, while the backend is still warm --
        # unloading it between each generation (as a naive interleaved loop
        # would) forces a full reload of a 24GB+ model from disk on every
        # following iteration. Still printed one at a time as each finishes
        # (not batched up silently) so a large --count shows live progress
        # instead of going quiet for however long the whole batch takes.
        records = []
        for i, seed in enumerate(start_seed + offset for offset in range(args.count)):
            model = pick_model(seed, args.model) if len(args.model) > 1 else None
            print_prompt_header(seed, as_json=args.json, progress=f"{i + 1}/{args.count}", model=model)
            final_prompt = generate(seed, models=args.model, host=args.api_host)
            records.append((seed, final_prompt))
            print_prompt_body(seed, final_prompt, as_json=args.json)

    if args.out:
        save_prompts(records, args.out)

    if args.generate_image:
        # Only replay mode (--in) reaches here -- fresh generation combined
        # with --generate-image is handled by run_streaming() above and
        # returns before this point. The backend is never touched in replay
        # mode, so there's nothing to unload.
        for seed, final_prompt in records:
            out_path = unique_output_path(args.outdir, seed)
            asyncio.run(generate_image(
                final_prompt, seed, args.generate_image, args.comfy_url, args.timeout, out_path
            ))
            print_status("", as_json=args.json)
            print_status(f"saved: {out_path}", as_json=args.json)


if __name__ == "__main__":
    main()
