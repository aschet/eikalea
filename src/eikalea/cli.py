# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

"""
eikalea: seed -> LLM-synthesized image prompt -> optional image via ComfyUI
--------------------------------------------------------------------------
1. A seed primes six axes (medium, composition, subject, palette, mood, art
   movement) from wildcard files via a template, then asks an LLM, over any
   OpenAI-compatible chat completions server (Ollama by default, via
   `ollama serve`, but LM Studio/vLLM/etc. work too), to synthesize them
   into one unified concept and write it as one natural-language paragraph
   (see llm_expander.py / template.md / expander_system_prompt.txt).
2. Optionally (--generate-image), the prompt is fed into a saved ComfyUI
   workflow using uncomfymcp's own client/patcher
   (https://github.com/aschet/uncomfymcp) -- the same library behind the
   `uncomfymcp` MCP server, reused here directly rather than reimplemented.
   It resolves the workflow ComfyUI has saved, converts it from UI to API
   format, patches in this generation's prompt and seed, submits it, and
   downloads the resulting image. ComfyUI must already be running (default
   http://127.0.0.1:8188) with the named workflow saved and its models
   installed.

`generate` is the default command -- `eikalea --count 20 --model X` and
`eikalea generate --count 20 --model X` are equivalent.

Setup:
    pip install -r requirements.txt              # pins uncomfymcp to a release tag, not main
    ollama pull nemotron-3.5-lightning:30b       # needs Ollama installed and running -- https://ollama.com

Run:
    eikalea --count 20 --model nemotron-3.5-lightning:30b                        # prompts only, printed to stdout
    eikalea --count 20 --model nemotron-3.5-lightning:30b --out prompts.jsonl    # ...and saved as JSONL
    eikalea --count 3 --model nemotron-3.5-lightning:30b --generate-image Krea2  # each prompt rendered right
    eikalea --seed 42 --model nemotron-3.5-lightning:30b \\                       # after it's generated, one
        --generate-image "Krea2+Upscale.app"                                    # seed at a time
    eikalea --count 20 --model nemotron-3.5-lightning:30b gemma4:26b             # model picked at random per seed
    eikalea --count -1 --model nemotron-3.5-lightning:30b                        # run until interrupted (Ctrl+C)
    eikalea --count -1 --model nemotron-3.5-lightning:30b --generate-image Krea2 # ...same, but rendering each too
    eikalea --count 20 --model nemotron-3.5-lightning:30b --json | jq .prompt    # pipeable JSONL on stdout

    # Replay a saved prompt list (as written by --out) instead of generating
    # fresh via the LLM -- the backend is never touched, so no --model, but
    # --generate-image is required (otherwise there's nothing to do):
    eikalea replay --in prompts.jsonl --generate-image Krea2

    # The six priming axes (medium, composition, subject, palette, mood,
    # art movement) are a template + wildcard files, resolved via
    # dynamicprompts -- both fully overridable. Get an editable copy of the
    # packaged defaults to start from, and check a custom one resolves
    # cleanly before spending an LLM call on it:
    eikalea templates export ./my-templates
    eikalea templates validate --template ./my-templates/template.md \\
        --wildcards-dir ./my-templates/wildcards
    eikalea --count 20 --model nemotron-3.5-lightning:30b \\
        --template ./my-templates/template.md \\
        --wildcards-dir ./my-templates/wildcards
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import openai

from .llm_expander import (
    build_user_message,
    export_templates,
    generate,
    pick_model,
    unload_ollama_model,
    validate_template,
)

# Matches uncomfymcp's own ComfyClient/server defaults exactly (comfy.py,
# server.py) -- no default workflow name, since uncomfymcp itself never
# assumes one either; the caller must always name it explicitly.
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_TIMEOUT = 300.0

COMMANDS = {"generate", "replay", "templates"}


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


def embed_author_metadata(path: str, author: str) -> None:
    """Name the model that wrote the prompt, without disturbing the
    "prompt"/"workflow" chunks uncomfymcp already embedded. Written two
    ways: a plain PNG "Author" text chunk (what exiftool/identify/Pillow
    read directly) and the EXIF Artist tag -- confirmed empirically that
    GNOME Files' and GIMP's own metadata views don't surface the plain PNG
    chunk despite it being spec-correct, but both read EXIF, which PNG's
    eXIf chunk carries just as validly as a JPEG would."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    img = Image.open(path)
    img.load()  # tEXt chunks can follow IDAT, so force a full read first
    info = PngInfo()
    for key, value in img.text.items():
        info.add_text(key, value)
    info.add_text("Author", author)

    exif = img.getexif()
    exif[0x013B] = author  # Artist
    img.save(path, pnginfo=info, exif=exif)


def load_prompts_jsonl(path: str) -> list[tuple[int, str, str | None]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append((obj["seed"], obj["prompt"], obj.get("model")))
    return records


def save_prompts(records: list[tuple[int, str, str | None]], out_path: str) -> None:
    with open(out_path, "a", encoding="utf-8") as f:
        for seed, prompt, model in records:
            record = {"seed": seed, "prompt": prompt}
            if model is not None:
                record["model"] = model
            f.write(json.dumps(record) + "\n")


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


def generate_with_connection_hint(*args, **kwargs) -> str:
    """Wraps generate() to turn a bare connection/timeout error into an
    actionable message -- these usually trace back to a GPU-heavy process
    (e.g. ComfyUI holding a model resident) starving the LLM backend of
    VRAM and forcing a slow CPU fallback, rather than a real bug."""
    try:
        return generate(*args, **kwargs)
    except openai.APIConnectionError as e:
        host = kwargs.get("host", "the backend")
        print(
            f"\nCould not reach the LLM backend at {host}: {e}\n"
            "If this is a timeout, check whether another GPU-heavy process (e.g. ComfyUI still "
            "holding a model resident) is starving it of VRAM -- `nvidia-smi` shows what's using it.",
            file=sys.stderr,
        )
        sys.exit(1)


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

            prompt = generate_with_connection_hint(
                seed, models=args.model, host=args.api_host,
                template_path=args.template, wildcards_dir=args.wildcards_dir,
                reasoning_effort=args.reasoning_effort,
            )
            print_prompt_body(seed, prompt, as_json=args.json)

            if args.out:
                save_prompts([(seed, prompt, model)], args.out)

            if args.generate_image:
                if not args.no_unload:
                    unload_ollama_model(model, host=args.api_host)
                out_path = unique_output_path(args.outdir, seed)
                asyncio.run(generate_image(
                    prompt, seed, args.generate_image, args.comfy_url, args.timeout, out_path
                ))
                embed_author_metadata(out_path, f"eikalea ({model})")
                print_status("", as_json=args.json)
                print_status(f"saved: {out_path}", as_json=args.json)

            seed += 1
            i += 1
    except KeyboardInterrupt:
        print_status("Stopped.", as_json=args.json)


def cmd_generate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.generate_image:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)

    missing_wildcards = validate_template(args.template, args.wildcards_dir)
    if missing_wildcards:
        parser.error(
            "template references undefined wildcard(s): " + ", ".join(missing_wildcards)
            + " -- check --template and --wildcards-dir"
        )

    if args.count < 0 or args.generate_image:
        run_streaming(args, limit=None if args.count < 0 else args.count)
        return

    start_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    # Generate every prompt first, while the backend is still warm --
    # unloading it between each generation (as a naive interleaved loop
    # would) forces a full reload of a 24GB+ model from disk on every
    # following iteration. Still printed one at a time as each finishes
    # (not batched up silently) so a large --count shows live progress
    # instead of going quiet for however long the whole batch takes.
    records = []
    for i, seed in enumerate(start_seed + offset for offset in range(args.count)):
        model = pick_model(seed, args.model)
        print_prompt_header(
            seed, as_json=args.json, progress=f"{i + 1}/{args.count}",
            model=model if len(args.model) > 1 else None,
        )

        final_prompt = generate_with_connection_hint(
            seed, models=args.model, host=args.api_host,
            template_path=args.template, wildcards_dir=args.wildcards_dir,
            reasoning_effort=args.reasoning_effort,
        )
        records.append((seed, final_prompt, model))
        print_prompt_body(seed, final_prompt, as_json=args.json)

    if args.out:
        save_prompts(records, args.out)


def cmd_replay(args: argparse.Namespace) -> None:
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    records = load_prompts_jsonl(args.in_path)
    for i, (seed, final_prompt, _model) in enumerate(records):
        progress = f"{i + 1}/{len(records)}"
        print_prompt_header(seed, as_json=args.json, progress=progress)
        print_prompt_body(seed, final_prompt, as_json=args.json)

    if args.out:
        save_prompts(records, args.out)

    # The backend is never touched in replay mode, so there's nothing to unload.
    for seed, final_prompt, model in records:
        out_path = unique_output_path(args.outdir, seed)
        asyncio.run(generate_image(
            final_prompt, seed, args.generate_image, args.comfy_url, args.timeout, out_path
        ))
        # Older or hand-written JSONL files may not carry a model -- fall
        # back to naming just the tool rather than skipping the metadata.
        author = f"eikalea ({model})" if model is not None else "eikalea"
        embed_author_metadata(out_path, author)
        print_status("", as_json=args.json)
        print_status(f"saved: {out_path}", as_json=args.json)


def cmd_templates_export(args: argparse.Namespace) -> None:
    dest = export_templates(args.dir)
    print(f"Wrote default template and wildcards to {dest}")


def cmd_templates_validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    missing = validate_template(args.template, args.wildcards_dir)
    if missing:
        parser.error(
            "template references undefined wildcard(s): " + ", ".join(missing)
            + " -- check --template and --wildcards-dir"
        )

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    print(f"Template is valid (seed {seed}):")
    print()
    print(build_user_message(seed, args.template, args.wildcards_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Experimental art generator inspired by the infinite monkey theorem -- "
                    "LLM-synthesized prompts from seeded pools, optionally rendered via ComfyUI.",
        epilog="Requires an OpenAI-compatible chat completions server (e.g. Ollama, via `ollama "
               "serve`) reachable at --api-host. --generate-image additionally requires a running "
               "ComfyUI instance (at --comfy-url) with the named workflow already saved.",
    )
    subparsers = parser.add_subparsers(dest="command")

    gen = subparsers.add_parser("generate", help="Generate prompts, and optionally images (default command).")
    gen.add_argument(
        "--count", type=int, default=1,
        help="How many prompts to generate. A negative value (e.g. -1) runs until interrupted "
             "(Ctrl+C) instead.",
    )
    gen.add_argument("--seed", type=int, default=None, help="Fixed starting seed (omit for random each run).")
    gen.add_argument(
        "--model", type=str, nargs="+", required=True,
        help="One or more model names. With more than one, a model is picked at random per seed "
             "(same seed -> same model, like every other axis).",
    )
    gen.add_argument(
        "--api-host", type=str, default="http://localhost:11434",
        help="Base URL of any OpenAI-compatible chat completions server (Ollama by default).",
    )
    gen.add_argument(
        "--template", type=str, default=None, metavar="FILE",
        help="Override the packaged prompt-assembly template (a dynamicprompts template -- see "
             "`templates export` to get an editable starting copy).",
    )
    gen.add_argument(
        "--wildcards-dir", type=str, default=None, metavar="DIR",
        help="Override the packaged wildcards directory the template's __axis__ tokens resolve "
             "against (see `templates export`).",
    )
    gen.add_argument(
        "--reasoning-effort", type=str, default="none", choices=["none", "low", "medium", "high"],
        help="Reasoning effort for models that support hidden chain-of-thought (default: none -- "
             "disables it, since it only adds latency for this task without improving output).",
    )
    gen.add_argument(
        "--out", type=str, default=None, metavar="FILE",
        help="Append generated prompts to this file, as JSONL ({\"seed\": ..., \"prompt\": ...} "
             "per line).",
    )
    gen.add_argument(
        "--json", action="store_true",
        help="Print each prompt to stdout as a JSON line ({\"seed\": ..., \"prompt\": ...}) "
             "instead of the human-readable block. Status/progress messages move to stderr so "
             "stdout stays pure, pipeable JSONL.",
    )
    gen.add_argument(
        "--generate-image", type=str, default=None, metavar="WORKFLOW",
        help="Also render each prompt via ComfyUI, using this workflow name (as ComfyUI's "
             "workflow list shows it).",
    )
    gen.add_argument(
        "--no-unload", action="store_true",
        help="Don't evict the model from VRAM before each image render. Skip this only if your "
             "GPU has enough VRAM to hold both the LLM and ComfyUI's models at once -- it avoids "
             "the reload cost between prompts.",
    )
    gen.add_argument("--comfy-url", type=str, default=DEFAULT_COMFY_URL)
    gen.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                      help="Seconds to wait for a generation before giving up.")
    gen.add_argument("--outdir", type=str, default="./eikalea_outputs")

    rep = subparsers.add_parser(
        "replay", help="Replay prompts from a JSONL file (as written by --out) and render them."
    )
    rep.add_argument(
        "--in", dest="in_path", type=str, required=True, metavar="FILE",
        help="Read {\"seed\": ..., \"prompt\": ...} records from this JSONL file. The LLM "
             "backend is never touched in this mode.",
    )
    rep.add_argument(
        "--generate-image", type=str, required=True, metavar="WORKFLOW",
        help="Render each prompt via ComfyUI, using this workflow name (as ComfyUI's workflow "
             "list shows it). Required -- otherwise there's nothing to do with the loaded prompts.",
    )
    rep.add_argument(
        "--out", type=str, default=None, metavar="FILE",
        help="Also append the replayed prompts to this file, as JSONL.",
    )
    rep.add_argument("--json", action="store_true", help="Print each prompt to stdout as a JSON line.")
    rep.add_argument("--comfy-url", type=str, default=DEFAULT_COMFY_URL)
    rep.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                      help="Seconds to wait for a generation before giving up.")
    rep.add_argument("--outdir", type=str, default="./eikalea_outputs")

    tmpl = subparsers.add_parser("templates", help="Work with the template/wildcards that build prompts.")
    tmpl_subparsers = tmpl.add_subparsers(dest="templates_command")
    tmpl_export = tmpl_subparsers.add_parser(
        "export", help="Write the packaged default template and wildcards to a directory for editing."
    )
    tmpl_export.add_argument("dir", type=str, metavar="DIR")
    tmpl_validate = tmpl_subparsers.add_parser(
        "validate",
        help="Check a template for undefined wildcard references, and print what it resolves to.",
    )
    tmpl_validate.add_argument("--template", type=str, default=None, metavar="FILE")
    tmpl_validate.add_argument("--wildcards-dir", type=str, default=None, metavar="DIR")
    tmpl_validate.add_argument(
        "--seed", type=int, default=None, help="Seed to resolve with (omit for random)."
    )

    if len(sys.argv) == 1:
        parser.print_help()
        return

    if sys.argv[1] not in COMMANDS and sys.argv[1] not in ("-h", "--help"):
        sys.argv.insert(1, "generate")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args, gen)
    elif args.command == "replay":
        cmd_replay(args)
    elif args.command == "templates":
        if args.templates_command == "export":
            cmd_templates_export(args)
        elif args.templates_command == "validate":
            cmd_templates_validate(args, tmpl_validate)
        else:
            tmpl.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
