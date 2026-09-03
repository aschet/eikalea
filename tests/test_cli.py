# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

import json
from pathlib import Path

import openai
import pytest

import eikalea.cli as cli


def test_main_requires_model_for_fresh_generation(monkeypatch, capsys):
    monkeypatch.setattr(cli, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr("sys.argv", ["eikalea", "--count", "5"])

    with pytest.raises(SystemExit):
        cli.main()

    assert "--model" in capsys.readouterr().err


def test_bare_flags_default_to_the_generate_command(monkeypatch):
    """`eikalea --count 1 --model X` must behave exactly like
    `eikalea generate --count 1 --model X` -- bare invocations predate the
    subcommand split and must keep working unchanged."""
    monkeypatch.setattr(cli, "generate", lambda seed, models, host, **kwargs: f"prompt for {seed}")
    monkeypatch.setattr("sys.argv", ["eikalea", "--count", "1", "--seed", "5", "--model", "test-model"])

    cli.main()  # must not raise / exit


def test_main_does_not_require_model_when_replaying_from_jsonl(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text('{"seed": 1, "prompt": "a"}\n')
    outdir = tmp_path / "out"

    async def fake_generate_image(prompt, seed, workflow_name, comfy_url, timeout, out_path):
        Path(out_path).write_bytes(b"fake png")
        return out_path

    monkeypatch.setattr(cli, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr(cli, "generate_image", fake_generate_image)
    monkeypatch.setattr(cli, "embed_author_metadata", lambda *a, **k: None)
    monkeypatch.setattr(
        "sys.argv",
        ["eikalea", "replay", "--in", str(prompts_path), "--generate-image", "MyWorkflow", "--outdir", str(outdir)],
    )

    cli.main()  # must not raise / exit


def test_main_requires_generate_image_when_replaying_from_jsonl(tmp_path, monkeypatch, capsys):
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text('{"seed": 1, "prompt": "a"}\n')

    monkeypatch.setattr(cli, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr("sys.argv", ["eikalea", "replay", "--in", str(prompts_path)])

    with pytest.raises(SystemExit):
        cli.main()

    assert "--generate-image" in capsys.readouterr().err


def test_main_with_no_args_prints_help_instead_of_generating(monkeypatch, capsys):
    monkeypatch.setattr(cli, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr("sys.argv", ["eikalea"])

    cli.main()

    out = capsys.readouterr().out
    assert "usage:" in out
    assert "generate" in out
    assert "replay" in out
    assert "templates" in out


def test_unique_output_path_uses_bare_name_when_free(tmp_path):
    assert cli.unique_output_path(str(tmp_path), 42) == str(tmp_path / "seed_42.png")


def test_unique_output_path_auto_suffixes_on_collision(tmp_path):
    (tmp_path / "seed_42.png").write_bytes(b"existing")

    assert cli.unique_output_path(str(tmp_path), 42) == str(tmp_path / "seed_42_2.png")

    (tmp_path / "seed_42_2.png").write_bytes(b"existing too")

    assert cli.unique_output_path(str(tmp_path), 42) == str(tmp_path / "seed_42_3.png")

    # A different seed is unaffected by seed 42's collisions.
    assert cli.unique_output_path(str(tmp_path), 43) == str(tmp_path / "seed_43.png")


def test_embed_author_metadata_adds_author_without_losing_existing_text_chunks(tmp_path):
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    path = tmp_path / "test.png"
    info = PngInfo()
    info.add_text("prompt", '{"fake": "workflow json"}')
    Image.new("RGB", (2, 2)).save(path, pnginfo=info)

    cli.embed_author_metadata(str(path), "eikalea (test-model)")

    reopened = Image.open(path)
    reopened.load()
    assert reopened.text["Author"] == "eikalea (test-model)"
    assert reopened.text["prompt"] == '{"fake": "workflow json"}'
    assert reopened.getexif()[0x013B] == "eikalea (test-model)"  # EXIF Artist tag


def test_main_does_not_overwrite_an_existing_image_for_the_same_seed(tmp_path, monkeypatch):
    outdir = tmp_path / "out"
    outdir.mkdir()
    (outdir / "seed_5.png").write_bytes(b"first run's image")

    monkeypatch.setattr(cli, "generate", lambda seed, models, host, **kwargs: f"prompt for {seed}")

    async def fake_generate_image(prompt, seed, workflow_name, comfy_url, timeout, out_path):
        Path(out_path).write_bytes(b"second run's image")
        return out_path

    monkeypatch.setattr(cli, "generate_image", fake_generate_image)
    monkeypatch.setattr(cli, "embed_author_metadata", lambda *a, **k: None)
    monkeypatch.setattr(cli, "unload_ollama_model", lambda *a, **k: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "eikalea", "--count", "1", "--seed", "5", "--model", "test-model",
            "--generate-image", "MyWorkflow", "--outdir", str(outdir),
        ],
    )

    cli.main()

    assert (outdir / "seed_5.png").read_bytes() == b"first run's image"
    assert (outdir / "seed_5_2.png").read_bytes() == b"second run's image"


def test_load_prompts_jsonl_roundtrips_save_prompts(tmp_path):
    path = tmp_path / "prompts.jsonl"
    records = [(1, "prompt one", "test-model"), (2, "prompt two", None)]

    cli.save_prompts(records, str(path))

    assert cli.load_prompts_jsonl(str(path)) == records


def test_load_prompts_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"seed": 1, "prompt": "a"}\n\n{"seed": 2, "prompt": "b"}\n')

    assert cli.load_prompts_jsonl(str(path)) == [(1, "a", None), (2, "b", None)]


def test_save_prompts_appends_rather_than_overwrites(tmp_path):
    path = tmp_path / "prompts.jsonl"

    cli.save_prompts([(1, "a", None)], str(path))
    cli.save_prompts([(2, "b", None)], str(path))

    assert cli.load_prompts_jsonl(str(path)) == [(1, "a", None), (2, "b", None)]


def test_main_replay_mode_renders_images_without_touching_ollama(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text('{"seed": 1, "prompt": "a scene"}\n')
    outdir = tmp_path / "out"

    calls = []

    async def fake_generate_image(prompt, seed, workflow_name, comfy_url, timeout, out_path):
        calls.append((prompt, seed, workflow_name, comfy_url, timeout, out_path))
        Path(out_path).write_bytes(b"fake png")
        return out_path

    unload_calls = []

    monkeypatch.setattr(cli, "generate_image", fake_generate_image)
    monkeypatch.setattr(cli, "embed_author_metadata", lambda *a, **k: None)
    monkeypatch.setattr(cli, "unload_ollama_model", lambda *a, **k: unload_calls.append((a, k)))
    monkeypatch.setattr(
        "sys.argv",
        [
            "eikalea", "replay",
            "--in", str(prompts_path),
            "--generate-image", "MyWorkflow",
            "--outdir", str(outdir),
        ],
    )

    cli.main()

    assert len(calls) == 1
    prompt, seed, workflow_name, comfy_url, timeout, out_path = calls[0]
    assert prompt == "a scene"
    assert seed == 1
    assert workflow_name == "MyWorkflow"
    assert out_path == str(outdir / "seed_1.png")
    assert (outdir / "seed_1.png").exists()
    # Replay mode never calls the LLM, so there's nothing to unload.
    assert unload_calls == []


def test_main_replay_embeds_the_recorded_model_or_falls_back_to_eikalea(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        '{"seed": 1, "prompt": "a scene", "model": "test-model"}\n'
        '{"seed": 2, "prompt": "another scene"}\n'
    )
    outdir = tmp_path / "out"

    async def fake_generate_image(prompt, seed, workflow_name, comfy_url, timeout, out_path):
        Path(out_path).write_bytes(b"fake png")
        return out_path

    authors = {}

    monkeypatch.setattr(cli, "generate_image", fake_generate_image)
    monkeypatch.setattr(cli, "embed_author_metadata", lambda path, author: authors.__setitem__(path, author))
    monkeypatch.setattr(
        "sys.argv",
        ["eikalea", "replay", "--in", str(prompts_path), "--generate-image", "MyWorkflow", "--outdir", str(outdir)],
    )

    cli.main()

    assert authors[str(outdir / "seed_1.png")] == "eikalea (test-model)"
    assert authors[str(outdir / "seed_2.png")] == "eikalea"


def test_main_fresh_generation_unloads_ollama_before_rendering(tmp_path, monkeypatch):
    outdir = tmp_path / "out"

    monkeypatch.setattr(cli, "generate", lambda seed, models, host, **kwargs: f"prompt for {seed}")

    order = []

    async def fake_generate_image(prompt, seed, workflow_name, comfy_url, timeout, out_path):
        order.append("generate_image")
        Path(out_path).write_bytes(b"fake png")
        return out_path

    monkeypatch.setattr(cli, "generate_image", fake_generate_image)
    monkeypatch.setattr(cli, "embed_author_metadata", lambda *a, **k: None)
    monkeypatch.setattr(cli, "unload_ollama_model", lambda *a, **k: order.append("unload"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "eikalea",
            "--count", "1",
            "--seed", "5",
            "--model", "test-model",
            "--generate-image", "MyWorkflow",
            "--outdir", str(outdir),
        ],
    )

    cli.main()

    assert order == ["unload", "generate_image"]
    assert (outdir / "seed_5.png").exists()


def test_main_unloads_only_the_model_actually_picked(tmp_path, monkeypatch):
    outdir = tmp_path / "out"

    monkeypatch.setattr(cli, "generate", lambda seed, models, host, **kwargs: f"prompt for {seed}")

    async def fake_generate_image(prompt, seed, workflow_name, comfy_url, timeout, out_path):
        Path(out_path).write_bytes(b"fake png")
        return out_path

    unloaded = []

    monkeypatch.setattr(cli, "generate_image", fake_generate_image)
    monkeypatch.setattr(cli, "embed_author_metadata", lambda *a, **k: None)
    monkeypatch.setattr(cli, "unload_ollama_model", lambda model, host: unloaded.append(model))
    monkeypatch.setattr(
        "sys.argv",
        [
            "eikalea",
            "--count", "1",
            "--seed", "5",
            "--model", "model-a", "model-b",
            "--generate-image", "MyWorkflow",
            "--outdir", str(outdir),
        ],
    )

    cli.main()

    expected_model = cli.pick_model(5, ["model-a", "model-b"])
    assert unloaded == [expected_model]


def test_main_no_unload_skips_evicting_the_model(tmp_path, monkeypatch):
    outdir = tmp_path / "out"

    monkeypatch.setattr(cli, "generate", lambda seed, models, host, **kwargs: f"prompt for {seed}")

    async def fake_generate_image(prompt, seed, workflow_name, comfy_url, timeout, out_path):
        Path(out_path).write_bytes(b"fake png")
        return out_path

    unloaded = []

    monkeypatch.setattr(cli, "generate_image", fake_generate_image)
    monkeypatch.setattr(cli, "embed_author_metadata", lambda *a, **k: None)
    monkeypatch.setattr(cli, "unload_ollama_model", lambda model, host: unloaded.append(model))
    monkeypatch.setattr(
        "sys.argv",
        [
            "eikalea",
            "--count", "1",
            "--seed", "5",
            "--model", "test-model",
            "--generate-image", "MyWorkflow",
            "--outdir", str(outdir),
            "--no-unload",
        ],
    )

    cli.main()

    assert unloaded == []


def test_main_interleaves_generation_and_rendering_across_a_batch(tmp_path, monkeypatch):
    """Regression test: a positive --count combined with --generate-image
    used to generate every prompt in the batch before rendering even the
    first image. It must render each prompt's image right after that
    prompt is generated, not after the whole batch finishes."""
    outdir = tmp_path / "out"

    generated_seeds = []
    rendered_seeds = []

    def fake_generate(seed, models, host, **kwargs):
        generated_seeds.append(seed)
        return f"prompt for {seed}"

    async def fake_generate_image(prompt, seed, workflow_name, comfy_url, timeout, out_path):
        # At the moment each image is rendered, no *later* seed should have
        # been generated yet -- that would mean the batch ran ahead of
        # rendering again.
        assert generated_seeds == rendered_seeds + [seed]
        rendered_seeds.append(seed)
        Path(out_path).write_bytes(b"fake png")
        return out_path

    monkeypatch.setattr(cli, "generate", fake_generate)
    monkeypatch.setattr(cli, "generate_image", fake_generate_image)
    monkeypatch.setattr(cli, "embed_author_metadata", lambda *a, **k: None)
    monkeypatch.setattr(cli, "unload_ollama_model", lambda *a, **k: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "eikalea",
            "--count", "3",
            "--seed", "5",
            "--model", "test-model",
            "--generate-image", "MyWorkflow",
            "--outdir", str(outdir),
        ],
    )

    cli.main()

    assert generated_seeds == [5, 6, 7]
    assert rendered_seeds == [5, 6, 7]


def test_main_prints_header_before_generation_completes(monkeypatch, capsys):
    """Generation can take a while (a live LLM call) -- the "N/total with
    seed X:" header must print before that call starts, not after, so the
    user sees a new seed has begun rather than the output going silent."""
    printed_before_generate = []

    def fake_generate(seed, models, host, **kwargs):
        printed_before_generate.append(capsys.readouterr().out)
        return f"prompt for {seed}"

    monkeypatch.setattr(cli, "generate", fake_generate)
    monkeypatch.setattr(
        "sys.argv",
        ["eikalea", "--count", "1", "--seed", "5", "--model", "test-model"],
    )

    cli.main()

    assert "1/1 with seed 5:" in printed_before_generate[0]


def test_main_json_flag_prints_prompts_as_jsonl_on_stdout(monkeypatch, capsys):
    monkeypatch.setattr(cli, "generate", lambda seed, models, host, **kwargs: f"prompt for {seed}")
    monkeypatch.setattr(
        "sys.argv",
        ["eikalea", "--count", "2", "--seed", "5", "--model", "test-model", "--json"],
    )

    cli.main()

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line]
    assert [json.loads(line) for line in lines] == [
        {"seed": 5, "prompt": "prompt for 5", "model": "test-model"},
        {"seed": 6, "prompt": "prompt for 6", "model": "test-model"},
    ]


def test_main_json_flag_moves_status_messages_to_stderr(tmp_path, monkeypatch, capsys):
    outdir = tmp_path / "out"

    monkeypatch.setattr(cli, "generate", lambda seed, models, host, **kwargs: f"prompt for {seed}")

    async def fake_generate_image(prompt, seed, workflow_name, comfy_url, timeout, out_path):
        Path(out_path).write_bytes(b"fake png")
        return out_path

    monkeypatch.setattr(cli, "generate_image", fake_generate_image)
    monkeypatch.setattr(cli, "embed_author_metadata", lambda *a, **k: None)
    monkeypatch.setattr(cli, "unload_ollama_model", lambda *a, **k: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "eikalea", "--count", "1", "--seed", "5", "--model", "test-model",
            "--generate-image", "MyWorkflow", "--outdir", str(outdir), "--json",
        ],
    )

    cli.main()

    captured = capsys.readouterr()
    # stdout carries only the one JSON prompt record -- nothing else.
    assert json.loads(captured.out.strip()) == {"seed": 5, "prompt": "prompt for 5", "model": "test-model"}
    assert "saved:" in captured.err


def test_main_prints_chosen_model_only_when_multiple_are_given(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "generate", lambda seed, models, host, **kwargs: f"prompt for {seed}")
    monkeypatch.setattr(
        "sys.argv",
        ["eikalea", "--count", "1", "--seed", "5", "--model", "model-a", "model-b"],
    )

    cli.main()

    out = capsys.readouterr().out
    assert "model " in out

    monkeypatch.setattr(
        "sys.argv",
        ["eikalea", "--count", "1", "--seed", "5", "--model", "model-a"],
    )

    cli.main()

    out = capsys.readouterr().out
    assert "model " not in out


def test_main_negative_count_runs_forever_until_interrupted(monkeypatch, capsys):
    generate_calls = []

    def fake_generate(seed, models, host, **kwargs):
        generate_calls.append(seed)
        if len(generate_calls) >= 3:
            raise KeyboardInterrupt
        return f"prompt for {seed}"

    monkeypatch.setattr(cli, "generate", fake_generate)
    monkeypatch.setattr(
        "sys.argv",
        ["eikalea", "--count", "-1", "--seed", "100", "--model", "test-model"],
    )

    cli.main()

    # Stopped by the (simulated) Ctrl+C on the 3rd call, seeds increment by 1.
    assert generate_calls == [100, 101, 102]
    out = capsys.readouterr().out
    assert "Stopped." in out
    assert "1 with seed 100:" in out
    assert "2 with seed 101:" in out


def test_main_forever_mode_saves_each_prompt_as_it_is_generated(tmp_path, monkeypatch):
    out_path = tmp_path / "prompts.jsonl"
    calls = []

    def fake_generate(seed, models, host, **kwargs):
        calls.append(seed)
        if len(calls) >= 2:
            raise KeyboardInterrupt
        return f"prompt for {seed}"

    monkeypatch.setattr(cli, "generate", fake_generate)
    monkeypatch.setattr(
        "sys.argv",
        ["eikalea", "--count", "-1", "--seed", "1", "--model", "test-model", "--out", str(out_path)],
    )

    cli.main()

    assert cli.load_prompts_jsonl(str(out_path)) == [(1, "prompt for 1", "test-model")]


def test_main_templates_export_writes_files_and_exits_without_generating(tmp_path, monkeypatch):
    dest = tmp_path / "exported"

    monkeypatch.setattr(cli, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr("sys.argv", ["eikalea", "templates", "export", str(dest)])

    cli.main()  # must not raise / exit -- and must not require --model

    assert (dest / "template.md").exists()
    assert list((dest / "wildcards").glob("*.txt"))


def test_main_templates_validate_reports_missing_wildcards(tmp_path, monkeypatch, capsys):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "axis1.txt").write_text("only_option\n")
    template_path = tmp_path / "template.txt"
    template_path.write_text("__axis1__ and __missing__")

    monkeypatch.setattr(
        "sys.argv",
        ["eikalea", "templates", "validate", "--template", str(template_path), "--wildcards-dir", str(wildcards_dir)],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert "missing" in capsys.readouterr().err


def test_main_templates_validate_prints_resolved_template_on_success(tmp_path, monkeypatch, capsys):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "axis1.txt").write_text("only_option\n")
    template_path = tmp_path / "template.txt"
    template_path.write_text("Custom: __axis1__.")

    monkeypatch.setattr(
        "sys.argv",
        [
            "eikalea", "templates", "validate",
            "--template", str(template_path), "--wildcards-dir", str(wildcards_dir), "--seed", "1",
        ],
    )

    cli.main()  # must not raise / exit

    out = capsys.readouterr().out
    assert "valid" in out
    assert "Custom: only_option." in out


def test_main_passes_template_and_wildcards_dir_overrides_to_generate(monkeypatch):
    captured = {}

    def fake_generate(seed, models, host, **kwargs):
        captured.update(kwargs)
        return f"prompt for {seed}"

    monkeypatch.setattr(cli, "generate", fake_generate)
    monkeypatch.setattr(cli, "validate_template", lambda *a, **k: [])
    monkeypatch.setattr(
        "sys.argv",
        [
            "eikalea", "--count", "1", "--seed", "5", "--model", "test-model",
            "--template", "my-template.txt", "--wildcards-dir", "my-wildcards",
        ],
    )

    cli.main()

    assert captured["template_path"] == "my-template.txt"
    assert captured["wildcards_dir"] == "my-wildcards"


def test_main_defaults_reasoning_effort_to_none(monkeypatch):
    captured = {}

    def fake_generate(seed, models, host, **kwargs):
        captured.update(kwargs)
        return f"prompt for {seed}"

    monkeypatch.setattr(cli, "generate", fake_generate)
    monkeypatch.setattr("sys.argv", ["eikalea", "--count", "1", "--seed", "5", "--model", "test-model"])

    cli.main()

    assert captured["reasoning_effort"] == "none"


def test_main_rejects_a_template_referencing_an_undefined_wildcard(tmp_path, monkeypatch, capsys):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "axis1.txt").write_text("only_option\n")
    template_path = tmp_path / "template.txt"
    template_path.write_text("__axis1__ and __missing__")

    monkeypatch.setattr(cli, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr(
        "sys.argv",
        [
            "eikalea", "--count", "1", "--model", "test-model",
            "--template", str(template_path), "--wildcards-dir", str(wildcards_dir),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert "missing" in capsys.readouterr().err


def test_generate_with_connection_hint_gives_an_actionable_message_instead_of_a_traceback(monkeypatch, capsys):
    def fake_generate(*args, **kwargs):
        raise openai.APIConnectionError(message="Request timed out.", request=object())

    monkeypatch.setattr(cli, "generate", fake_generate)

    with pytest.raises(SystemExit):
        cli.generate_with_connection_hint(5, models=["test-model"], host="http://x")

    err = capsys.readouterr().err
    assert "http://x" in err
    assert "VRAM" in err


def test_main_passes_reasoning_effort_override_to_generate(monkeypatch):
    captured = {}

    def fake_generate(seed, models, host, **kwargs):
        captured.update(kwargs)
        return f"prompt for {seed}"

    monkeypatch.setattr(cli, "generate", fake_generate)
    monkeypatch.setattr(
        "sys.argv",
        ["eikalea", "--count", "1", "--seed", "5", "--model", "test-model", "--reasoning-effort", "high"],
    )

    cli.main()

    assert captured["reasoning_effort"] == "high"
