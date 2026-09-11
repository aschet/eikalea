# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

import pytest

import eikalea.gpt2_expander as ge


def test_generate_gpt2_seed_text_uses_magic_prompt_generator(monkeypatch):
    captured = {}

    class FakeMagicPromptGenerator:
        def __init__(self, model_name, seed, device, max_prompt_length):
            captured["model_name"] = model_name
            captured["seed"] = seed
            captured["device"] = device

        def generate(self, template, num_images=1):
            captured["template"] = template
            captured["num_images"] = num_images
            return ["  raw draft text  "]

    monkeypatch.setattr(ge, "MagicPromptGenerator", FakeMagicPromptGenerator)

    result = ge.generate_gpt2_seed_text(42)

    assert result == "raw draft text"
    assert captured["model_name"] == ge.DEFAULT_GPT2_MODEL
    assert captured["seed"] == 42
    assert captured["device"] is None
    assert captured["template"] == ""
    assert captured["num_images"] == 1


def test_generate_gpt2_seed_text_honors_model_name_override(monkeypatch):
    captured = {}

    class FakeMagicPromptGenerator:
        def __init__(self, model_name, seed, device, max_prompt_length):
            captured["model_name"] = model_name

        def generate(self, template, num_images=1):
            return ["draft"]

    monkeypatch.setattr(ge, "MagicPromptGenerator", FakeMagicPromptGenerator)

    ge.generate_gpt2_seed_text(1, model_name="custom/model")

    assert captured["model_name"] == "custom/model"


def test_generate_gpt2_seed_text_passes_seed_text_as_continuation_seed(monkeypatch):
    captured = {}

    class FakeMagicPromptGenerator:
        def __init__(self, model_name, seed, device, max_prompt_length):
            pass

        def generate(self, template, num_images=1):
            captured["template"] = template
            return ["draft"]

    monkeypatch.setattr(ge, "MagicPromptGenerator", FakeMagicPromptGenerator)

    ge.generate_gpt2_seed_text(1, seed_text="Medium: oil painting.")

    assert captured["template"] == "Medium: oil painting."


def test_build_gpt2_user_message_substitutes_the_draft_into_the_default_template():
    msg = ge.build_gpt2_user_message("a rough draft")

    assert "a rough draft" in msg
    assert "{gpt2_seed}" not in msg


def test_build_gpt2_user_message_honors_template_override(tmp_path):
    template_path = tmp_path / "custom.md"
    template_path.write_text("Custom: {gpt2_seed}.")

    msg = ge.build_gpt2_user_message("draft text", template_path=template_path)

    assert msg == "Custom: draft text."


def test_build_axis_message_with_gpt2_subject_replaces_only_the_subject_axis(tmp_path, monkeypatch):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "medium.txt").write_text("oil painting\n")
    (wildcards_dir / "subject.txt").write_text("a portrait\n")
    template_path = tmp_path / "template.txt"
    template_path.write_text("Medium: __medium__. Subject: __subject__.")

    monkeypatch.setattr(ge, "generate_gpt2_seed_text", lambda seed, model_name, device="cpu": "a gpt2 draft")

    msg = ge.build_axis_message_with_gpt2_subject(1, template_path=template_path, wildcards_dir=wildcards_dir)

    assert msg == "Medium: oil painting. Subject: a gpt2 draft."


def test_build_axis_message_with_gpt2_subject_passes_seed_and_model_name_through(tmp_path, monkeypatch):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "subject.txt").write_text("a portrait\n")
    template_path = tmp_path / "template.txt"
    template_path.write_text("Subject: __subject__.")

    captured = {}

    def fake_generate(seed, model_name, device="cpu"):
        captured["seed"] = seed
        captured["model_name"] = model_name
        return "draft"

    monkeypatch.setattr(ge, "generate_gpt2_seed_text", fake_generate)

    ge.build_axis_message_with_gpt2_subject(
        7, gpt2_model_name="custom/model", template_path=template_path, wildcards_dir=wildcards_dir
    )

    assert captured == {"seed": 7, "model_name": "custom/model"}


def test_build_axis_message_with_gpt2_subject_falls_back_to_the_pool_on_an_empty_draft(tmp_path, monkeypatch):
    """Regression test: injecting an empty-string wildcard value used to hang
    dynamicprompts' RandomSampler forever (confirmed with a 10s timeout) --
    an empty draft must skip the override and fall back to subject.txt."""
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "subject.txt").write_text("a portrait\n")
    template_path = tmp_path / "template.txt"
    template_path.write_text("Subject: __subject__.")

    monkeypatch.setattr(ge, "generate_gpt2_seed_text", lambda seed, model_name, device=None: "")

    msg = ge.build_axis_message_with_gpt2_subject(1, template_path=template_path, wildcards_dir=wildcards_dir)

    assert msg == "Subject: a portrait."


def test_generate_gpt2_draft_retries_on_an_empty_draft(monkeypatch):
    """Regression test: a bare (seed_text="") draft had no fallback for an
    empty result -- confirmed at a ~33% rate on GPU with the default model
    -- so build_gpt2_user_message substituted nothing into "Raw draft: "
    and the synthesis LLM asked for the draft instead of writing a
    prompt."""
    captured = []

    def fake_generate_gpt2_seed_text(seed, model_name, device=None):
        captured.append(seed)
        return "" if len(captured) < 3 else f"draft-{seed}"

    monkeypatch.setattr(ge, "generate_gpt2_seed_text", fake_generate_gpt2_seed_text)

    result = ge.generate_gpt2_draft(1, "custom/model", device="cpu")

    assert result == f"draft-{captured[-1]}"
    assert captured == [1, (1 + 999_999_937) % 2**32, (1 + 2 * 999_999_937) % 2**32]


def test_generate_gpt2_draft_raises_after_max_attempts(monkeypatch):
    """Regression test: silently returning the empty draft once attempts run
    out would just reproduce the bug this function exists to avoid, at a
    lower rate instead of never."""
    monkeypatch.setattr(ge, "generate_gpt2_seed_text", lambda seed, model_name, device=None: "")

    with pytest.raises(RuntimeError, match=r"empty.*3 times.*seed 1"):
        ge.generate_gpt2_draft(1, None, device="cpu", max_attempts=3)


def test_generate_gpt2_draft_with_seed_text_skips_the_retry_entirely(monkeypatch):
    """A non-empty seed_text can never come back empty (MagicPromptGenerator
    reattaches it as the result's prefix regardless of what GPT-2 adds), so
    there's nothing to retry -- confirms generate_gpt2_draft calls
    generate_gpt2_seed_text exactly once in that case, not up to
    max_attempts times."""
    calls = []

    def fake_generate_gpt2_seed_text(seed, model_name, seed_text="", max_prompt_length=100, device=None):
        calls.append(seed)
        return seed_text + " continued"

    monkeypatch.setattr(ge, "generate_gpt2_seed_text", fake_generate_gpt2_seed_text)

    result = ge.generate_gpt2_draft(1, seed_text="Medium: oil painting.", max_prompt_length=90)

    assert result == "Medium: oil painting. continued"
    assert calls == [1]


def test_generate_gpt2_nudged_draft_resolves_the_nudge_template(tmp_path, monkeypatch):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "medium.txt").write_text("oil painting\n")
    nudge_template_path = tmp_path / "nudge.md"
    nudge_template_path.write_text("Medium: __medium__.")

    captured = {}

    def fake_generate_gpt2_seed_text(seed, model_name=None, seed_text="", max_prompt_length=100, device="cpu"):
        captured["seed"] = seed
        captured["model_name"] = model_name
        captured["seed_text"] = seed_text
        return seed_text + " continued by gpt2"

    monkeypatch.setattr(ge, "generate_gpt2_seed_text", fake_generate_gpt2_seed_text)

    result = ge.generate_gpt2_nudged_draft(1, nudge_template_path=nudge_template_path, wildcards_dir=wildcards_dir)

    assert captured["seed"] == 1
    assert captured["seed_text"] == "Medium: oil painting."
    assert result == "Medium: oil painting. continued by gpt2"


def test_generate_gpt2_nudged_draft_uses_the_entire_resolved_template(tmp_path, monkeypatch):
    """Regression test: an earlier version resolved the full six-axis
    template.md and sliced out what it assumed was the first paragraph
    (axis_text.split("\\n\\n", 1)[0]) to drop trailing instructions --
    broken for any template not shaped exactly like the packaged default
    (no blank line, or instructions first, silently leaked into the
    result). A dedicated nudge template has no trailing paragraph to slice
    around -- the entire resolved text must be used, not a guessed
    prefix."""
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "medium.txt").write_text("oil painting\n")
    nudge_template_path = tmp_path / "nudge.md"
    nudge_template_path.write_text("Medium: __medium__.\nWrite a description in one paragraph.")

    captured = {}

    def fake_generate_gpt2_seed_text(seed, model_name=None, seed_text="", max_prompt_length=100, device="cpu"):
        captured["seed_text"] = seed_text
        return "result"

    monkeypatch.setattr(ge, "generate_gpt2_seed_text", fake_generate_gpt2_seed_text)

    ge.generate_gpt2_nudged_draft(1, nudge_template_path=nudge_template_path, wildcards_dir=wildcards_dir)

    assert captured["seed_text"] == "Medium: oil painting.\nWrite a description in one paragraph."


def test_generate_gpt2_nudged_draft_sizes_max_prompt_length_off_the_resolved_text(tmp_path, monkeypatch):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "medium.txt").write_text("oil painting\n")
    nudge_template_path = tmp_path / "nudge.md"
    nudge_template_path.write_text("Medium: __medium__.")

    captured = {}

    def fake_generate_gpt2_seed_text(seed, model_name=None, seed_text="", max_prompt_length=100, device="cpu"):
        captured["max_prompt_length"] = max_prompt_length
        return "result"

    monkeypatch.setattr(ge, "generate_gpt2_seed_text", fake_generate_gpt2_seed_text)

    ge.generate_gpt2_nudged_draft(1, nudge_template_path=nudge_template_path, wildcards_dir=wildcards_dir)

    assert captured["max_prompt_length"] == len("Medium: oil painting.") // 3 + 80


def test_generate_gpt2_nudged_draft_defaults_to_the_packaged_medium_palette_mood_template(monkeypatch):
    captured = {}

    def fake_generate_gpt2_seed_text(seed, model_name=None, seed_text="", max_prompt_length=100, device=None):
        captured["seed_text"] = seed_text
        return "result"

    monkeypatch.setattr(ge, "generate_gpt2_seed_text", fake_generate_gpt2_seed_text)

    ge.generate_gpt2_nudged_draft(1)

    assert captured["seed_text"].startswith("Medium: ")
    assert "Palette:" in captured["seed_text"]
    assert "Mood:" in captured["seed_text"]
    assert "Composition:" not in captured["seed_text"]
    assert "Subject:" not in captured["seed_text"]


def test_generate_gpt2_nudged_draft_passes_model_name_through(tmp_path, monkeypatch):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "medium.txt").write_text("oil painting\n")
    nudge_template_path = tmp_path / "nudge.md"
    nudge_template_path.write_text("Medium: __medium__.")

    captured = {}

    def fake_generate_gpt2_seed_text(seed, model_name=None, seed_text="", max_prompt_length=100, device="cpu"):
        captured["model_name"] = model_name
        return "result"

    monkeypatch.setattr(ge, "generate_gpt2_seed_text", fake_generate_gpt2_seed_text)

    ge.generate_gpt2_nudged_draft(
        1, gpt2_model_name="custom/model", nudge_template_path=nudge_template_path, wildcards_dir=wildcards_dir
    )

    assert captured["model_name"] == "custom/model"
