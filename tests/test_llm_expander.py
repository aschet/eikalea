# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

from pathlib import Path

import eikalea.llm_expander as le


def test_build_user_message_is_deterministic_per_seed():
    seed = 999
    assert le.build_user_message(seed) == le.build_user_message(seed)


def test_build_user_message_resolves_all_axis_wildcards():
    msg = le.build_user_message(7)
    assert "__" not in msg  # no unresolved __axis__ token left over
    for label in ["Medium:", "Composition:", "Subject:", "Palette:", "Mood:", "Art movement/tradition:"]:
        assert label in msg


def test_build_user_message_honors_template_and_wildcards_dir_overrides(tmp_path):
    wildcards_dir = tmp_path / "wildcards"
    wildcards_dir.mkdir()
    (wildcards_dir / "axis1.txt").write_text("only_option\n")
    template_path = tmp_path / "template.txt"
    template_path.write_text("Custom: __axis1__.")

    msg = le.build_user_message(1, template_path=template_path, wildcards_dir=wildcards_dir)

    assert msg == "Custom: only_option."


def test_pick_model_is_deterministic_per_seed():
    seed = 999
    models = ["a", "b", "c"]
    assert le.pick_model(seed, models) == le.pick_model(seed, models)
    assert le.pick_model(seed, models) in models


def test_load_system_prompt_reads_nonempty_file():
    text = le.load_system_prompt()
    assert text.strip()


def test_export_templates_copies_default_template_and_wildcards(tmp_path):
    dest_dir = tmp_path / "exported"

    dest = le.export_templates(dest_dir)

    assert dest == dest_dir
    assert (dest_dir / "template.md").read_text() == le.TEMPLATE_PATH.read_text()
    exported_wildcards = sorted(p.name for p in (dest_dir / "wildcards").glob("*.txt"))
    default_wildcards = sorted(p.name for p in le.WILDCARDS_DIR.glob("*.txt"))
    assert exported_wildcards == default_wildcards


def test_generate_with_llm_sends_expected_request(monkeypatch):
    captured = {}

    class FakeMessage:
        content = "  a generated prompt  "

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, base_url, api_key, timeout):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["timeout"] = timeout
            self.chat = FakeChat()

    monkeypatch.setattr(le, "OpenAI", FakeOpenAI)

    result = le.generate_with_llm(42, "system prompt text", model="test-model", host="http://x")

    assert result == "a generated prompt"
    assert captured["base_url"] == "http://x/v1"
    kwargs = captured["kwargs"]
    assert kwargs["model"] == "test-model"
    assert kwargs["seed"] == 42
    assert kwargs["extra_body"] == {"reasoning_effort": "none"}
    assert kwargs["messages"][0] == {"role": "system", "content": "system prompt text"}
    assert kwargs["messages"][1]["content"] == le.build_user_message(42)


def test_generate_with_llm_honors_reasoning_effort_override(monkeypatch):
    captured = {}

    class FakeMessage:
        content = "a generated prompt"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, base_url, api_key, timeout):
            self.chat = FakeChat()

    monkeypatch.setattr(le, "OpenAI", FakeOpenAI)

    le.generate_with_llm(42, "system prompt text", model="test-model", host="http://x", reasoning_effort="high")

    assert captured["kwargs"]["extra_body"] == {"reasoning_effort": "high"}


def test_unload_ollama_model_sends_keep_alive_zero(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json

    monkeypatch.setattr("requests.post", fake_post)

    le.unload_ollama_model("test-model", host="http://x")

    assert captured["url"] == "http://x/api/generate"
    assert captured["json"] == {"model": "test-model", "keep_alive": 0}
