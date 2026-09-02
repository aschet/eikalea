# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-only

import eikalea.llm_expander as le


def test_pick_functions_return_pool_members():
    seed = 12345
    assert le.pick_medium(seed) in le.MEDIA_TECHNIQUES
    assert le.pick_composition(seed) in le.COMPOSITIONS
    assert le.pick_scenery(seed) in le.SCENERY
    assert le.pick_palette(seed) in le.PALETTES
    assert le.pick_mood(seed) in le.MOODS
    assert le.pick_art_movement(seed) in le.ART_MOVEMENTS


def test_pick_functions_are_deterministic_per_seed():
    seed = 999
    assert le.pick_medium(seed) == le.pick_medium(seed)
    assert le.pick_composition(seed) == le.pick_composition(seed)
    assert le.pick_scenery(seed) == le.pick_scenery(seed)
    assert le.pick_palette(seed) == le.pick_palette(seed)
    assert le.pick_mood(seed) == le.pick_mood(seed)
    assert le.pick_art_movement(seed) == le.pick_art_movement(seed)


def test_axis_offsets_are_distinct():
    # Each axis is sampled from `seed + offset`; if two offsets collided,
    # two axes would always move in lockstep across every seed.
    offsets = {
        le._COMPOSITION_SEED_OFFSET,
        le._SCENERY_SEED_OFFSET,
        le._PALETTE_SEED_OFFSET,
        le._MOOD_SEED_OFFSET,
        le._MOVEMENT_SEED_OFFSET,
        0,  # medium uses the bare seed
    }
    assert len(offsets) == 6


def test_build_user_message_contains_all_six_axes():
    seed = 7
    msg = le.build_user_message(seed)
    assert le.pick_medium(seed) in msg
    assert le.pick_composition(seed) in msg
    assert le.pick_scenery(seed) in msg
    assert le.pick_palette(seed) in msg
    assert le.pick_mood(seed) in msg
    assert le.pick_art_movement(seed) in msg


def test_load_system_prompt_reads_nonempty_file():
    text = le.load_system_prompt()
    assert text.strip()


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


def test_unload_ollama_model_sends_keep_alive_zero(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json

    monkeypatch.setattr("requests.post", fake_post)

    le.unload_ollama_model("test-model", host="http://x")

    assert captured["url"] == "http://x/api/generate"
    assert captured["json"] == {"model": "test-model", "keep_alive": 0}
