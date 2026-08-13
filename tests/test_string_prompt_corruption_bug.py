"""Reproduces a missing-opening-quote bug in get_parameters for functions
with consecutive 'string' parameters (e.g. fn_substitute_string_with_regex
in this project's own functions_definition.json).

Run with:
    pytest tests/test_string_prompt_corruption_bug.py -v

Expected result on the CURRENT (buggy) code: FAILS, proving the running
prompt loses the opening quote for every string value once a second
string parameter is decoded after it.
"""
from unittest.mock import MagicMock

import src.engine as engine
from src.parser import PromptContext


def test_consecutive_string_params_keep_valid_quoting(monkeypatch):
    monkeypatch.setattr(engine, "get_vocab", lambda llm: ["dummy"])

    captured_prompts = []

    def fake_decode_str(llm, vocab, prompt):
        captured_prompts.append(prompt)
        return "dummy_value", "dummy_value"

    monkeypatch.setattr(engine, "_decode_str", fake_decode_str)

    # Same shape as this project's real fn_substitute_string_with_regex:
    # three consecutive 'string' parameters.
    schema_mapping = {
        "fn_substitute_string_with_regex": {
            "name": "fn_substitute_string_with_regex",
            "description": (
                "Replace all occurrences matching a regex pattern "
                "in a string."
            ),
            "parameters": {
                "source_string": {"type": "string"},
                "regex": {"type": "string"},
                "replacement": {"type": "string"},
            },
        }
    }

    context = PromptContext(
        output_path="out.json",
        raw_functions=[schema_mapping["fn_substitute_string_with_regex"]],
        available_tool_names=["fn_substitute_string_with_regex"],
        user_prompts=["Replace all numbers in 'abc 123' with NUMBERS"],
        tool_parameters={
            "fn_substitute_string_with_regex": [
                "source_string", "regex", "replacement",
            ]
        },
        schema_mapping=schema_mapping,
    )

    fake_llm = MagicMock()
    partial_json = (
        '\n{"prompt":"Replace all numbers in \'abc 123\' with NUMBERS",'
        '"name":"fn_substitute_string_with_regex","parameters":{'
    )

    engine.get_parameters(
        context,
        fake_llm,
        "Replace all numbers in 'abc 123' with NUMBERS",
        "fn_substitute_string_with_regex",
        partial_json,
    )

    # The prompt shown to the decoder for the SECOND string param
    # ('regex') should contain a properly quoted value for the first
    # param ('source_string'): ..."source_string":"dummy_value","regex":
    assert captured_prompts[1].count(
        '"source_string":"dummy_value",'
    ) == 1, (
        "the running prompt is missing the opening quote before the "
        "first string value once a second string parameter follows it "
        f"-- got: {captured_prompts[1]!r}"
    )

    # Same check for the THIRD string param ('replacement'): both
    # earlier values should still be validly quoted in the prompt.
    assert captured_prompts[2].count('"dummy_value",') == 2, (
        "opening quotes are missing for prior string values by the "
        f"time a third string parameter is reached -- got: "
        f"{captured_prompts[2]!r}"
    )