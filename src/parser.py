import json
import argparse
from typing import Any
from pydantic import BaseModel


class PromptContext(BaseModel):
    """Container for all parsed data needed for the LLM generation pipeline."""
    output_path: str
    raw_functions: list[dict[str, Any]]
    available_tool_names: list[str]
    user_prompts: list[str]
    tool_parameters: dict[str, list[str]]
    schema_mapping: dict[str, dict[str, Any]]
    generated_results: list[dict[str, Any]] = []


def get_cl_arguments() -> tuple[str, str, str]:
    """Reads and returns paths from command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calls.json")

    args = parser.parse_args()
    return args.functions_definition, args.input, args.output


def load_functions(filepath: str) -> list[dict[str, Any]]:
    """Reads the JSON file containing the tools/functions definitions."""
    with open(filepath, "r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


def get_function_names(schemas: list[dict[str, Any]]) -> list[str]:
    """Returns a simple list of all function names."""
    return [tool['name'] for tool in schemas]


def get_functions_dict(
    schemas: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Indexes function definitions by their name for quick access."""
    return {tool['name']: tool for tool in schemas}


def get_function_params(schemas: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Creates a mapping between function names
    and their required parameter keys."""
    param_map: dict[str, list[str]] = {}
    for tool in schemas:
        param_map[tool['name']] = list(tool['parameters'].keys())
    return param_map


def load_user_prompts(filepath: str) -> list[str]:
    """Reads and validates the user queries from the input JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    for item in data:
        prompt = item.get('prompt', '')
        if not prompt:
            raise ValueError("Encountered an empty prompt in the input file.")
        prompts.append(prompt)

    return prompts
