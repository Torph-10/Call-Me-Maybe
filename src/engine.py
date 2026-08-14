import numpy as np
import json
from typing import Any
from src.parser import PromptContext
from llm_sdk import Small_LLM_Model


def generate_json_calls(context: PromptContext, llm: Small_LLM_Model) -> None:
    """Processes all user queries and forces
    the LLM to output valid JSON calls."""
    for query in context.user_prompts:
        func_name = get_function_name(context, llm, query)

        partial_json = (
            f'\n{{"prompt":"{query}","name":'
            f'"{func_name}","parameters":{{'
        )

        params = get_parameters(context, llm, query, func_name, partial_json)

        result = {
            "prompt": query,
            "name": func_name,
            "parameters": params
        }
        context.generated_results.append(result)


def get_vocab(llm: Small_LLM_Model) -> list[str]:
    """Builds a fast index-to-string array from the model's tokenizer vocab."""
    vocab_path = llm.get_path_to_vocab_file()

    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = [""] * (max(data.values()) + 1)
    for token_str, token_id in data.items():
        vocab[token_id] = token_str

    return vocab


def get_function_name(
    context: PromptContext, llm: Small_LLM_Model, query: str
) -> str:
    """Masks logits to ensure the LLM only outputs a valid function name."""
    functions_text = "\n".join(
        f"- {func['name']}: {func.get('description', '')}"
        for func in context.raw_functions
    )
    prompt = (
        "Choose the function that best matches the user request.\n"
        "Use the function descriptions to determine the correct function.\n\n"
        f"User request: {query}\n\n"
        f"Available functions:\n{functions_text}\n\n"
        "Function name:"
    )
    result_name = ""
    valid_names = context.available_tool_names.copy()
    vocab = get_vocab(llm)
    input_ids = llm.encode(prompt).tolist()[0]

    while len(valid_names) > 1:
        logits = llm.get_logits_from_input_ids(input_ids)

        for token_id in range(len(logits)):
            if token_id >= len(vocab):
                logits[token_id] = float("-inf")
                continue

            token = vocab[token_id].replace("Ġ", " ")
            if not any(
                name.startswith(result_name + token) for name in valid_names
            ):
                logits[token_id] = float("-inf")

        best_id = int(np.argmax(logits))
        input_ids.append(best_id)

        token = vocab[best_id].replace("Ġ", " ")
        result_name += token
        valid_names = (
            [name for name in valid_names if name.startswith(result_name)]
        )

    return valid_names[0]


def _decode_int(
    llm: Small_LLM_Model, vocab: list[str],
    prompt: str, remaining_text: str
) -> tuple[int, str, str]:
    """Constrains generation to integer characters only."""
    input_ids = llm.encode(prompt).tolist()[0]
    value = ""

    while True:
        logits = llm.get_logits_from_input_ids(input_ids)
        for token_id in range(len(logits)):
            if token_id >= len(vocab):
                logits[token_id] = float("-inf")
                continue

            token = vocab[token_id]
            is_valid = (
                all(c in "-0123456789" for c in token)
                or token.strip() in [',', '}']
            )
            candidate = value + token

            try:
                if candidate.strip() and candidate.strip() not in [',', '}']:
                    int(candidate)
            except (ValueError, TypeError):
                is_valid = False

            if (
                not is_valid or
                (',' in token and len(value) < 1) or
                ('}' in token and len(value) < 1) or
                token not in remaining_text or
                candidate not in remaining_text or
                (token == '0' and value == '0')
            ):
                logits[token_id] = float("-inf")

        best_id = int(np.argmax(logits))
        token = vocab[best_id].replace("Ġ", " ").replace("Ċ", "\n")

        if token.strip() in [',', '}'] or np.all(np.isneginf(logits)):
            break

        input_ids.append(best_id)
        value += token

    remaining_text = remaining_text.replace(str(value), "", 1)
    return int(value.strip()) if value else 0, str(value), remaining_text


def _decode_float(
    llm: Small_LLM_Model, vocab: list[str],
    prompt: str, remaining_text: str
) -> tuple[float, str, str]:
    """Constrains generation to floating-point number characters only."""
    input_ids = llm.encode(prompt).tolist()[0]
    value = ""

    while True:
        logits = llm.get_logits_from_input_ids(input_ids)
        for token_id in range(len(logits)):
            if token_id >= len(vocab):
                logits[token_id] = float("-inf")
                continue

            token = vocab[token_id].replace("Ġ", " ")
            is_valid = (
                all(c in "-0123456789." for c in token)
                or token.strip() in [',', '}']
            )
            candidate = value + token

            try:
                if candidate.strip() and candidate.strip() not in [',', '}']:
                    float(candidate)
            except (ValueError, TypeError):
                is_valid = False

            if (
                not is_valid or
                ('.' in token and '.' in value) or
                ('.' in token and len(value) < 1) or
                candidate not in remaining_text or
                (',' in token and len(value) < 1) or
                ('}' in token and len(value) < 1) or
                (token == '0' and value == '0') or
                ('.' in token and '.' not in remaining_text) or
                '!' in token
            ):
                logits[token_id] = float("-inf")

        best_id = int(np.argmax(logits))
        token = vocab[best_id].replace("Ġ", " ").replace("Ċ", "")

        if token.strip() in [',', '}'] or np.all(np.isneginf(logits)):
            break

        input_ids.append(best_id)
        value += token

    remaining_text = remaining_text.replace(str(value), "", 1)
    return float(value.strip()) if value else 0.0, str(value), remaining_text


def _decode_str(
    llm: Small_LLM_Model, vocab: list[str], prompt: str
) -> tuple[str, str]:
    """Constrains generation to string boundaries (quotes)."""
    prompt += '"'
    input_ids = llm.encode(prompt).tolist()[0]
    value = ""

    while True:
        logits = llm.get_logits_from_input_ids(input_ids)
        sorted_logits = (
            sorted(enumerate(logits), key=lambda item: item[1], reverse=True)
        )

        i = 0
        for token_id, score in sorted_logits:
            if token_id >= len(vocab):
                logits[token_id] = float("-inf")
                continue

            token = vocab[token_id].replace("Ġ", " ").replace("Ċ", "")

            if i < 20:
                i += 1
                if (
                    (token == '"' and len(value) == 0) or
                    (token == "'" and len(value) == 0) or
                    ('"' in token and token[0] != '"')
                ):
                    logits[token_id] = float("-inf")

        best_id = int(np.argmax(logits))
        input_ids.append(best_id)
        token = vocab[best_id].replace("Ġ", " ").replace("Ċ", "\n")

        if (
            token[0] == '"' or np.all(np.isneginf(logits))
            or (token[0] == '}' and '{' not in value)
        ):
            break

        value += token

    return value.strip(), str(value)


def get_parameters(
    context: PromptContext, llm: Small_LLM_Model,
    query: str, func_name: str, partial_json: str
) -> dict[str, Any]:
    """Routes parameter extraction based on the schema types."""
    params = context.schema_mapping[func_name]['parameters']
    vocab = get_vocab(llm)
    result: dict[str, Any] = {}
    remaining_text = query

    prompt = (
        "You are a JSON filler.\n"
        "Do not invent values.\n"
        "Do not invent decimals.\n"
        "Do not explain.\n"
        "Do not repeat previous parameters.\n"
        "Do not include surrounding text.\n"
        f"Fill this JSON to call this function: "
        f"{context.schema_mapping[func_name]}:\n"
        f"{partial_json}"
    )

    for arg_name, arg_schema in params.items():
        prompt += f'"{arg_name}":'

        if arg_schema['type'] == 'integer':
            val_int, raw_str, remaining_text = (
                _decode_int(llm, vocab, prompt, remaining_text)
            )
            result[arg_name] = val_int
            prompt += f"{raw_str},"

        elif arg_schema['type'] == 'number':
            val_float, raw_str, remaining_text = (
                _decode_float(llm, vocab, prompt, remaining_text)
            )
            result[arg_name] = val_float
            prompt += f"{raw_str},"

        elif arg_schema['type'] == 'string':
            val_str, raw_str = _decode_str(llm, vocab, prompt)
            result[arg_name] = val_str
            prompt += f'{raw_str}",'

    return result
