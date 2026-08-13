*This project has been created as part of the 42 curriculum by abelgarh.*

# call me maybe — Function Calling with Constrained Decoding

## Description

**call me maybe** is a function calling tool that turns natural language prompts into
structured, machine-executable function calls, entirely by controlling the token-by-token
generation of a small local language model (`Qwen/Qwen3-0.6B`, ~0.6B parameters).

Given a prompt such as *"What is the sum of 40 and 2?"*, the goal is **not** to have the
model answer the question directly, but to have it output:

```json
{"name": "fn_add_numbers", "parameters": {"a": 40, "b": 2}}
```

Small models are notoriously unreliable at producing valid, schema-compliant JSON on
their own — free-form prompting might only succeed a fraction of the time. This project
solves that problem with **constrained decoding**: at every generation step, the raw
logits produced by the model are masked so that only tokens compatible with a valid
function name, JSON structure, and the expected argument schema can ever be selected.
The result is 100% parseable, schema-compliant output, without depending on the model
"getting it right" by chance.

The program reads a list of available functions (`functions_definition.json`) and a list
of natural language prompts (`function_calling_tests.json`), and produces a single JSON
file, `function_calling_results.json`, containing the resolved function name and
arguments for every prompt.

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk` package (provided alongside this project, expected next to `src/`)

### Installation

```bash
make install
# or, equivalently:
uv sync
```

This installs `numpy`, `pydantic`, and the rest of the pinned dependencies from
`pyproject.toml` .

### Running the program

```bash
make run
# or, equivalently:
uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>]
```

By default the program reads from `data/input/` and writes to `data/output/`:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Makefile targets

| Target  | Description                                                          |
|---------|-----------------------------------------------------------------------|
| `install` | Installs dependencies via `uv sync`                                 |
| `run`     | Runs the main script (`uv run python -m src`)                       |
| `debug`   | Runs the main script under `pdb`                                    |
| `lint`    | Runs `flake8 .` and `mypy .` with the mandatory flags                |
| `clean`   | Removes `__pycache__`, `.mypy_cache`, and other build artifacts      |

## Algorithm Explanation

Constrained decoding is applied in two distinct phases, each with its own masking
strategy.

### 1. Function name selection (`get_function_name`)

Rather than letting the model free-generate a name and hoping it matches, generation is
constrained to a **trie-like prefix match** over the set of valid function names:

1. At each step, the model's logits are computed from the tokens generated so far.
2. Every token in the vocabulary is checked: a token is only kept valid if
   `result_name + token` is still a *prefix* of at least one remaining valid function
   name. All other tokens are set to `-inf`.
3. The highest-scoring valid token is appended, and the candidate set of valid names is
   narrowed to those still matching the accumulated prefix.
4. This repeats until only one candidate name remains.

This guarantees the model can never "hallucinate" a function name that doesn't exist —
every token it is allowed to emit is one that keeps at least one real function name
reachable.

### 2. Parameter extraction (`get_parameters` + `_decode_int` / `_decode_float` / `_decode_str`)

Once the function name is fixed, its parameter schema (from `functions_definition.json`)
is walked argument by argument, and a **type-specific constrained decoder** is used for
each:

- **Integers** (`_decode_int`): only digit characters and `-` are allowed. Each candidate
  token is validated by attempting `int()` on the accumulated value, and by checking that
  the resulting substring still appears in the remaining slice of the original prompt
  (`remaining_text`) — this prevents the model from inventing numbers that were not
  actually present in the user's request. Generation stops on a `,` or `}` token.
- **Numbers/floats** (`_decode_float`): same principle as integers, but also allows a
  single `.`, and rejects a second decimal point or a decimal point that doesn't appear
  in the remaining text.
- **Strings** (`_decode_str`): generation is bounded by quote characters. The decoder
  re-ranks the top candidate tokens at each step and rejects a leading `"`/`'` on an
  empty value (to avoid an immediately-empty string) and any token that would break out
  of the string boundary unexpectedly.

At every step, invalid tokens are masked to `-inf` in the logits *before* `argmax` is
applied, so the model can only ever emit tokens that keep the output both **structurally
valid JSON** and **schema-compliant**. The vocabulary itself is loaded once from the
tokenizer's vocab file (`get_vocab`) and turned into an index → token-string array for
O(1) lookups during masking.

### Simplified pipeline

```
prompt -> tokenize -> input_ids -> LLM -> logits -> mask invalid tokens -> argmax -> append token -> repeat
```

## Design Decisions

- **Greedy masked decoding over sampling**: at each step we take `argmax` of the masked
  logits rather than sampling, which keeps the process deterministic and reproducible —
  useful for grading/testing where identical inputs should produce identical outputs.
- **Grounding numeric/string values in `remaining_text`**: rather than trusting the model
  to copy digits or words correctly, every candidate token for numbers and strings is
  checked against what's left of the original prompt. This anchors generation to the
  actual user input instead of letting the model invent values, and is what keeps
  accuracy high even on a 500M-parameter model.
- **Type-specific decoders instead of one generic JSON grammar**: rather than building a
  full generic JSON-with-schema grammar engine, we specialize a decoder per primitive
  type (`integer`, `number`, `string`). This keeps each decoder small, easy to reason
  about, and easy to extend with a new type.
- **`pydantic` for the shared context**: `PromptContext` centralizes everything the
  pipeline needs (raw functions, function names, prompts, schema mapping, accumulated
  results) in a single validated object, instead of passing five or six loose arguments
  between functions.
- **Public SDK surface only**: only the public methods of `Small_LLM_Model`
  (`encode`, `decode`, `get_logits_from_input_ids`, `get_path_to_vocab_file`,
  `get_path_to_merges_file`) are used — no private attributes/methods of `llm_sdk` are
  touched, per the subject's constraints.

## Performance Analysis

- **Accuracy**: because function-name selection is constrained to a prefix-trie over the
  real function names, name selection is always correct-or-nothing — the model cannot
  emit a name that doesn't exist. Argument extraction accuracy depends on the
  `remaining_text` grounding described above, which in practice keeps numeric and string
  extraction close to the values actually present in the prompt.
- **Validity**: JSON validity is 100% by construction — logits for any token that would
  break the JSON structure or the expected schema are masked to `-inf` before the token
  is ever selected, so the resulting output is always parseable.
- **Speed**: generation is token-by-token and re-computes logits for every candidate
  step (one forward pass per generated token per field), which is the dominant cost.
  For the target scale (a handful of short fields per prompt, over a small set of test
  prompts), this comfortably completes well within the 5-minute budget on standard
  hardware; the main lever for further speed, if needed, is caching repeated prefixes
  or batching independent decode calls.
- **Reliability**: every decoding loop has an explicit stopping condition (a valid
  terminator token, or all logits being `-inf`), which prevents infinite loops even on
  unexpected model behavior.

## Challenges Faced

- **Reconstructing the tokenizer's "space" convention**: many BPE tokenizers encode a
  leading space as a special character (`Ġ`) rather than a literal space. Every masking
  step needs to strip/replace this consistently, or valid tokens get rejected simply
  because of a mismatched space representation.
- **Preventing the model from inventing values**: without grounding candidate tokens
  against the original prompt's remaining text, a small model will happily generate
  plausible-looking but incorrect numbers or words. Checking `candidate not in
  remaining_text` at every step was key to keeping extraction faithful to the input.
- **Avoiding infinite loops in constrained decoding**: masking too aggressively can leave
  a step with *no* valid tokens at all. Every decode loop checks
  `np.all(np.isneginf(logits))` and breaks out of generation gracefully rather than
  looping forever or crashing.
- **Balancing strictness with usability of the JSON schema**: enforcing schema
  compliance (correct type, correct key) without also over-constraining and rejecting
  legitimate values (e.g. negative numbers, decimals near the end of a sentence) required
  careful edge-case handling in `_decode_int` / `_decode_float`.

## Testing Strategy

- **Unit tests** (pytest, not shipped/graded per subject rules) were written to cover:
  - malformed/missing `functions_definition.json` and `function_calling_tests.json`
    files (invalid JSON, missing file, empty prompt);
  - each parameter type decoder in isolation (`_decode_int`, `_decode_float`,
    `_decode_str`) against known prompts with expected numeric/string values;
  - function name resolution against ambiguous/ overlapping-prefix function names.
- **End-to-end tests**: running `uv run python -m src` against the example
  `functions_definition.json` / `function_calling_tests.json` provided in the subject,
  and validating that `function_calling_results.json`:
  - is valid JSON;
  - contains exactly the `prompt`, `name`, and `parameters` keys for every entry;
  - has argument types matching the schema exactly.
- **Edge cases** specifically exercised: empty strings, large numbers, special
  characters in prompts, ambiguous prompts, and functions with multiple parameters of
  mixed types, per the subject's testing guidance.
- **Static analysis**: `make lint` (flake8 + mypy with the mandatory flags) is run on
  every change to catch typing and style regressions early.

## Example Usage

Given `data/input/functions_definition.json`:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {"s": {"type": "string"}},
    "returns": {"type": "string"}
  }
]
```

and `data/input/function_calling_tests.json`:

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Reverse the string 'hello'"}
]
```

running:

```bash
uv run python -m src
```

produces `data/output/function_calling_results.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

## Resources

### References on the topic

- [Hugging Face — Transformers documentation](https://huggingface.co/docs/transformers)
- [Hugging Face — Tokenizers / BPE overview](https://huggingface.co/docs/tokenizers/)
- OpenAI — [Function calling guide](https://platform.openai.com/docs/guides/function-calling)
- [Guidance library (Microsoft)](https://github.com/guidance-ai/guidance) 

### How AI was used in this project

An AI assistant (chatgpt) was used during development for the following, well-scoped
tasks:

- **Debugging**: diagnosing and fixing a `mypy` type error in `src/engine.py`
  (`get_parameters`), where an unannotated loop variable inferred as `int` from one
  branch conflicted with a `float` assignment in another branch; the fix was an explicit
  `int | float | str` annotation.
- **Documentation**: drafting this `README.md` from the subject requirements and the
  existing source code (`engine.py`, `parser.py`, `main.py`, `__main__.py`), including
  the algorithm explanation, design rationale, and example usage sections.