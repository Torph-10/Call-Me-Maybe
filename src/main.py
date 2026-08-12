import json
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model
import src.parser as parser
import src.engine as engine


def main() -> None:
    """
    Main execution flow:
    1. Parse arguments.
    2. Load definitions and user prompts.
    3. Initialize context and LLM.
    4. Generate outputs via constrained decoding.
    5. Save results to disk.
    """
    # 1. Parse arguments
    functions_def_path, input_path, output_path = parser.get_cl_arguments()

    # 2. Load and parse data
    try:
        functions = parser.load_functions(functions_def_path)
        functions_dict = parser.get_functions_dict(functions)
        function_names = parser.get_function_names(functions)
        prompts = parser.load_user_prompts(input_path)
        function_params = parser.get_function_params(functions)
    except Exception as e:
        print(f"File reading error: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Create Central Data Structure
    context: parser.PromptContext = parser.PromptContext(
        output_path=output_path,
        raw_functions=functions,
        available_tool_names=function_names,
        user_prompts=prompts,
        tool_parameters=function_params,
        schema_mapping=functions_dict
    )

    print("Initializing the LLM...")
    llm: Small_LLM_Model = Small_LLM_Model()

    print("Starting constrained generation...")
    try:
        engine.generate_json_calls(context, llm)
    except KeyboardInterrupt:
        print("\nThe program interrupted by user!")
        sys.exit(1)
    out_file: Path = Path(context.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as file:
        json.dump(context.generated_results, file, indent=4)

    print(f"Execution complete. Check: {context.output_path}")


if __name__ == "__main__":
    main()
