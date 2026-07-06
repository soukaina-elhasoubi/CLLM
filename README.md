*This project has been created as part of the 42 curriculum by <sel-haso>.

# Description

`call me maybe` is a Python project that demonstrates constrained decoding for function calling with a small local LLM wrapper. The program reads natural language prompts and available function definitions, then produces valid JSON objects describing which function should be invoked and with what typed arguments.

# Instructions

1. Install dependencies:
   - `make install`
2. Run the program with defaults:
   - `make run`
3. Run with custom input or output:
   - `uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json`

# Algorithm explanation

The implementation uses a token-by-token generation pipeline with a local `Small_LLM_Model` wrapper from `llm_sdk`. The model returns logits for the next character in a target JSON completion. A separate constraint layer validates the partial JSON prefix and restricts next tokens to those that preserve syntactic and schema-compatible JSON structure.

This creates a deterministic constrained decoding loop:

- Build a prompt containing the user query and available function definitions.
- Encode the prompt and generate output characters one at a time.
- At each step, compute valid next tokens based on the current JSON prefix.
- Choose the highest-scoring valid token from model logits.
- Stop only when the generated text parses successfully as the required JSON schema.

# Design decisions

- `llm_sdk.Small_LLM_Model` is implemented as a small local wrapper using a printable ASCII vocabulary to satisfy the required interface.
- Constrained decoding is enforced by `src.constraints.JsonPrefixConstraints`, which tracks JSON structure and key expectations from the function schema.
- `json.dumps(..., separators=(",", ":"))` is used to keep generated JSON compact and easier to validate.
- Pydantic models validate input file structure and final output records.

# Performance analysis

The generator processes prompts character-by-character, but the problem size remains small for reasonable prompt sets. The output is always valid JSON, and the generator performs only one forward pass per token with a local vocabulary of printable ASCII.

# Challenges faced

- Balancing a lightweight local LLM wrapper with the project requirement to use token-level logits.
- Designing a prefix validator that accepts incomplete JSON while enforcing the required output schema.
- Handling typed argument extraction from natural language prompts without relying on external LLM APIs.

# Testing strategy

- Validate JSON file loading and error handling for missing or malformed files.
- Check generated output against the function schema after decoding.
- Use sample input files in `data/input/` and run the program to verify `data/output/function_calling_results.json` creation.

# Example usage

- Default run:
  - `uv run python -m src`
- Custom paths:
  - `uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json`

# Resources

- JSON specification: https://www.json.org/json-en.html
- Pydantic documentation: https://docs.pydantic.dev/
- Python `argparse` documentation

# AI use disclosure

AI was used to generate the initial project scaffolding, the constrained decoding design, and the implementation of the local model wrapper. All generated code was reviewed and adapted manually to meet project requirements.
