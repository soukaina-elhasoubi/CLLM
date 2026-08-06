*This project has been created as part of the 42 curriculum by **sel-haso**.*

# Call me Maybe

## Description

**Call me Maybe** is a Python project that turns natural-language prompts into structured function calls using a local Large Language Model (LLM) and constrained decoding.

Instead of returning free-form text, the program reads a list of available function definitions, identifies the most relevant function, generates the matching arguments, and writes a valid JSON object to disk. The implementation is designed to be robust, deterministic, and schema-aware while keeping the logic readable and easy to extend.

The current project uses the provided **llm_sdk** wrapper around **Qwen/Qwen3-0.6B**, a custom tokenizer built from a vocabulary trie, and Pydantic models to validate function metadata and internal state.

Whenever no function clearly matches the request, the program falls back to **`fn_unknown`**, ensuring that every prompt still yields a valid tool call.

---

# Features

- Local LLM inference through the provided SDK.
- Custom tokenizer and trie-based encoder for efficient token lookup.
- Constrained decoding for function selection and argument generation.
- Pydantic-based validation for function definitions and internal objects.
- Support for:
  - strings
  - integers
  - floating-point numbers
  - booleans
  - regular expressions
  - file paths
- JSON output generation with schema validation.
- Graceful error handling for malformed input, missing files, invalid JSON, and unexpected failures.
- Logits caching for faster repeated inference.
- Optional CLI flags for choosing a model and device.
- A small unittest suite under the **tests/** directory for encoder and function validation contracts.

---

# Requirements

- Python 3.10+
- uv
- llm_sdk (provided)
- numpy
- pydantic

---

# Installation

```bash
make install
```

or

```bash
uv sync
```

---

# Usage

Run with the default files:

```bash
make run
```

or

```bash
uv run python -m src
```

Run with custom files:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

You can also override the model and device used by the SDK:

```bash
uv run python -m src \
    --model_name Qwen/Qwen3-0.6B \
    --device cpu
```

---

# Debugging

```bash
make debug
```

---

# Static analysis

```bash
make lint
```

or, with stricter checks:

```bash
make lint-strict
```

---

# Algorithm explanation

The generation process follows four main steps.

## 1. Load resources

The application loads:

- the tokenizer vocabulary;
- the LLM wrapper;
- the function definitions;
- the instruction templates used to guide tool selection.

Each function schema is encoded once and reused throughout the run.

## 2. Select the function

The system prompt exposes the available tools to the model. The generation loop does not allow arbitrary free text at this stage; instead, it restricts the next-token choices to valid function names and valid JSON fragments. This is the constrained decoding step that makes the output reliable.

## 3. Generate arguments

Once the function is selected, each parameter is filled according to its expected type.

### Strings

Candidate strings are extracted from the prompt and reused as valid argument values.

### Numbers

Numbers are extracted with regular expressions and normalized so they can be stored as JSON numbers. The implementation supports forms such as:

```text
12
-15
+8
3.14
-8.5
.25
-.5
5.
```

and converts them to normalized JSON-friendly values.

### Booleans

Only the valid JSON literals

```text
true
false
```

are accepted.

### Regex, paths and other special values

The project also extracts regex patterns, file-system paths, and other prompt-derived values when the function schema requires them.

## 4. Produce the result

The final structure is assembled as a JSON object with the keys **prompt**, **name**, and **parameters**, then written to the output file.

---

# Design choices

Several implementation decisions were made to keep the project deterministic and aligned with the assignment requirements.

- A custom tokenizer implemented from the provided vocabulary.
- A trie structure for efficient token lookup.
- Pydantic-based validation for function definitions and internal state.
- Constrained decoding rather than free-form generation.
- Cached logits to avoid repeating expensive model evaluations.
- Explicit fallback to **`fn_unknown`** when the prompt is ambiguous or unsupported.
- Clear input validation and user-facing error messages.

---

# Input validation

Before every run, the application validates its inputs.

The program checks:

- input files exist;
- JSON syntax is valid;
- the vocabulary can be loaded;
- `functions_definition.json` contains a list of function objects;
- every function has a valid name, parameter schema and return type;
- duplicate function names are rejected;
- `function_calling_tests.json` contains a non-empty list of objects with a string `prompt` field.

Invalid inputs terminate the program with a clear error message and exit status `1`.

---

# Error handling

The application explicitly handles:

- missing files;
- malformed JSON;
- invalid function definitions;
- duplicate functions;
- unsupported parameter types;
- invalid generated JSON;
- invalid tool calls;
- vocabulary loading failures.

Unexpected exceptions are also caught and reported cleanly so the program does not fail silently.

---

# Performance

The implementation minimizes unnecessary computation while preserving reliability.

Performance improvements include:

- trie-based token lookup;
- cached logits;
- single encoding of the function schemas;
- constrained decoding to reduce the search space;
- prompt-based extraction of numbers and words.

The result is a reasonable balance between speed, correctness and robustness, even when using a small local model.

---

# Limitations

The quality of the generated function call still depends on the underlying language model and the clarity of the prompt.

Although constrained decoding guarantees valid JSON and schema-compliant arguments, ambiguous prompts may still lead to a plausible but different function choice than the one a human would expect.

---

# Testing strategy

A small unittest suite is included in the **tests/** directory to validate the most important contracts of the project.

Current tests cover:

- encoder round-trip behavior;
- word-sequence extraction;
- function validation for blank names, descriptions and parameter types.

These tests help catch regressions in the tokenizer and the function wrapper without requiring the full model pipeline to run.

---

# Example

Input

```text
What is the sum of 40 and 2?
```

Output

```json
{
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {
        "a": 40.0,
        "b": 2.0
    }
}
```

Unknown request

```text
Book me a flight to Tokyo tomorrow.
```

Output

```json
{
    "prompt": "Book me a flight to Tokyo tomorrow.",
    "name": "fn_unknown",
    "parameters": {}
}
```

---

# Bonus features

The current implementation includes a few extras beyond the base assignment requirements:

- CLI support for selecting the model and device;
- logits caching to speed up repeated generation steps;
- richer extraction for paths, regexes and prompt-derived tokens;
- a dedicated unittest suite for the core encoder and function validation logic;
- robust error reporting for the main execution path.

---

# Project structure

```text
.
├── data
│   ├── input
│   └── output
├── llm_sdk
├── moulinette
├── src
│   ├── __main__.py
│   ├── callmemaybe.py
│   ├── encoder.py
│   ├── function.py
│   ├── llm.py
│   └── __init__.py
├── tests
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Resources

- https://docs.python.org/3/
- https://docs.pydantic.dev/
- https://www.json.org/
- https://huggingface.co/Qwen/Qwen3-0.6B
- https://platform.openai.com/docs/guides/function-calling
- https://en.wikipedia.org/wiki/Trie

---

# AI use disclosure

Artificial intelligence was used during the development of this project as a learning tool.

AI was mainly used to:

- understand constrained decoding and the project's concepts;

All generated suggestions were manually reviewed, adapted, tested and integrated by **sel-haso** before being kept in the project.