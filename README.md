*This project has been created as part of the 42 curriculum by sel-haso.*

# Call me Maybe

## Description

**Call me Maybe** is a Python project implementing a lightweight function calling system powered by a local Large Language Model (LLM).

Instead of generating free-form text, the program converts natural language requests into structured JSON function calls. Given a list of available function definitions, it selects the most appropriate function and generates correctly typed arguments while always producing valid JSON output.

The implementation relies on the provided **llm_sdk** wrapper around **Qwen/Qwen3-0.6B** and performs constrained token selection rather than unrestricted text generation.

If no available function matches the user's request, the program falls back to a dedicated **`fn_unknown`** function, allowing every prompt to produce a valid and predictable result.

---

# Instructions

## Requirements

- Python 3.10 or newer
- uv
- llm_sdk (provided with the project)

---

## Installation

```bash
make install
```

or

```bash
uv sync
```

---

## Run

```bash
make run
```

or

```bash
uv run python -m src
```

---

## Run with custom files

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

---

## Debug

```bash
make debug
```

---

## Static analysis

```bash
make lint
```

---

# Algorithm explanation

The project follows a constrained decoding pipeline.

Instead of asking the language model to freely generate an entire JSON object, generation is divided into two independent steps.

## 1. Function selection

All function definitions are loaded from the JSON file and encoded once during initialization.

A system prompt describing every available tool is built and sent to the model.

The language model is then allowed to choose **only** among the available function names by using:

```python
llm.next_option(...)
```

which evaluates only valid candidate token sequences.

If no suitable function can be selected, the fallback function **`fn_unknown`** is used.

---

## 2. Argument generation

Once the function has been selected, each parameter is generated independently according to its expected type.

### Strings

Candidate strings are extracted directly from the user's prompt.

### Numbers

Numbers are extracted using regular expressions and normalized before insertion into JSON.

Supported formats include:

```
12
-15
+8
3.14
-8.5
.25
-.5
5.
```

which become respectively

```
12.0
-15.0
8.0
3.14
-8.5
0.25
-0.5
5.0
```

### Booleans

Only the two valid JSON values

```
true
false
```

are proposed.

### Regular expressions

Regex parameters are generated from keywords detected inside the user's prompt (digits, letters, vowels, whitespace, punctuation, etc.).

Finally, every generated field is assembled into the final JSON object written to the output file.

---

# Design decisions

Several implementation choices were made to keep the solution simple, deterministic and compliant with the project requirements.

- A custom `Encoder` was implemented using the vocabulary supplied by `llm_sdk`.
- The encoder stores its vocabulary inside a trie for efficient token lookup.
- Function definitions are tokenized only once during initialization.
- Candidate generation and token selection are separated.
- The language model never generates arbitrary JSON tokens.
- `llm.next_option()` always selects among a restricted list of valid candidates.
- Numeric values are normalized before being inserted into JSON.
- Pydantic models are used to validate the application's internal objects.
- A fallback function (`fn_unknown`) guarantees that unknown prompts still produce valid JSON output instead of failing.

---

# Performance analysis

The implementation remains efficient because constrained decoding drastically reduces the search space.

Instead of evaluating the entire vocabulary at every decoding step, the model only evaluates a small number of valid candidates.

Initialization performs the expensive operations only once:

- loading the language model;
- building the tokenizer trie;
- encoding every function definition.

During inference, generation is fast because:

- function names are selected from a finite candidate list;
- strings are extracted directly from the prompt;
- numbers are pre-extracted and normalized;
- booleans have only two possible values.

The implementation consistently produces valid JSON while remaining lightweight enough for local execution.

---

# Challenges faced

The main challenges encountered during development included:

- understanding constrained decoding at the token level;
- implementing a tokenizer compatible with the SDK vocabulary;
- extracting structured information from natural language;
- handling many different numeric formats;
- generating only schema-compatible JSON;
- preventing invalid function selections.

A significant challenge was preserving very large numeric values while still producing valid JSON output. The solution was to avoid unnecessary numeric conversions and preserve the original textual representation whenever possible.

Another challenge was handling prompts that do not correspond to any available function. This was solved by introducing the dedicated fallback function `fn_unknown`.

---

# Testing strategy

The implementation was validated using both the provided examples and numerous additional edge cases.

Tests covered:

- integer values;
- floating-point values;
- signed numbers;
- decimal values without leading digits;
- trailing decimal points;
- extremely large numbers;
- quoted strings;
- punctuation;
- regular-expression generation;
- boolean parameters;
- malformed JSON files;
- missing input files;
- unknown user requests.

Generated outputs were systematically parsed using Python's JSON parser to verify that every produced object was syntactically valid.

---

# Example usage

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

---

Unknown request

Input

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

# Project structure

```
.
├── data/
│   ├── input/
│   └── output/
├── llm_sdk/
├── src/
│   ├── __main__.py
│   ├── callmemaybe.py
│   ├── encoder.py
│   ├── function.py
│   ├── llm.py
│   └── main.py
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Resources

## Documentation

- https://docs.python.org/3/
- https://docs.pydantic.dev/
- https://www.json.org/
- https://huggingface.co/Qwen/Qwen3-0.6B
- https://platform.openai.com/docs/guides/function-calling
- https://en.wikipedia.org/wiki/Trie

---

## AI use disclosure

Artificial intelligence was used as a learning and productivity tool during the development of this project.

AI was primarily used to:

- understand constrained decoding concepts;
- discuss implementation strategies;
- identify possible edge cases;

All proposed solutions were manually reviewed, adapted, tested and integrated by the project author. Every submitted source file was understood and verified before inclusion in the final project.