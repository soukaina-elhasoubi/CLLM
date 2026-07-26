*This project has been created as part of the 42 curriculum by **sel-haso**.*

# Call me Maybe

## Description

**Call me Maybe** is a Python project implementing a lightweight function-calling system powered by a local Large Language Model (LLM).

Instead of generating free-form text, the program converts natural language requests into structured JSON function calls. Given a set of function definitions, it selects the most appropriate function and generates arguments that respect the expected parameter types.

The project relies on the provided **llm_sdk** wrapper around **Qwen/Qwen3-0.6B** and uses **constrained decoding** to ensure that only valid function names and argument values are generated.

Whenever a request cannot be matched to one of the available functions, the program falls back to the dedicated **`fn_unknown`** function, guaranteeing that every prompt produces a valid output.

---

# Features

- Local LLM inference using the provided SDK.
- Trie-based tokenizer for efficient encoding.
- Constrained decoding for function selection.
- Automatic argument generation.
- Support for:
  - strings
  - integers
  - floating-point numbers
  - booleans
  - regular expressions
  - file paths
- JSON validation.
- Input validation.
- Graceful error handling with `exit(1)`.
- Logits caching for faster inference.
- Automatic fallback to `fn_unknown`.

---

# Requirements

- Python 3.10+
- uv
- llm_sdk (provided)

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

---

Run with custom files:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
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

---

# Project workflow

The program follows four main steps.

## 1. Load resources

The application:

- loads the LLM;
- loads the tokenizer vocabulary;
- builds the tokenizer trie;
- loads every function definition.

Each function definition is encoded only once during initialization.

---

## 2. Select the function

A system prompt containing every available function is constructed.

The language model is **not** allowed to generate arbitrary text.

Instead, it must choose only among the available function names using constrained decoding.

---

## 3. Generate arguments

After selecting the function, every parameter is generated independently according to its expected type.

### Strings

Candidate strings are extracted directly from the user's prompt.

### Numbers

Numbers are extracted using regular expressions and normalized before insertion into the JSON output.

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

which become:

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

Only the valid JSON values

```
true
false
```

can be generated.

### Regular expressions

Regex parameters are inferred from keywords contained in the prompt.

Examples:

- digits
- letters
- uppercase
- lowercase
- vowels
- whitespace
- punctuation


---

## 4. Produce the result

The generated function call is assembled into a valid JSON object and written to the output file.

---

# Design choices

Several implementation decisions were made to keep the project deterministic and compliant with the subject.

- Custom tokenizer implemented from the provided vocabulary.
- Trie structure for efficient token lookup.
- Function definitions encoded only once.
- Cached logits to avoid repeated model evaluations.
- Constrained decoding instead of unrestricted generation.
- Strict JSON generation.
- Automatic numeric normalization.
- Pydantic models for internal data representation.
- Dedicated fallback function (`fn_unknown`).

---

# Input validation

Before processing any request, the application validates every input file.

The program checks:

- input files exist;
- JSON syntax is valid;
- vocabulary can be loaded;
- `functions_definition.json` contains a list;
- every function has:
  - a name;
  - valid parameters;
  - valid return type;
- duplicate function names are rejected;
- `function_calling_tests.json` contains a non-empty list;
- every test contains a `"prompt"` field of type `string`.

Invalid inputs immediately terminate the program with:

```text
exit(1)
```

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

Unexpected exceptions also terminate the program safely.

---

# Performance

The implementation minimizes unnecessary computations.

Performance improvements include:

- trie-based token lookup;
- cached logits;
- single encoding of function definitions;
- constrained decoding;
- pre-extracted numbers;
- pre-extracted words.

Only valid candidates are evaluated during decoding, significantly reducing the search space.

---

# Limitations

The quality of the generated function depends on the underlying language model.

Although constrained decoding guarantees valid outputs, the selected function may occasionally differ from the user's intention for ambiguous prompts.

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

---

Unknown request

```text
Book me a flight to Tokyo tomorrow.
```

Output

```json
{
    "prompt": "What is my age",
    "name": "fn_unknown",
    "parameters": {}
}
```

---

# Project structure

```
.
├── data
│   ├── input
│   └── output
├── llm_sdk
├── src
│   ├── __main__.py
│   ├── callmemaybe.py
│   ├── encoder.py
│   ├── function.py
│   ├── llm.py
│   └── __init__.py
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

- understand constrained decoding;
- identify edge cases;
- review algorithms.

All generated suggestions were manually reviewed, adapted, tested and integrated by <i>sel-haso</i> before submission.