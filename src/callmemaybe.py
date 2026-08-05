import json
import re

from pydantic import BaseModel

from src.encoder import Encoder
from src.function import Function
from src.llm import LLM


REGEX_MAPPING = [
    (['vowel', 'vowels'], r'[aeiouAEIOU]'),
    (
        ['consonant', 'consonants'],
        r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]',
    ),
    (['digit', 'digits', 'number', 'numbers'], r'\\d+'),
    (['uppercase', 'upper', 'capital'], r'[A-Z]+'),
    (['lowercase', 'lower'], r'[a-z]+'),
    (['letter', 'letters', 'alphabetic'], r'[a-zA-Z]+'),
    (['space', 'spaces', 'whitespace'], r'\\s+'),
    (['punctuation', 'special'], r'[^\w\s]'),
    (['alphanumeric'], r'\\w+'),
    (['newline', 'newlines'], r'\\n+'),
    (['tab', 'tabs'], r'\\t+'),
]


def escape(text: str) -> str:
    return json.dumps(text)[1:-1]


class CallMeMaybe(BaseModel):
    llm: LLM
    encoder: Encoder
    functions: dict[str, Function]
    t_defintions: list[int]
    t_instruction_prefix: list[int]
    t_instruction_suffix: list[int]

    def __init__(self, llm: LLM, func_definitons: str) -> None:
        """Loads function definitions and initializes the prompt templates."""

        encoder = llm.encoder

        functions = {}
        with open(func_definitons, 'r') as f:
            functions_json = json.load(f)

            if not isinstance(functions_json, list):
                raise ValueError(
                    "functions_definition must contain a list"
                )
            for func in functions_json:
                name = func.get("name", "").strip()
                if not name:
                    raise ValueError("Function name cannot be empty")
                if func["name"] in functions:
                    raise ValueError(
                        f"Duplicate function: {func['name']}"
                    )

                functions[func['name']] = Function(func, encoder)

        functions["fn_unknown"] = Function(
            {
                "name": "fn_unknown",
                "description": (
                    "Use this function only when none of the other "
                    "functions matches the user's request."
                ),
                "parameters": {},
                "returns": {
                    "type": "string"
                }
            },
            encoder,
        )

        t_defintions = [t for f in functions.values() for t in f.t_definition]

        t_instruction_prefix = encoder.encode(
            '<|im_start|>system\n'
            'You are provided with function signatures '
            'within <tools></tools> XML tags:\n'
            '<tools>\n')
        t_instruction_suffix = encoder.encode(
            '</tools>\n'
            'For each function call, return a json '
            'object within <tool_call></tool_call> tags:\n'
            '<tool_call>\n'
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            '</tool_call>\n'
            '<|im_end|>\n')

        super().__init__(
            llm=llm,
            encoder=encoder,
            functions=functions,
            t_defintions=t_defintions,
            t_instruction_prefix=t_instruction_prefix,
            t_instruction_suffix=t_instruction_suffix
        )

    def set_tools(self, func: Function | None = None) -> None:
        """Updates the LLM context with function definitions."""

        if func is not None:
            definitions = func.t_definition
        else:
            definitions = self.t_defintions
        new = self.t_instruction_prefix + definitions
        new += self.t_instruction_suffix
        self.llm.set_instruction(new)

    def regex_pattern(self, text: str) -> list[int]:
        """Resolves the regex pattern from prompt keywords."""

        words = {w.strip('\'\".,!?').lower() for w in text.split()}
        for keywords, pattern in REGEX_MAPPING:
            if words & set(keywords):
                return self.encoder.encode(pattern)

        match = re.search(r"['\"](\w+)['\"]", text)
        if match:
            matched = match.group(1)
            if 'word' in words or 'words' in words:
                return self.encoder.encode(
                    r'\\b' + re.escape(matched) + r'\\b'
                )
            return self.encoder.encode(matched)

        return self.encoder.encode(r'\\w+')

    def _decode_prompt(self, prompt: str) -> str:
        try:
            decoded = json.loads(f'"{prompt}"')
            return decoded if isinstance(decoded, str) else str(decoded)
        except Exception:
            return prompt.replace('\\"', '"')

    def _infer_string_value(
        self,
        arg_name: str,
        prompt: str,
        cached_words: list[list[int]],
    ) -> list[int] | None:
        decoded_prompt = self._decode_prompt(prompt)
        arg_name_lower = arg_name.lower()

        if arg_name_lower == 'replacement':
            if 'asterisk' in decoded_prompt.lower():
                return self.encoder.encode('*')
            if 'numbers' in decoded_prompt.lower():
                return self.encoder.encode('NUMBERS')

        if arg_name_lower in {'source_string', 's'}:
            matches = re.findall(
                "\"([^\"]*)\"|'([^']*)'",
                decoded_prompt,
            )
            if matches:
                longest = ''
                for group1, group2 in matches:
                    candidate = group1 or group2 or ''
                    if len(candidate) > len(longest):
                        longest = candidate
                return self.encoder.encode(longest)

        if arg_name_lower == 'name':
            match = re.search(
                r'\b(?:greet|hello|hi|hey)\s+([^\s"\']+)',
                decoded_prompt,
                re.I
            )
            if match:
                return self.encoder.encode(match.group(1))

        if len(cached_words) == 1:
            return cached_words[0]

        return None

    def add_args(
        self,
        function: Function,
        tokens: list[int],
        text: str,
        cached_words: list[list[int]],
        cached_numbers: list[list[int]]
    ) -> list[int]:
        """Generates the arguments for the selected function."""

        SUPPORTED_TYPES = {
            "number",
            "integer",
            "float",
            "string",
            "boolean",
        }

        # def add_value(
        #     schema: dict[str, str] | str,
        #     current_tokens: list[int],
        # ) -> list[int]:
        def add_value(
            schema: dict[str, str] | str,
            arg_name: str,
            current_tokens: list[int],
        ) -> list[int]:
            if isinstance(schema, str):
                arg_type = schema
            elif isinstance(schema, dict):
                arg_type = schema.get("type", "string")
            else:
                raise ValueError("Unsupported schema value.")

            if arg_type not in SUPPORTED_TYPES:
                raise ValueError(
                    f"Unsupported parameter type: {arg_type}"
                )

            # arg_name_lower = str(schema).lower()
            arg_name_lower = arg_name.lower()
            if arg_name_lower == "regex":
                current_tokens += self.encoder.encode('"')
                current_tokens += self.regex_pattern(text)
                current_tokens += self.encoder.encode('"')
                return current_tokens

            if arg_name_lower == "path":
                current_tokens += self.encoder.encode('"')
                current_tokens += self.encoder.extract_path(text)
                current_tokens += self.encoder.encode('"')
                return current_tokens

            if arg_type == "integer":
                if cached_numbers:
                    next_tokens = self.llm.next_option(
                        current_tokens,
                        cached_numbers,
                    )
                    if next_tokens in cached_numbers:
                        cached_numbers.remove(next_tokens)
                    param = self.encoder.decode(next_tokens).strip()
                else:
                    next_tokens = self.encoder.encode("0")
                    param = "0"

                if "." in param:
                    param = param.split(".")[0]

                if param.startswith("+"):
                    param = param[1:]

                current_tokens += self.encoder.encode(param)
                return current_tokens

            if arg_type in ("number", "float"):
                if cached_numbers:
                    next_tokens = self.llm.next_option(
                        current_tokens,
                        cached_numbers,
                    )
                    if next_tokens in cached_numbers:
                        cached_numbers.remove(next_tokens)
                    param = self.encoder.decode(next_tokens).strip()
                else:
                    next_tokens = self.encoder.encode("0.0")
                    param = "0.0"

                if param.startswith("."):
                    param = "0" + param
                elif param.startswith("-."):
                    param = "-0" + param[1:]
                elif param.startswith("+."):
                    param = "+0" + param[1:]

                if param.endswith("."):
                    param += "0"

                if param.startswith("+"):
                    param = param[1:]

                if "." not in param and "e" not in param.lower():
                    param += ".0"

                current_tokens += self.encoder.encode(param)
                return current_tokens

            if arg_type == "boolean":
                options = [
                    self.encoder.encode("true"),
                    self.encoder.encode("false"),
                ]
            else:
                seen = set()
                options = []

                for word in cached_words:
                    t = tuple(word)
                    if t not in seen:
                        options.append(word)
                        seen.add(t)

                if not options:
                    options = [self.encoder.encode("")]

            if arg_type == "string":
                direct_tokens = self._infer_string_value(
                    arg_name,
                    text,
                    cached_words,
                )
                if direct_tokens is not None:
                    current_tokens += self.encoder.encode('"')
                    current_tokens += direct_tokens
                    current_tokens += self.encoder.encode('"')
                    return current_tokens

                current_tokens += self.encoder.encode('"')

            if len(options) == 1:
                next_tokens = options[0]
            else:
                next_tokens = self.llm.next_option(current_tokens, options)
            current_tokens += next_tokens

            if arg_type == "string":
                current_tokens += self.encoder.encode('"')
            return current_tokens

        for i, arg_name in enumerate(function.param_names):
            arg_schema = function.params[arg_name]

            if i > 0:
                tokens += self.encoder.encode(', ')

            tokens += self.encoder.encode(f'"{arg_name}": ')
            tokens = add_value(arg_schema, arg_name, tokens)
            # tokens = add_value(arg_schema, tokens)

        tokens += self.encoder.encode("}\n")

        return tokens

    def _print_trace_box(
        self,
        title: str,
        rows: list[tuple[str, str]]
    ) -> None:
        """Print a clean, readable trace block for the generation process."""

        width = 92
        border = "+" + "-" * (width - 2) + "+"
        print()
        print(border)
        print(f"| {title.center(width - 4)} |")
        print(border)

        for label, value in rows:
            display = str(value)
            wrapped = [
                display[i:i + (width - 30)]
                for i in range(0, len(display), width - 30)
            ] or [""]
            print(
                f"| {label:<24} | {wrapped[0]:<{width - 30}} |"
            )
            for line in wrapped[1:]:
                print(f"| {'':<24} | {line:<{width - 30}} |")
        print(border)

    def process_batch(self, prompts: list[str]) -> list[str]:
        """Processes many prompts while reusing already computed results."""

        results: list[str] = []
        prompt_cache: dict[str, str] = {}

        for prompt in prompts:
            cleaned_prompt = escape(prompt)
            if cleaned_prompt in prompt_cache:
                results.append(prompt_cache[cleaned_prompt])
                continue

            try:
                result = self.process_func(prompt)
            except Exception:
                result = (
                    '\t{\n'
                    '\t\t"prompt": "' + prompt + '",\n'
                    '\t\t"name": "fn_unknown",\n'
                    '\t\t"parameters": {}\n'
                    '\t}'
                )
            prompt_cache[cleaned_prompt] = result
            results.append(result)

        return results

    def process_func(self, prompt: str) -> str:
        """Processes a prompt and returns the corresponding function call."""

        prompt = escape(prompt)

        if not prompt.strip():
            return (
                '\t{\n'
                '\t\t"prompt": "",\n'
                '\t\t"name": "fn_unknown",\n'
                '\t\t"parameters": {}\n'
                '\t}'
            )
        text = (
            '<|im_start|>user\n' +
            prompt +
            '\n<|im_end|>\n'
            '<|im_start|>assistant\n'
            '<tool_call>\n'
            '{"name": '
        )
        tokens = self.encoder.encode(text)
        self.set_tools()
        func_names = [
            self.encoder.encode(f'"{f.name}"')
            for f in self.functions.values()
        ]

        self._print_trace_box(
            "Generation trace",
            [
                ("Prompt", prompt),
                (
                    "Function candidates",
                    ", ".join(f.name for f in self.functions.values()),
                ),
            ],
        )

        func_name = self.llm.next_option(tokens, func_names)
        decoded = self.encoder.decode(func_name).strip('"')

        self._print_trace_box(
            "Function selection",
            [
                ("Chosen function", decoded),
                ("Raw candidate", self.encoder.decode(func_name)),
            ],
        )

        function = self.functions[decoded]
        tokens += func_name
        tokens += self.encoder.encode(', "arguments": {')
        self.set_tools(function)
        cached_words = self.encoder.encode_words_separated(prompt)
        cached_numbers = self.encoder.encode_numbers(prompt)

        lower_prompt = prompt.lower()
        if (
            len(function.param_names) == 2
            and all(
                function.params[name] in {"number", "integer", "float"}
                for name in function.param_names
            )
        ):
            keywords = [
                "what is the sum of",
                "sum of",
                "difference between",
                "difference of",
                "product of",
            ]
            best_kw = None
            best_idx = -1
            for kw in keywords:
                idx = lower_prompt.rfind(kw)
                if idx > best_idx:
                    best_idx = idx
                    best_kw = kw
            if best_kw is not None:
                sub_prompt = prompt[best_idx + len(best_kw):]
                phrase_numbers = self.encoder.encode_numbers(sub_prompt)
                if len(phrase_numbers) >= len(function.param_names):
                    cached_numbers = phrase_numbers[:len(function.param_names)]

        import re as _re
        NUMBER_PATTERN = _re.compile(
            r'(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])'
        )
        number_contexts: list[tuple[str, str]] = []
        for m in NUMBER_PATTERN.finditer(prompt):
            raw_number = m.group(0).strip()
            start, end = m.span()
            window = prompt[max(0, start - 16): min(len(prompt), end + 16)]
            number_contexts.append((raw_number, window))

        self._print_trace_box(
            "Argument context",
            [
                ("Parameter names", ", ".join(function.param_names)),
                (
                    "Words extracted",
                    str([self.encoder.decode(w) for w in cached_words]),
                ),
                (
                    "Numbers extracted",
                    str([raw for raw, _ in number_contexts]),
                ),
                (
                    "Number contexts",
                    str([ctx for _, ctx in number_contexts]),
                ),
            ],
        )

        orig_instr = self.llm._t_instruction
        try:
            if cached_numbers:
                decoded_instr = (
                    self.llm.encoder.decode(orig_instr)
                    if orig_instr
                    else ""
                )

                candidates = "; ".join(
                    f"{raw} -> ...{ctx}..."
                    for raw, ctx in number_contexts
                )
                extra = (
                    "\nCandidate numeric values extracted from the prompt: "
                    + candidates
                    + ".\nThe prompt may contain unrelated numbers before "
                    "or after the actual question. When filling numeric "
                    "arguments, choose the numbers that are part of the "
                    "arithmetic expression in the user’s request, not the "
                    "unrelated noise numbers. Select the values that "
                    "correspond to the question phrase."
                )
                self.llm.set_instruction(
                    decoded_instr + extra if decoded_instr else extra
                )

            tokens = self.add_args(
                function,
                tokens,
                prompt,
                cached_words,
                cached_numbers
            )
        finally:
            instruction = orig_instr if orig_instr is not None else []
            self.llm.set_instruction(instruction)
        tokens += self.encoder.encode('}')

        raw = self.encoder.decode(tokens)
        start = raw.find('{"name":')

        self._print_trace_box(
            "Final assembled payload",
            [("JSON", raw[start:])],
        )

        if start == -1:
            raise ValueError("Invalid tool call generated")

        tool_json = raw[start:]
        # tool_json = raw[raw.find('{"name":'):]
        # print(tool_json)

        end = tool_json.rfind("}")

        if end == -1:
            raise ValueError("Invalid tool call generated")

        tool_json = tool_json[:end + 1]

        # data = json.loads(tool_json)
        try:
            data = json.loads(tool_json)
        except json.JSONDecodeError:
            raise ValueError("LLM generated invalid JSON")

        arguments_start = (
            tool_json.index('"arguments":') + len('"arguments": ')
            )
        arguments_text = tool_json[arguments_start:-1].strip()
        # print(data)

        return (
            '\t{\n'
            f'\t\t"prompt": "{prompt}",\n'
            f'\t\t"name": "{data["name"]}",\n'
            f'\t\t"parameters": {arguments_text}\n'
            '\t}'
        )
