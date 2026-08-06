from pydantic import BaseModel, PrivateAttr
from typing import Any
import re


WORD_PATTERN = re.compile(r'''
    "(?:\\.|[^"])*"   |
    '(?:\\.|[^'])*'   |
    \S+
''', re.VERBOSE)

NUMBER_PATTERN = re.compile(
    r'(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])'
)


class Encoder(BaseModel):
    _trie: dict[str, Any] = PrivateAttr()
    _vocab: list[str | None] = PrivateAttr()

    def __init__(self, tokens: dict[str, int]):
        """Builds the vocabulary and trie from tokenizer tokens."""

        vocab: list[str | None] = [None] * len(tokens)
        trie: dict[str, Any] = {}

        print('Encoder: Building trie and vocab...')

        for word, token in tokens.items():
            vocab[token] = word
            node = trie

            for char in word:
                node = node.setdefault(char, {})

            node['token'] = token

        super().__init__()
        self._trie = trie
        self._vocab = vocab

        print('Encoder created.')

    def encode(self, text: str) -> list[int]:
        """Encodes text into tokenizer ids."""

        text = standart_to_special(text)

        ids: list[int] = []
        i = 0

        while i < len(text):
            node = self._trie
            match_id = None
            match_len = -1
            j = i

            while j < len(text) and text[j] in node:
                node = node[text[j]]
                j += 1

                if 'token' in node:
                    match_id = node['token']
                    match_len = j - i

            if match_id is not None:
                ids.append(match_id)
                i += match_len
            else:
                i += 1

        return ids

    def encode_words(self, text: str) -> set[int]:
        """Extracts unique token ids for the prompt words."""

        ids = set()
        words = WORD_PATTERN.findall(text)

        for word in words:
            word = word.strip('.,!?')
            word = word.strip('"\'')

            if not word:
                continue

            for token_id in self.encode(word):
                ids.add(token_id)

            spaced = self.encode(' ' + word)
            if spaced:
                ids.add(spaced[0])

        return ids

    def encode_words_separated(self, text: str) -> list[list[int]]:
        """Encodes prompt words as separate token sequences."""

        ids: list[list[int]] = []

        colon_match = re.search(r':\s*(.+)$', text)

        if colon_match:
            full_value = colon_match.group(1).strip()
            encoded = self.encode(full_value)

            if encoded:
                ids.append(encoded)

        unescaped = text.replace('\\"', '"')
        parts = WORD_PATTERN.findall(unescaped)

        for part in parts:
            part = part.strip('".,!?:;\\')
            part = part.strip("'")

            if not part:
                continue

            encoded = self.encode(part)

            if encoded:
                ids.append(encoded)

        return ids

    def encode_numbers(self, text: str) -> list[list[int]]:
        """Extracts and encodes numeric values from the prompt."""

        ids: list[list[int]] = []

        for n in NUMBER_PATTERN.findall(text):

            token_ids = self.encode(' ' + n)

            if not token_ids:
                token_ids = self.encode(n)

            if token_ids:
                ids.append(token_ids)
        return ids

    def extract_path(self, text: str) -> list[int]:
        """Extracts and encodes a filesystem path."""

        match = re.search(
            r'([A-Za-z]:\\[^\s]+|/[^\s]+)',
            text
        )

        if match:
            return self.encode(match.group(1))

        return self.encode("")

    def decode(self, tokens: list[int] | int) -> str:
        """Decodes tokenizer ids into text."""

        if isinstance(tokens, int):
            return self._vocab[tokens] or ''

        return special_to_standart(
            ''.join(
                self._vocab[t] or ''
                for t in tokens
            )
        )

    def debug(self, text: str) -> None:
        """Prints the encoding and decoding of a text."""

        ids = self.encode(text)
        print(text)
        print(ids)
        print([self.decode(i) for i in ids])
        print()


def special_to_standart(text: str) -> str:
    """Converts tokenizer special symbols to plain text."""

    return (
        text
        .replace('Ġ', ' ')
        .replace('Ċ', '\n')
        .replace('ĉ', '\t')
    )


def standart_to_special(text: str) -> str:
    """Converts plain text to tokenizer special symbols."""

    return (
        text
        .replace(' ', 'Ġ')
        .replace('\n', 'Ċ')
        .replace('\t', 'ĉ')
    )
