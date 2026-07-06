# from pydantic import BaseModel, PrivateAttr
# from typing import Any
# import re


# WORD_PATTERN = re.compile(r'''
#     "(?:\\.|[^"])*"   |
#     '(?:\\.|[^'])*'   |
#     \S+
# ''', re.VERBOSE)

# NUMBER_PATTERN = re.compile(
#     r'[+-]?\d+(?:\.\d+)?'
# )


# class Encoder(BaseModel):
#     _trie: dict[str, Any] = PrivateAttr()
#     _vocab: list[str | None] = PrivateAttr()

#     def __init__(self, tokens: dict[str, int]):
#         vocab: list[str | None] = [None] * len(tokens)
#         trie: dict[str, Any] = {}
#         print('Encoder: Building trie and vocab...')
#         for word, token in tokens.items():
#             vocab[token] = word
#             node = trie
#             for char in word:
#                 node = node.setdefault(char, {})
#             node['token'] = token
#         super().__init__()
#         self._trie = trie
#         self._vocab = vocab
#         print('Encoder created.')

#     def encode(self, text: str) -> list[int]:
#         """Translates human text to a list of tokens for LLM."""

#         text = standart_to_special(text)
#         ids: list[int] = []
#         for n in NUMBER_PATTERN.findall(text):
#             print("raw :", repr(n), self.encode(n))
#             print("spc :", repr(" " + n), self.encode(" " + n))
#         i = 0
#         while i < len(text):
#             node = self._trie
#             match_id = None
#             match_len = -1
#             j = i
#             while j < len(text) and text[j] in node:
#                 node = node[text[j]]
#                 j += 1
#                 if 'token' in node:
#                     match_id = node['token']
#                     match_len = j - i
#             if match_id is not None:
#                 ids.append(match_id)
#                 i += match_len
#             else:
#                 i += 1
#         return ids

#     def encode_words(self, text: str) -> set[int]:
#         """Returns all possible tokens from the string"""

#         ids = set()
#         words = WORD_PATTERN.findall(text)
#         for word in words:
#             word = word.strip('.,!?')
#             word = word.strip('"\'')
#             if not word:
#                 continue
#             for token_id in self.encode(word):
#                 ids.add(token_id)
#             ids.add(self.encode(' ' + word)[0])
#         return ids

#     def encode_words_separated(self, text: str) -> list[list[int]]:
#         """Returns tokenized prompt fragments."""
#         ids: list[list[int]] = []

#         colon_match = re.search(r':\s*(.+)$', text)
#         if colon_match:
#             full_value = colon_match.group(1).strip()
#             ids.append(self.encode(full_value))

#         unescaped = text.replace('\\"', '"')
#         parts = WORD_PATTERN.findall(unescaped)
#         for part in parts:
#             part = part.strip('".,!?:;\\')
#             part = part.strip("'")
#             if not part:
#                 continue
#             ids.append(self.encode(part))

#         return ids

#     # def encode_numbers(self, text: str) -> list[list[int]]:
#     #     ids = []

#     #     for n in NUMBER_PATTERN.findall(text):
#     #         ids.append(self.encode(n))

#     #     return ids
#     def encode_numbers(self, text: str) -> list[list[int]]:
#         ids = []

#         for n in NUMBER_PATTERN.findall(text):
#             token_ids = self.encode(n)
#             print(f"{n} -> {token_ids}")

#             if token_ids:
#                 ids.append(token_ids)

#         return ids
#     def decode(self, tokens: list[int] | int) -> str:
#         """Translates LLM tokens to human-readable text."""
#         if isinstance(tokens, int):
#             return self._vocab[tokens] or ''
#         return special_to_standart(
#             ''.join(self._vocab[t] or '' for t in tokens)
#         )


# def special_to_standart(text: str) -> str:
#     """
#     Replaces special AI characters for space, tab and new line with
#     standart human ones
#     """
#     return text.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')


# def standart_to_special(text: str) -> str:
#     """
#     Replaces standart spaces, tabs and new line chars with special ones
#     AI can understand
#     """
#     return text.replace(' ', 'Ġ').replace('\n', 'Ċ').replace('\t', 'ĉ')


from pydantic import BaseModel, PrivateAttr
from typing import Any
import re


WORD_PATTERN = re.compile(r'''
    "(?:\\.|[^"])*"   |
    '(?:\\.|[^'])*'   |
    \S+
''', re.VERBOSE)

# NUMBER_PATTERN = re.compile(
#     r'[+-]?\d+(?:\.\d+)?'
# )
NUMBER_PATTERN = re.compile(
    r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)'
)

class Encoder(BaseModel):
    _trie: dict[str, Any] = PrivateAttr()
    _vocab: list[str | None] = PrivateAttr()

    def __init__(self, tokens: dict[str, int]):
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
        """Translates human text to a list of tokens for LLM."""

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
        """Returns all possible tokens from the string"""

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
        """Returns tokenized prompt fragments."""

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
        """
        Returns tokenized numbers found in text.
        Tries first with a leading space because BBPE models
        often tokenize numbers as ' -3' rather than '-3'.
        """

        ids: list[list[int]] = []

        for n in NUMBER_PATTERN.findall(text):

            token_ids = self.encode(' ' + n)

            if not token_ids:
                token_ids = self.encode(n)

            # print(f'{n} -> {token_ids}')

            if token_ids:
                ids.append(token_ids)

        return ids

    def decode(self, tokens: list[int] | int) -> str:
        """Translates LLM tokens to human-readable text."""

        if isinstance(tokens, int):
            return self._vocab[tokens] or ''

        return special_to_standart(
            ''.join(
                self._vocab[t] or ''
                for t in tokens
            )
        )


def special_to_standart(text: str) -> str:
    return (
        text
        .replace('Ġ', ' ')
        .replace('Ċ', '\n')
        .replace('ĉ', '\t')
    )


def standart_to_special(text: str) -> str:
    return (
        text
        .replace(' ', 'Ġ')
        .replace('\n', 'Ċ')
        .replace('\t', 'ĉ')
    )
