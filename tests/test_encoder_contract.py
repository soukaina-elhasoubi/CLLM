import unittest

from src.encoder import Encoder


class EncoderContractTestCase(unittest.TestCase):
    def test_encode_and_decode_round_trip(self) -> None:
        vocab = {
            "a": 0,
            "b": 1,
            "ab": 2,
            " ": 3,
            "": 4,
        }
        encoder = Encoder(vocab)

        encoded = encoder.encode("ab")
        self.assertEqual(encoded, [2])
        self.assertEqual(encoder.decode(encoded), "ab")

    def test_encode_words_separated_keeps_word_sequences(self) -> None:
        vocab = {
            "hello": 0,
            "world": 1,
            " ": 2,
            "hello world": 3,
        }
        encoder = Encoder(vocab)
        words = encoder.encode_words_separated("hello world")
        self.assertTrue(words)


if __name__ == "__main__":
    unittest.main()
