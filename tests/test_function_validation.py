import unittest

from src.encoder import Encoder
from src.function import Function


class FunctionValidationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        vocab = {
            "a": 0,
            "b": 1,
            "c": 2,
            "string": 3,
            "number": 4,
            "integer": 5,
            "boolean": 6,
            "array": 8,
            "ok": 9,
        }
        self.encoder = Encoder(vocab)

    def test_blank_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Function(
                {
                    "name": "   ",
                    "description": "ok",
                    "parameters": {},
                    "returns": {"type": "string"},
                },
                self.encoder,
            )

    def test_blank_description_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Function(
                {
                    "name": "fn_ok",
                    "description": "   ",
                    "parameters": {},
                    "returns": {"type": "string"},
                },
                self.encoder,
            )

    def test_blank_parameter_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Function(
                {
                    "name": "fn_ok",
                    "description": "ok",
                    "parameters": {"x": {"type": "   "}},
                    "returns": {"type": "string"},
                },
                self.encoder,
            )


if __name__ == "__main__":
    unittest.main()
