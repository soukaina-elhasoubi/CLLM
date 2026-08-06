import json
from typing import Any

from pydantic import BaseModel, PrivateAttr

from src.encoder import Encoder


SUPPORTED_TYPES = {
    "string",
    "number",
    "integer",
    "float",
    "boolean",
}


class Function(BaseModel):
    _name: str = PrivateAttr()
    _t_name: list[int] = PrivateAttr()

    _description: str = PrivateAttr()
    _t_description: list[int] = PrivateAttr()

    _params: dict[str, str] = PrivateAttr()
    _t_params: dict[str, list[int]] = PrivateAttr()

    _return_type: str = PrivateAttr()

    _t_definition: list[int] = PrivateAttr()

    def __init__(
        self,
        function: dict[str, Any],
        encoder: Encoder,
    ):
        """Create a validated function definition."""

        super().__init__()

        if not isinstance(function, dict):
            raise ValueError("Function definition must be an object.")

        name = function.get("name")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Missing or invalid function name.")

        self._name = name.strip()
        self._t_name = encoder.encode(self._name)

        description = function.get("description")

        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"Function '{self._name}' has no valid description."
            )

        self._description = description.strip()
        self._t_description = encoder.encode(self._description)

        parameters = function.get("parameters")

        if not isinstance(parameters, dict):
            raise ValueError(
                f"Function '{self._name}' has invalid parameters."
            )

        self._params = {}
        self._t_params = {}

        for param_name, info in parameters.items():

            if not isinstance(param_name, str) or not param_name.strip():
                raise ValueError(
                    f"Function '{self._name}' has an invalid parameter name."
                )

            if not isinstance(info, dict):
                raise ValueError(
                    f"Parameter '{param_name}' must be an object."
                )

            param_type = info.get("type")

            if not isinstance(param_type, str) or not param_type.strip():
                raise ValueError(
                    f"Parameter '{param_name}' has no type."
                )

            normalized_type = param_type.strip()
            if normalized_type not in SUPPORTED_TYPES:
                raise ValueError(
                    f"Unsupported type '{normalized_type}' "
                    f"for parameter '{param_name}'."
                )

            param_key = param_name.strip()
            self._params[param_key] = normalized_type
            self._t_params[param_key] = encoder.encode(normalized_type)

        returns = function.get("returns")

        if not isinstance(returns, dict):
            raise ValueError(
                f"Function '{self._name}' has invalid returns."
            )

        return_type = returns.get("type")

        if not isinstance(return_type, str) or not return_type.strip():
            raise ValueError(
                f"Function '{self._name}' has no return type."
            )

        normalized_return_type = return_type.strip()
        if normalized_return_type not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported return type '{normalized_return_type}'."
            )

        self._return_type = normalized_return_type

        self._t_definition = encoder.encode(
            self._to_tool_schema()
        )

    def _to_tool_schema(self) -> str:
        """Return the JSON tool schema."""

        return json.dumps(
            {
                "name": self._name,
                "description": self._description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        key: {"type": value}
                        for key, value in self._params.items()
                    },
                    "required": list(self._params.keys()),
                },
            },
            separators=(",", ":"),
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def t_name(self) -> list[int]:
        return self._t_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def t_description(self) -> list[int]:
        return self._t_description

    @property
    def params(self) -> dict[str, str]:
        return self._params

    @property
    def t_params(self) -> dict[str, list[int]]:
        return self._t_params

    @property
    def param_names(self) -> list[str]:
        return list(self._params.keys())

    @property
    def return_type(self) -> str:
        return self._return_type

    @property
    def t_definition(self) -> list[int]:
        return self._t_definition
