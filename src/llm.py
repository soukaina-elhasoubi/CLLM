import numpy as np
from pydantic import BaseModel, PrivateAttr
from llm_sdk.llm_sdk import Small_LLM_Model

from src.encoder import Encoder


class LLM(BaseModel):
    _llm: Small_LLM_Model = PrivateAttr()
    _encoder: Encoder = PrivateAttr()
    _t_instruction: list[int] | None = PrivateAttr()

    # def __init__(self, llm: Small_LLM_Model, encoder: Encoder):
    #     super().__init__()
    #     self._llm = llm
    #     self._encoder = encoder
    #     self._t_instruction = None
    #     print('LLM created.')
    def __init__(self, llm: Small_LLM_Model, encoder: Encoder):
        super().__init__()

        self._llm = llm
        self._encoder = encoder
        self._t_instruction = None

        # cache utilisé pour éviter de recalculer plusieurs fois
        # les mêmes logits
        self._logits_cache = {}

        print("LLM created.")
    def next_token(self,
                   tokens: list[int],
                   mask: set[int] | None = None) -> int:
        """Returns the next token for the provided tokens."""

        logits = self.get_logits(tokens, mask)
        best_token = int(np.argmax(logits))
        return best_token

    def next_option(
        self,
        tokens: list[int],
        options: list[list[int]]
    ) -> list[int]:
        """Returns the best allowed option."""
        # result: list[int] = []
        # context = tokens + result
        # while options:
        #     allowed: set[int] = {option[0] for option in options}
        #     next_token = self.next_token(context, allowed)
        #     result.append(next_token)
        #     context.append(next_token)
        #     options = [
        #         option[1:]
        #         for option in options
        #         if option[0] == next_token and len(option) > 1
        #     ]
        # return result
        result = []
        context = list(tokens)

        while options:
            allowed = {o[0] for o in options}

            next_token = self.next_token(
                context + result,
                allowed
            )

            result.append(next_token)

            options = [
                o[1:]
                for o in options
                if o[0] == next_token and len(o) > 1
            ]

        return result

    def set_instruction(self, new: list[int] | str) -> None:
        """Sets the instruction with information for LLM."""

        if isinstance(new, str):
            new = self._encoder.encode(new)
        self._t_instruction = new

    # def get_logits(self,
    #                tokens: list[int],
    #                mask: set[int] | None = None) -> list[float]:
    #     """
    #     Returns the list of logits for provided tokens.
    #     Applies the mask optionally.
    #     """
    #     instr = self._t_instruction if self._t_instruction is not None else []
    #     # lgt = self._llm.get_logits_from_input_ids(instr + tokens)
    #     key = tuple(instr + tokens)

    #     if key in self._logits_cache:
    #         lgt = self._logits_cache[key]
    #     else:
    #         lgt = self._llm.get_logits_from_input_ids(instr + tokens)
    #         self._logits_cache[key] = lgt
    #     if mask is not None:
    #         lgt = self._apply_mask(mask, lgt)
    #     return lgt
    def get_logits(
        self,
        tokens: list[int],
        mask: set[int] | None = None
    ) -> list[float]:

        instr = tuple(self._t_instruction) if self._t_instruction else ()
        key = (instr, tuple(tokens))

        if key in self._logits_cache:
            logits = self._logits_cache[key]
        else:
            logits = self._llm.get_logits_from_input_ids(
                list(instr) + tokens
            )
            self._logits_cache[key] = logits

        if mask is not None:
            logits = self._apply_mask(mask, logits)

        return logits

    def _apply_mask(self,
                    mask: set[int] | list[int],
                    logits: list[float]) -> list[float]:
        """
        Returns logits with mask applied by setting all forbidden
        token scores to -infinity.
        """
        # masked = np.full_like(logits, -float('inf'))
        masked = np.full(
            len(logits),
            -float('inf'),
            dtype=float
        )
        for id in mask:
            if 0 <= id < len(logits):
                masked[id] = logits[id]

        return masked.tolist()
        # for id in mask:
        #     masked[id] = logits[id]
        # return list(masked)

    @property
    def encoder(self) -> Encoder:
        return self._encoder
