import random
from typing import List, Tuple
from enum import Enum


class Strategy:
    """
    Represents an IPD strategy using a 4-bit list.
    """
    OUTCOMES = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (C,C), (C,D), (D,C), (D,D)

    def __init__(self, bitstring: List[int]):
        """
        Initialises a strategy.

        Args:
            bitstring: A 4-bit list representing the strategy's response to each outcome.

        Raises:
            ValueError: If the bitstring is not exactly 4 bits long.
        """
        if len(bitstring) != 4:
            raise ValueError("Strategy must be a 4-bit list.")
        self.bitstring = bitstring

    def move(self, last_round: Tuple[int, int]) -> int:
        """
        Determines the next move based on the last round's outcome.

        Args:
            last_round: The previous round's moves (player, opponent).

        Returns:
            The next move (0 for cooperate, 1 for defect).
        """
        return self.bitstring[self.OUTCOMES.index(last_round)]


class FixedStrategies(Enum):
    """
    Predefined fixed strategies.
    """
    ALWAYS_COOPERATE = Strategy([0, 0, 0, 0])
    ALWAYS_DEFECT = Strategy([1, 1, 1, 1])
    TIT_FOR_TAT = Strategy([0, 1, 0, 1])


def random_strategy() -> Strategy:
    """
    Generates a random 4-bit Strategy.

    Returns:
        A randomly generated Strategy.
    """
    return Strategy([random.choice([0, 1]) for _ in range(4)])
