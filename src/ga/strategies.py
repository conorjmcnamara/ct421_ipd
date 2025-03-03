import random
from abc import ABC, abstractmethod
from typing import List, Type


class Strategy(ABC):
    """
    Abstract base class representing a strategy.
    """

    @abstractmethod
    def decide(self, history: List[int]) -> int:
        """
        Decide what action to take based on a history of past opponent moves.

        Args:
            history: A list representing past opponent moves (0 for cooperate, 1 for defect).

        Returns:
            The decision (0 for cooperate, 1 for defect).
        """
        pass


class AlwaysCooperate(Strategy):
    def decide(self, history: List[int]) -> int:
        return 0


class AlwaysDefect(Strategy):
    def decide(self, history: List[int]) -> int:
        return 1


class TitForTat(Strategy):
    def decide(self, history: List[int]) -> int:
        return history[-1] if history else 0


class TwoTitsForTat(Strategy):
    def decide(self, history: List[int]) -> int:
        return 1 if 1 in history[-2:] else 0


class GrimTrigger(Strategy):
    def decide(self, history: List[int]) -> int:
        return 1 if 1 in history else 0


class RandomStrategy(Strategy):
    def decide(self, history: List[int]) -> int:
        return random.choice([0, 1])


def generate_bit_representation(
    strategy: Strategy,
    memory_size: int
) -> List[int]:
    """
    Generates a bit string representation for a strategy based on 'memory_size' past opponent moves.

    - The first move is independent of history.
    - The next moves consider all possible history lengths up to 'memory_size'.
    - The bit string length is sum(2^i) for i = 0 to memory_size.

    Args:
        strategy: The strategy object that implements the 'decide' method.
        memory_size: The number of past opponent moves to consider for the strategy's decision.

    Returns:
        Bit string representing the strategy's responses for all possible histories.
    """
    bit_representation = []

    # Handle the decision for the first move (no history)
    first_move = strategy.decide([])
    bit_representation.append(first_move)

    # Handle decisions for partial histories (before full memory is reached)
    for i in range(1, memory_size):
        num_partial_histories = 2 ** i

        for j in range(num_partial_histories):
            bit_string = f"{j:0{i}b}"
            history = [int(bit) for bit in bit_string]
            decision = strategy.decide(history)
            bit_representation.append(decision)

    # Handle decisions for full memory histories
    num_histories = 2 ** memory_size
    for i in range(num_histories):
        bit_string = f"{i:0{memory_size}b}"
        history = [int(bit) for bit in bit_string]

        decision = strategy.decide(history)
        bit_representation.append(decision)

    return bit_representation


def get_bit_representations_for_strategies(
    strategies: List[Type[Strategy]],
    memory_size: int
) -> List[List[int]]:
    """
    Generates bit string representations for multiple strategies based on memory size.

    Args:
        strategies: A list of strategy classes.
        memory_size: The number of past opponent moves to consider.

    Returns:
        A list of bit strings representing each strategy.
    """
    strategy_instances = [strategy() for strategy in strategies]
    bit_representations = [
        generate_bit_representation(strategy, memory_size) for strategy in strategy_instances
    ]
    return bit_representations
