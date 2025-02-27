import random
from typing import Tuple
from src.ga.representation import Strategy


def single_point_crossover(parent1: Strategy, parent2: Strategy) -> Tuple[Strategy, Strategy]:
    """
    Performs single-point crossover between two parents to produce two offspring.

    Args:
        parent1: The first parent.
        parent2: The second parent.

    Returns:
        A tuple containing two offspring.
    """
    point = random.randint(1, len(parent1.bitstring) - 1)
    offspring1 = Strategy(parent1.bitstring[:point] + parent2.bitstring[point:])
    offspring2 = Strategy(parent2.bitstring[:point] + parent1.bitstring[point:])

    return offspring1, offspring2
