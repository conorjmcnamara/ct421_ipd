import random
from typing import List, Tuple


def single_point_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Performs single-point crossover between two parents to produce two offspring.

    Args:
        parent1: The first parent.
        parent2: The second parent.

    Returns:
        A tuple containing two offspring.
    """
    point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]

    return offspring1, offspring2
