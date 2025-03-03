import random
from typing import List


def bit_flip_mutation(individual: List[int], mutation_rate: float) -> List[int]:
    """
    Applies bit flip mutation to an individual.

    Args:
        strategy: The individual to mutate.
        mutation_rate: Probability of mutating each bit.

    Returns:
        The mutated individual.
    """
    return [bit if random.random() > mutation_rate else 1 - bit for bit in individual]
