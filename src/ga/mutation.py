import random
from src.ga.representation import Strategy


def bit_flip_mutation(individual: Strategy, mutation_rate: float) -> Strategy:
    """
    Applies bit flip mutation to an individual.

    Args:
        strategy: The individual to mutate.
        mutation_rate: Probability of mutating each bit.

    Returns:
        The mutated individual.
    """
    return Strategy(
        [bit if random.random() > mutation_rate else 1 - bit for bit in individual.bitstring]
    )
