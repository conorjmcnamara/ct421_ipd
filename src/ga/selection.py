import copy
import random
from typing import List
from src.ga.representation import Strategy


def elitism(
    population: List[Strategy],
    fitness_scores: List[int],
    elitism_count: int
) -> List[Strategy]:
    """
    Selects the top `elitism_count` individuals from a population based on their fitness scores.

    Args:
        population: A list of individuals.
        fitness_scores: A list of fitness scores associated with each individual in the population.
        elitism_count: The number of individuals to select.

    Returns:
        A list of the top `elitism_count` individuals.
    """
    return [
        copy.deepcopy(population[i]) for i in sorted(
            range(len(fitness_scores)),
            key=lambda j: fitness_scores[j]
        )[:elitism_count]
    ]


def tournament_selection(
    population: List[Strategy],
    fitness_scores: List[int],
    tournament_size: int,
    num_rounds: int
) -> List[Strategy]:
    """
    Selects individuals from a population using tournament selection, where each tournament selects
    a winner among a random sample of individuals based on their fitness scores. This is repeated
    for a specified number of rounds.

    Args:
        population: A list of individuals.
        fitness_scores: A list of fitness scores associated with each individual in the population.
        tournament_size: The number of individuals randomly selected for each tournament.
        num_rounds: The number of rounds of tournament selection to perform.

    Returns:
        A list of individuals selected through tournament selection.
    """
    selected = []
    for _ in range(num_rounds):
        competitors = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = max(competitors, key=lambda competitor: competitor[1])
        selected.append(winner[0])
    return selected
