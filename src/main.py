import os
from typing import List, Callable, Tuple, Type, Dict
from src.ga.crossover import single_point_crossover
from src.ga.mutation import bit_flip_mutation
from src.ga.strategies import (
    Strategy,
    AlwaysCooperate,
    AlwaysDefect,
    TitForTat,
    TitForTwoTats,
    GrimTrigger
)
from src.ga.genetic_algorithm import GeneticAlgorithm


def run_ga(
    curr_dir: str = "",
    population_sizes: List[int] = [75],
    crossover_rates: List[float] = [0.8],
    crossover_funcs: List[
        Callable[[List[int], List[int]], Tuple[List[int], List[int]]]
    ] = [single_point_crossover],
    mutation_rates: List[float] = [0.05],
    mutation_funcs: List[Callable[[List[int], float], List[int]]] = [bit_flip_mutation],
    generations: int = 500,
    early_stop_threshold: int = 200,
    elitism_rate: float = 0.05,
    tournament_size: int = 3,
    opponent_environments: List[List[Type[Strategy]]] = [
        [AlwaysCooperate],
        [AlwaysDefect],
        [AlwaysCooperate, AlwaysDefect],
        [TitForTat],
        [TitForTat, AlwaysDefect],
        [TitForTwoTats],
        [GrimTrigger],
        [AlwaysCooperate, AlwaysDefect, TitForTat, TitForTwoTats, GrimTrigger]
    ],
    memory_size: int = 2,
    rounds: int = 50,
    payoff_matrix: Dict[Tuple[int, int], Tuple[int, int]] = {
        (0, 0): (3, 3),  # Both cooperate
        (0, 1): (0, 5),  # Player cooperates, opponent defects
        (1, 0): (5, 0),  # Player defects, opponent cooperates
        (1, 1): (1, 1)   # Both defect
    },
    noise_rates: List[float] = [0, 0.05, 0.1, 0.2],
    co_evolutions: List[bool] = [False, True]
) -> None:
    """
    Runs the genetic algorithm using various parameter combinations.

    Args:
        curr_dir: The base directory where the results are stored (default: "").
        population_sizes: A list of population sizes to test (default: [75]).
        crossover_rates: A list of crossover rates to test (default: [0.8]).
        crossover_funcs: A list of crossover functions to test (default: [single_point_crossover]).
        mutation_rates: A list of mutation rates to test (default: [0.05]).
        mutation_funcs: A list of mutation functions to test (default: [bit_flip_mutation]).
        generations: The number of generations to run the algorithm for (default: 500).
        early_stop_threshold: The number of generations without improvement before stopping (
            default: 200).
        elitism_rate: The proportion of individuals to retain through elitism (default: 0.05).
        tournament_size: The size of the tournament for selection (default: 3).
        opponent_environments: A nested list of opponent strategy classes.
        memory_size: The number of past opponent moves each strategy considers (default: 2).
        rounds: The number of IPD rounds to play (default: 50).
        payoff_matrix: A dictionary representing a payoff matrix.
        noise_rates: A list of noise rates to test (default: [0, 0.05, 0.1, 0.2]).
        co_evolutions: A list of co-evolution scenarios to test (default: [False, True]).
    """
    for i, opponents in enumerate(opponent_environments):
        for population_size in population_sizes:
            for crossover_rate in crossover_rates:
                for mutation_rate in mutation_rates:
                    for crossover_func in crossover_funcs:
                        for mutation_func in mutation_funcs:
                            for noise_rate in noise_rates:
                                for co_evolution in co_evolutions:
                                    ga = GeneticAlgorithm(
                                        population_size,
                                        crossover_rate,
                                        crossover_func,
                                        mutation_rate,
                                        mutation_func,
                                        generations,
                                        early_stop_threshold,
                                        elitism_rate,
                                        tournament_size,
                                        opponents,
                                        memory_size,
                                        rounds,
                                        payoff_matrix,
                                        noise_rate,
                                        co_evolution
                                    )
                                    ga.evolve()

                                    if co_evolution:
                                        results_path = os.path.join(
                                            curr_dir,
                                            f"data/results/co-evo/env_{i}/{memory_size}_mem_"
                                            f"{population_size}_pop_{crossover_rate}_"
                                            f"{crossover_func.__name__}_{mutation_rate}_"
                                            f"{mutation_func.__name__}_{noise_rate}_noise_"
                                            f"{co_evolution}_co-evolution.json"
                                        )
                                    else:
                                        results_path = os.path.join(
                                            curr_dir,
                                            f"data/results/env_{i}/{memory_size}_mem_"
                                            f"{population_size}_pop_{crossover_rate}_"
                                            f"{crossover_func.__name__}_{mutation_rate}_"
                                            f"{mutation_func.__name__}_{noise_rate}_noise_"
                                            f"{co_evolution}_co-evolution.json"
                                        )

                                    ga.save_results(results_path)


if __name__ == "__main__":
    run_ga()
