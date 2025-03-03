import os
from typing import List, Callable, Tuple, Type, Dict
from src.ga.crossover import single_point_crossover
from src.ga.mutation import bit_flip_mutation
from src.ga.strategies import Strategy, AlwaysCooperate, AlwaysDefect, TitForTat, RandomStrategy
from src.ga.genetic_algorithm import GeneticAlgorithm
from src.utils.analysis import analyse_results


def run_ga(
    curr_dir: str = "",
    population_sizes: List[int] = [50, 75, 100],
    crossover_rates: List[float] = [0.7, 0.8, 0.9],
    crossover_funcs: List[
        Callable[[List[int], List[int]], Tuple[List[int], List[int]]]
    ] = [single_point_crossover],
    mutation_rates: List[float] = [0.01, 0.05, 0.1],
    mutation_funcs: List[Callable[[List[int], float], List[int]]] = [bit_flip_mutation],
    generations: int = 500,
    early_stop_threshold: int = 200,
    elitism_rate: float = 0.05,
    tournament_size: int = 3,
    opponents: List[Type[Strategy]] = [AlwaysCooperate, AlwaysDefect, TitForTat, RandomStrategy],
    memory_size: int = 3,
    rounds: int = 30,
    payoff_matrix: Dict[Tuple[int, int], Tuple[int, int]] = {
        (0, 0): (3, 3),  # Both cooperate
        (0, 1): (0, 5),  # Player cooperates, opponent defects
        (1, 0): (5, 0),  # Player defects, opponent cooperates
        (1, 1): (1, 1)   # Both defect
    }
) -> None:
    """
    Runs the genetic algorithm using various parameter combinations.

    Args:
        curr_dir: The base directory where the results are stored (default: "").
        population_sizes: A list of population sizes to test (default: [50, 75, 100]).
        crossover_rates: A list of crossover rates to test (default: [0.7, 0.8, 0.9]).
        crossover_funcs: A list of crossover functions to test (default: [single_point_crossover]).
        mutation_rates: A list of mutation rates to test (default: [0.01, 0.05, 0.1]).
        mutation_funcs: A list of mutation functions to test (default: [bit_flip_mutation]).
        generations: The number of generations to run the algorithm for (default: 500).
        early_stop_threshold: The number of generations without improvement before stopping (
            default: 200).
        elitism_rate: The proportion of individuals to retain through elitism (default: 0.05).
        tournament_size: The size of the tournament for selection (default: 3).
        opponents: A list of opponent strategy classes (default: [AlwaysCooperate, AlwaysDefect,
            TitForTat, RandomStrategy]).
        memory_size: The number of past opponent moves each strategy considers (default: 3).
        rounds: The number of IPD rounds to play (default: 30).
        payoff_matrix: A dictionary representing a payoff matrix.
    """
    for population_size in population_sizes:
        for crossover_rate in crossover_rates:
            for mutation_rate in mutation_rates:
                for crossover_func in crossover_funcs:
                    for mutation_func in mutation_funcs:
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
                            payoff_matrix
                        )
                        ga.evolve()

                        results_path = os.path.join(
                            curr_dir,
                            f"data/results/{population_size}_pop_{crossover_rate}_"
                            f"{crossover_func.__name__}_{mutation_rate}_{mutation_func.__name__}"
                            ".json"
                        )
                        ga.save_results(results_path)


if __name__ == "__main__":
    run_ga()
    analyse_results("data/results/")
