import os
from typing import List, Callable, Tuple
from src.ga.representation import Strategy
from src.ga.crossover import single_point_crossover
from src.ga.mutation import bit_flip_mutation
from src.ga.genetic_algorithm import GeneticAlgorithm
from src.utils.analysis import analyse_results


def run_ga(
    curr_dir: str = "",
    population_sizes: List[int] = [50, 75, 100],
    crossover_rates: List[float] = [0.7, 0.8, 0.9],
    crossover_funcs: List[
        Callable[[Strategy, Strategy], Tuple[Strategy, Strategy]]
    ] = [single_point_crossover],
    mutation_rates: List[float] = [0.05, 0.1, 0.2],
    mutation_funcs: List[Callable[[Strategy, float], Strategy]] = [bit_flip_mutation],
    generations: int = 500,
    early_stop_threshold: int = 100,
    elitism_rate: float = 0.01,
    tournament_size: int = 3,
    ipd_rounds: int = 20,
) -> None:
    """

    Args:
        curr_dir: The base directory where the results are stored (default: "").
        population_sizes: A list of population sizes to test (default: [50, 75, 100]).
        crossover_rates: A list of crossover rates to test (default: [0.7, 0.8, 0.9]).
        crossover_funcs: A list of crossover functions to test (default: [single_point_crossover]).
        mutation_rates: A list of mutation rates to test (default: [0.05, 0.1, 0.2]).
        mutation_funcs: A list of mutation functions to test (default: [bit_flip_mutation]).
        generations: The number of generations to run the algorithm for (default: 500).
        early_stop_threshold: The number of generations without improvement before stopping (
            default: 100).
        elitism_rate: The proportion of individuals to retain through elitism (default: 0.02).
        tournament_size: The size of the tournament for selection (default: 3).
        ipd_rounds: The number of IPD rounds (default: 20).
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
                            ipd_rounds
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
