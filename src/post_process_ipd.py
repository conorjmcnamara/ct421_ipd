import json
from typing import List, Type, Dict, Tuple
from src.ga.strategies import (
    Strategy,
    AlwaysCooperate,
    AlwaysDefect,
    TitForTat,
    TitForTwoTats,
    GrimTrigger,
    get_bit_representations_for_strategies
)
from src.ga.fitness import play_ipd


def post_process_ipd(
    results_path: str,
    evolved_strategy_path: str,
    strategies: List[Type[Strategy]] = [
        AlwaysCooperate, AlwaysDefect, TitForTat, TitForTwoTats, GrimTrigger
    ],
    memory_size: int = 2,
    rounds: int = 50,
    payoff_matrix: Dict[Tuple[int, int], Tuple[int, int]] = {
        (0, 0): (3, 3),  # Both cooperate
        (0, 1): (0, 5),  # Player cooperates, opponent defects
        (1, 0): (5, 0),  # Player defects, opponent cooperates
        (1, 1): (1, 1)   # Both defect
    },
    noise_rate: float = 0.0
) -> None:
    """
    Runs a series of IPD games where each strategy plays against every other strategy, including
    the evolved strategy.

    Args:
        results_path: The path where the results will be saved.
        evolved_strategy_path: The path to the GA results JSON file.
        strategies: A list of strategy classes to play against each other.
        memory_size: The number of past opponent moves each strategy considers (default: 2).
        rounds: The number of IPD rounds to play (default: 50).
        payoff_matrix: A dictionary representing a payoff matrix.
        noise_rate: The probability of flipping a player's move (default: 0.0).
    """
    # Determine the evolved strategy
    with open(evolved_strategy_path, 'r') as file:
        data = json.load(file)
        best_solutions = data["results"]["best_solutions"]

        # Find the strategy with the highest count
        best_strategy_str = max(best_solutions, key=best_solutions.get)
        evolved_strategy = [int(bit) for bit in best_strategy_str.strip("()").split(", ")]

    # Create provided strategies
    strategy_representations = get_bit_representations_for_strategies(strategies, memory_size)
    strategy_names = [s.__name__ for s in strategies] + ["Evolved"]
    strategy_representations.append(evolved_strategy)

    # Play IPD
    results = {
        name: {
            "overall_score": 0,
            "bit_representation": strategy_representations[i],
            "vs_opponents": {}
        } for i, name in enumerate(strategy_names)
    }

    for i, (player, player_name) in enumerate(zip(strategy_representations, strategy_names)):
        for j, (opponent, opponent_name) in enumerate(zip(
            strategy_representations,
            strategy_names
        )):
            if i == j:
                continue

            player_score, opponent_score = play_ipd(
                player, opponent, memory_size, rounds, payoff_matrix, noise_rate
            )

            results[player_name]["overall_score"] += player_score
            results[opponent_name]["overall_score"] += opponent_score

            results[player_name]["vs_opponents"][opponent_name] = player_score
            results[opponent_name]["vs_opponents"][player_name] = opponent_score

    # Sort results
    for strategy in results:
        results[strategy]["vs_opponents"] = dict(
            sorted(results[strategy]["vs_opponents"].items(), key=lambda x: x[1], reverse=True)
        )
        results[strategy]["vs_opponents"]["total"] = sum(results[strategy]["vs_opponents"].values())

    ranked_results = dict(
        sorted(results.items(), key=lambda x: x[1]["overall_score"], reverse=True)
    )

    with open(results_path, 'w') as file:
        json.dump(ranked_results, file, indent=4)
