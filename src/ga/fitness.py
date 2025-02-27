from src.ga.representation import Strategy, FixedStrategies


def fitness(individual: Strategy, ipd_rounds: int) -> int:
    """
    Evaluates an individual's fitness based on performance against fixed strategies.

    Args:
        individual: The individual to evaluate.
        ipd_rounds: The number of IPD rounds.

    Returns:
        The total score against fixed strategies.
    """
    opponents = [strategy.value for strategy in FixedStrategies]
    return sum(play_ipd(individual, opponent, ipd_rounds) for opponent in opponents)


def play_ipd(player: Strategy, opponent: Strategy, rounds: int) -> int:
    """
    Simulates an Iterated Prisoner's Dilemma match.

    Args:
        player: The player's strategy.
        opponent: The opponent's strategy.
        rounds: The number of rounds.

    Returns:
        The total score achieved by the player's strategy.
    """
    score = 0
    last_round = (0, 0)

    for _ in range(rounds):
        p_move = player.move(last_round)
        o_move = opponent.move(last_round)

        if (p_move, o_move) == (0, 0):
            score += 3
        elif (p_move, o_move) == (0, 1):
            score += 0
        elif (p_move, o_move) == (1, 0):
            score += 5
        elif (p_move, o_move) == (1, 1):
            score += 1

        last_round = (p_move, o_move)

    return score
