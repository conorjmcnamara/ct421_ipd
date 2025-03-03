from typing import List, Dict, Tuple


def fitness(
    player: List[int],
    opponents: List[List[int]],
    memory_size: int,
    rounds: int,
    payoff_matrix: Dict[Tuple[int, int], Tuple[int, int]]
) -> int:
    """
    Evaluates a player's fitness based on performance against opponents.

    Args:
        player: A bit string representing the player strategy to evaluate.
        opponents: The opponent bit string representations to evaluate against.
        memory_size: The number of past opponent moves each strategy considers.
        rounds: The number of IPD rounds to play.
        payoff_matrix: A dictionary representing a payoff matrix.

    Returns:
        The accumulated score achieved by the player against all the opponents.
    """
    return sum(
        play_ipd(player, opponent, memory_size, rounds, payoff_matrix)[0] for opponent in opponents
    )


def play_ipd(
    player: List[int],
    opponent: List[int],
    memory_size: int,
    rounds: int,
    payoff_matrix: Dict[Tuple[int, int], Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Simulates an Iterated Prisoner's Dilemma match between two players.

    Each player's decision is determined by indexing their bit-string representation using the
    opponent's past moves (history).

    Args:
        player: A bit string representing the player strategy.
        opponent: A bit string representing the opponent strategy.
        memory_size: The number of past opponent moves each strategy considers.
        rounds: The number of rounds to play.
        payoff_matrix: A dictionary representing a payoff matrix.

    Returns:
        A tuple (player_score, opponent_score) with the accumulated scores.
    """
    player_score = 0
    opponent_score = 0
    player_history = []
    opponent_history = []

    for _ in range(rounds):
        player_idx = get_move_index(opponent_history, memory_size)
        opponent_idx = get_move_index(player_history, memory_size)

        player_move = player[player_idx]
        opponent_move = opponent[opponent_idx]

        score_player, score_opponent = payoff_matrix[(player_move, opponent_move)]
        player_score += score_player
        opponent_score += score_opponent

        player_history.append(player_move)
        opponent_history.append(opponent_move)

    return player_score, opponent_score


def get_move_index(history: List[int], memory_size: int) -> int:
    """
    Computes the move index into the bit string representation based on the opponent's history.

    To compute the index:
    - First, consider the effective history length, L, as min(len(history), memory_size). This
        enures that only the most recent moves are used.
    - Next, compute an offset that accounts for all histories shorter than L. This is computed as
        (2^L) -1.
    - Then, convert the effective history moves into a binary number
    - Finally, sum the offset and this binary number

    Args:
        history: List of past moves (each 0 or 1).
        memory_size: The number of past opponent moves each strategy considers.

    Returns:
        The computed move index into the bit string representation.
    """
    # Determine effective history length
    L = min(len(history), memory_size)

    if L == 0:
        # First move always corresponds to index 0
        return 0

    # Use only the last L moves
    effective_history = history[-L:]

    offset = (2 ** L) - 1

    # Convert effective history (list of bits) to its integer value
    binary_string = "".join(map(str, effective_history))
    binary_val = int(binary_string, 2)

    return offset + binary_val
