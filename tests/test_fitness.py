from src.ga.fitness import play_ipd
from src.ga.strategies import generate_bit_representation, AlwaysDefect, AlwaysCooperate, TitForTat


def test_play_ipd():
    memory_sizes = [2, 1, 2, 3]
    rounds = 6
    payoff_matrix = {
        (0, 0): (3, 3),  # Both cooperate
        (0, 1): (0, 5),  # Player cooperates, opponent defects
        (1, 0): (5, 0),  # Player defects, opponent cooperates
        (1, 1): (1, 1)   # Both defect
    }

    bit_representations = [
        (
            generate_bit_representation(AlwaysDefect(), 2),
            generate_bit_representation(AlwaysCooperate(), 2),
        ),
        (
            generate_bit_representation(TitForTat(), 1),
            generate_bit_representation(AlwaysDefect(), 1)
        ),
        (
            generate_bit_representation(TitForTat(), 2),
            generate_bit_representation(AlwaysDefect(), 2)
        ),
        (
            generate_bit_representation(TitForTat(), 3),
            [
                1,                      # First move
                1, 1,                   # C, D
                0, 0, 1, 0,             # CC, CD, DC, DD
                1, 1, 0, 1, 0, 0, 1, 1  # CCC, CCD, CDC, CDD, DCC, DCD, DDC, DDD
            ]
        )
    ]

    expected_scores = [
        (30, 0),
        (5, 10),
        (5, 10),
        (12, 12),
    ]

    """
    Round breakdown for memory size 3 case:
    Round 1 (start):    Player: 0, Opponent: 1 -> Scores: (0, 5)
    Round 2 (mem 1):    Player: 1, Opponent: 1 -> Scores: (1, 1)
    Round 3 (mem 2):    Player: 1, Opponent: 0 -> Scores: (5, 0)
    Round 4 (mem 3):    Player: 0, Opponent: 1 -> Scores: (0, 5)
    Round 5 (mem 3):    Player: 1, Opponent: 1 -> Scores: (1, 1)
    Round 6 (mem 3):    Player: 1, Opponent: 0 -> Scores: (5, 0)
    Total score:        Player: 12 Opponent: 12
    """

    for i, (player, opponent) in enumerate(bit_representations):
        player_score, opponent_score = play_ipd(
            player,
            opponent,
            memory_sizes[i],
            rounds,
            payoff_matrix
        )

        assert player_score == expected_scores[i][0]
        assert opponent_score == expected_scores[i][1]
