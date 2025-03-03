from src.ga.strategies import generate_bit_representation, TitForTat


def test_generate_bit_representation():
    memory_sizes = [1, 2, 3]
    expected_bit_representation_lengths = [
        sum(2 ** i for i in range(memory_size + 1)) for memory_size in memory_sizes
    ]
    expected_bit_representations = [
        [
            0,                      # First move
            0, 1                    # C, D
        ],
        [
            0,                      # First move
            0, 1,                   # C, D
            0, 1, 0, 1              # CC, CD, DC, DD
        ],
        [
            0,                      # First move
            0, 1,                   # C, D
            0, 1, 0, 1,             # CC, CD, DC, DD
            0, 1, 0, 1, 0, 1, 0, 1  # CCC, CCD, CDC, CDD, DCC, DCD, DDC, DDD
        ]
    ]

    for i, memory_size in enumerate(memory_sizes):
        bit_representation = generate_bit_representation(TitForTat(), memory_size)

        assert len(bit_representation) == expected_bit_representation_lengths[i]
        assert bit_representation == expected_bit_representations[i]
