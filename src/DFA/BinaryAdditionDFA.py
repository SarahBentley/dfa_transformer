import numpy as np
from .DFA import DFA

# Define the DFA parameters
def generate_binary_lists(length=3):
    if length <= 0:
        return [[]]

    smaller_lists = generate_binary_lists(length - 1)
    result = []

    for lst in smaller_lists:
        result.append(lst + [0])
        result.append(lst + [1])

    return result

alphabet = [ str(lst) for lst in generate_binary_lists()]
start_state = '<init>'
accept_states = ['<equal>']
states = ['<init>', '<equal>', '<carry>', '<unequal>']

# Define the transition matrix. Should be of shape (|states|, |states|, |alphabet|)
transition_matrix = np.array([
    # Transition for init state
    [[0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 1, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 1, 0, 1, 0, 0, 1]],
    # Transition for equal state
    [[0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 1, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 1, 0, 1, 0, 0, 1]],
    # Transition for carry state
    [[0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 1, 0, 0, 1],
     [1, 0, 0, 1, 0, 1, 1, 0]],
    # Transition for unequal state
    [[0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 1, 1, 1, 1, 1, 1]]
])
# Transpose the first two dimensions
transition_matrix = np.transpose(transition_matrix, (1, 0, 2))

# Create the DFA instance
BinaryAdditionDFA = DFA(alphabet, start_state, accept_states, states, transition_matrix)

if __name__ == "__main__":
    # Test the DFA with some input strings
    random_array = np.random.choice(BinaryAdditionDFA.alphabet, size=8)  # Generate a random array of integers (0 or 1)

    print("input sequence:", random_array)
    # Test the DFA with some input strings
    seq = BinaryAdditionDFA.process_sequence(random_array) # Output: True (even number of 1s)
    print(seq)