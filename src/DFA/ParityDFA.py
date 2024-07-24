from .DFA import DFA
import numpy as np

# Define the DFA parameters
alphabet = ['0', '1']
start_state = '<even>'
accept_states = ['<even>']
states = ['<even>', '<odd>']

# Define the transition matrix
transition_matrix = np.array([
    # Transition for even state
    [[1, 0],
     [0, 1]],
    # Transition for odd state
    [[0, 1],
     [1, 0]],
])
transition_matrix = np.transpose(transition_matrix, (1, 0, 2))
# Create the DFA instance
ParityDFA = DFA(alphabet, start_state, accept_states, states, transition_matrix)

if __name__ == "__main__":
    # Test the DFA with some input strings
    seq = ParityDFA.process_sequence('0101') # Output: True (even number of 1s)
    print(seq)
    seq = ParityDFA.process_sequence('1011')    # Output: False (odd number of 1s)
    print(seq)