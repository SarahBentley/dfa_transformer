import numpy as np
import torch
from ..DFA.DFA import OneHotEncoder

  
class DFADataloader:
    def __init__(self, DFA, seq_len):
        self.DFA = DFA
                
        self.alphabet = self.DFA.alphabet + ['<', '>']
        self.states = self.DFA.states + ['<START>', '<ACCEPT>', '<REJECT>']

        # OneHotEncoder for states and inputs
        self.state_encoder = OneHotEncoder(self.states)
        self.input_encoder = OneHotEncoder(self.alphabet)

        self.seq_len = seq_len

    def generate(self):
        ''' Generates input sequence and returns encoded input sequence and output sequence from the DFA.'''
        # Generate a training example:
        random_array = np.random.choice(self.DFA.alphabet, size=self.seq_len)  # Generate a random array of integers (0 or 1)
        input = np.concatenate((['<'], random_array, ['>']))
        input = self.input_encoder.multi_encode(input)

        output = self.DFA.process_sequence(random_array).split(' ')
        output = self.state_encoder.multi_encode(output)

        return input, output

if __name__ == "__main__":
    # sanity checks
    from ..DFA.ParityDFA import ParityDFA
    dataloader = DFADataloader(ParityDFA, 8)
    input, output = dataloader.generate()
    print(input)
    print(output)
    decoded_input = dataloader.input_encoder.multi_decode(input)
    decoded_output = dataloader.state_encoder.multi_decode(output)
    print(decoded_input)
    print(decoded_output)