import numpy as np
import torch

def interweave_sequences(states, sequence):
    ''' '''
    interwoven = [states[0]]
    sequence_index = 0

    # Loop through states and interweave with sequence
    for i in range(1, len(states) - 1):
        interwoven.append(states[i])
        if sequence_index < len(sequence):
          interwoven.append(sequence[sequence_index])
        elif sequence_index == len(sequence):
          interwoven.append('<END>')
        
        sequence_index += 1
    
    # Append the last state
    interwoven.append(states[-1])

    return interwoven
  
class DFADataloader:
    def __init__(self, DFA, max_seq_len, pad_idx=0):
        self.DFA = DFA
        self.pad_idx = pad_idx

        self.vocab = self.DFA.states + ['<START>', '<END>', '<ACCEPT>', '<REJECT>'] + self.DFA.alphabet
        self.vocab.insert(self.pad_idx, '<pad>')

        self.vocab_size = len(self.vocab)
        self.max_seq_len = max_seq_len

    def encode(self, sequence, max_len):
        return [self.vocab.index(i) for i in sequence] + [self.pad_idx for _ in range(max_len - len(sequence)-1)]
    
    def decode(self, sequence):
        return [self.vocab[i] for i in sequence]

    def generate_batch(self, batch_size, same_lengths=False):
        src = []
        tgt = []

        for _ in range(batch_size):
            if same_lengths:
                symbols = np.random.choice(self.DFA.alphabet, size=self.max_seq_len)
            else:
                symbols = np.random.choice(self.DFA.alphabet, size=np.random.randint(low=4, high=self.max_seq_len+1))
            states = self.DFA.process_sequence(symbols).split(' ')

            # create src
            seq = interweave_sequences(states, symbols)
            src.append(seq)

            # create tgt with padding in position of input symbols
            tgt_seq = []
            for i in seq:
                if i in self.DFA.alphabet or i == '<START>' or i == '<END>':
                    tgt_seq.append('<pad>')
                else:
                    tgt_seq.append(i)

            tgt.append(tgt_seq)

        # Encode input to DFA and output from DFA. Shift right.
        max_len = max([len(i) for i in src])
        src_encoded = torch.tensor([self.encode(seq[:-1], max_len) for seq in src])  # Shape: (batch_size, max_seq_len)
        tgt_encoded = torch.tensor([self.encode(seq[1:], max_len) for seq in tgt])

        return src_encoded, tgt_encoded

if __name__ == "__main__":
    # sanity checks
    from ..DFA.ParityDFA import ParityDFA
    dataloader = DFADataloader(ParityDFA, max_seq_len=6, pad_idx=0)
    encoded_src, encoded_tgt = dataloader.generate_batch(batch_size=10, same_lengths=False)
    print("----------RESULTS----------------------")
    print(encoded_src)
    print(encoded_tgt)
    decoded_src = [dataloader.decode(seq) for seq in encoded_src]
    decoded_tgt = [dataloader.decode(seq) for seq in encoded_tgt]
    print(decoded_src)
    print(decoded_tgt)