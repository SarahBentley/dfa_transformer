import numpy as np
import torch

class OneHotEncoder:
  ''' A one hot encoder that outputs encoded elements as tensors '''
  def __init__(self, list_of_elts):
    self.decoded = np.array(list_of_elts, dtype=object)
    transformed = np.zeros((self.decoded.shape[0], self.decoded.shape[0]), dtype=np.float32)

    for i in range(self.decoded.shape[0]):
      transformed[i][i] = 1

    self.encoded = transformed

  def single_encode(self, elt):
    idx = np.where(self.decoded == elt)
    res = np.squeeze(self.encoded[idx])
    return torch.from_numpy(res)

  def multi_encode(self, elts):
    output = [self.single_encode(elt) for elt in elts]
    res = np.array(output)
    return torch.from_numpy(res)

  def single_decode(self, elt):
    elt = elt.numpy()
    idx = np.where(np.all(self.encoded == elt, axis=1))
    res = np.squeeze(self.decoded[idx])
    return str(res)

  def multi_decode(self, elts):
    output = [self.single_decode(elt) for elt in elts]
    res = np.array(output)
    return res
  
class OneHotEncoderNP:
  ''' A one hot encoder that outputs encoded elements as Numpy Arrays '''
  def __init__(self, list_of_elts):
    self.decoded = np.array(list_of_elts, dtype=object)
    transformed = np.zeros((self.decoded.shape[0], self.decoded.shape[0]))

    for i in range(self.decoded.shape[0]):
      transformed[i][i] = 1

    self.encoded = transformed

  def single_encode(self, elt):
    idx = np.where(self.decoded == elt)
    res = np.squeeze(self.encoded[idx])
    return res

  def multi_encode(self, elts):
    output = [self.single_encode(elt) for elt in elts]
    res = np.array(output)
    return res

  def single_decode(self, elt):
    idx = np.where(np.all(self.encoded == elt, axis=1))
    res = np.squeeze(self.decoded[idx])
    return str(res)

  def multi_decode(self, elts):
    output = [self.single_decode(elt) for elt in elts]
    res = np.array(output)
    return res
  
class DFA:
    def __init__(self, alphabet, start_state, accept_states, states, transition_matrix):
        ''' Assume states and alphabet always consists of strings '''
        self.alphabet = alphabet
        self.start_state = start_state
        self.accept_states = accept_states
        self.states = states
        self.transition_matrix = transition_matrix

        # OneHotEncoder for states and inputs
        self.state_encoder = OneHotEncoderNP(self.states)
        self.input_encoder = OneHotEncoderNP(self.alphabet)

        self.start_state_vec = self.state_encoder.single_encode(self.start_state)
        self.accept_states_vec = self.state_encoder.multi_encode(self.accept_states)

        self.current_state_vec = self.start_state_vec

    def reset(self):
        """Resets the DFA to the start state."""
        self.current_state_vec = self.start_state_vec

    def transition(self, symbol):
        """Performs a transition based on the input symbol using matrix multiplication."""
        # Decode current State
        current_state = self.state_encoder.single_decode(self.current_state_vec)

        # Encode input symbol
        input_vec = self.input_encoder.single_encode(symbol)
        # Perform matrix multiplication for transition
        next_state_vec = self.current_state_vec @ self.transition_matrix @ input_vec.T
        next_state_vec = next_state_vec.T

        # Decode next state
        next_state = self.state_encoder.single_decode(next_state_vec)

        # print(f"Transition: ({current_state}, {symbol}) -> {next_state}")

        # Update current state
        self.current_state_vec = next_state_vec

    def is_accepting(self):
        """Checks if the current state is an accepting state."""
        return any(np.all(self.current_state_vec == state_vec) for state_vec in self.accept_states_vec)

    def process_sequence(self, input_string):
        """Processes a sequence (string or list) of symbols through the DFA."""
        seq = '<START>' + ' ' + self.start_state
        self.reset()
        for symbol in input_string:
          self.transition(symbol)
          state = self.state_encoder.single_decode(self.current_state_vec)
          # seq += ' ' + str(symbol)
          seq += ' ' + state

        if self.is_accepting():
          seq += " <ACCEPT>"
        else:
          seq += " <REJECT>"

        return seq