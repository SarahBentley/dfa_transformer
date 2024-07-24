import torch
import torch.nn as nn
from ..DFA.DFA import OneHotEncoder

# Given num states and len of alphabet, this model will learn: the transition tensor, the start state, and which states are accepting vs rejecting
class BilinearFunction(nn.Module):
  def __init__(self, n_alphabet, n_states):
    super(BilinearFunction, self).__init__()

    self.n_alphabet = n_alphabet
    self.n_states = n_states

    # Create bilinear layer with x_1 A x_2 where A has shape (#states, #states, #alphabet)
    # x_1 has shape n_states
    # x_2 has shape n_alphabet
    self.bilinear = nn.Bilinear(self.n_states, self.n_alphabet, self.n_states, bias=False)

  def forward(self, last_state, symbol):
    # Compute output of bilinear layer
    logits = self.bilinear(last_state, symbol)
    return logits