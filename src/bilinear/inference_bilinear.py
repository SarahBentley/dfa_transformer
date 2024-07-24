import numpy as np
import torch
from torch.nn import functional as F
from .dataloader import DFADataloader
from ..utils.metrics import train_eval_metrics
import plotly.graph_objs as go

# Inference
def infer(model, dataloader, num_seq=50):
  ''' Returns decoded np array from inference on num_seq sequences'''
  dfa_outputs = []
  learner_outputs = []

  for i in range(num_seq):
    # Generate a training example:
    input, output = dataloader.generate()
    decoded_output = dataloader.state_encoder.multi_decode(output)
    dfa_outputs.append(decoded_output)

    learner_output = ['<START>']

    for i in range(len(input)):
      # Predict next 'token' for each character in training example
      symbol = input[i]
      last_state = dataloader.state_encoder.single_encode(learner_output[i])


      logits = model(last_state, symbol)
      probabilities = F.softmax(logits, dim=0)
      # Predict the next token (example: using argmax)
      predicted_state_idx = torch.argmax(probabilities).item()
      predicted_state= dataloader.states[predicted_state_idx]
      learner_output.append(predicted_state)

    learner_outputs.append(learner_output)

  return np.array(dfa_outputs), np.array(learner_outputs)

def metrics_by_length(model, DFA, lengths, num_seq=50):
    macro_accuracies = []
    micro_accuracies = []
    aucs = []
    maps = []

    for length in lengths:
        dataloader = DFADataloader(DFA,  length)
        # Perform inference
        y_true, y_pred = infer(model, dataloader, num_seq)

        # Compute metrics
        metrics = train_eval_metrics(y_true, y_pred)
        
        # Append metrics to lists
        macro_accuracies.append(metrics['macro_accuracy'])
        micro_accuracies.append(metrics['micro_accuracy'])
        aucs.append(metrics['auc'])
        maps.append(metrics['mAP'])

    # Create Plotly figures
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lengths, y=macro_accuracies, mode='lines+markers', name='Macro Accuracy'))

    fig.add_trace(go.Scatter(x=lengths, y=micro_accuracies, mode='lines+markers', name='Micro Accuracy'))

    fig.add_trace(go.Scatter(x=lengths, y=aucs, mode='lines+markers', name='AUC'))

    fig.add_trace(go.Scatter(x=lengths, y=maps, mode='lines+markers', name='mAP'))

    # Customize layout
    fig.update_layout(
        xaxis_title='Input Length',
        yaxis_title='Performance Metric',
        title=f'Performance Metrics vs. Input Length',
        template='plotly_dark',
        legend=dict(x=0, y=1, traceorder='normal')
    )
    return {'Performance metrics by length': fig}