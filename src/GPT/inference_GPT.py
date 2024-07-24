import torch
import torch.nn.functional as F
import numpy as np
from .dataloader import DFADataloader
from ..utils.metrics import train_eval_metrics
import plotly.graph_objs as go

def infer(model, dataloader, batch_size, same_lengths=False, temperature=1.0, top_k=None):
    """
    Perform inference using the model, predicted every other token (to avoid predicting input symbols to DFA).
    For example: given <START> predict <EVEN>, then given 1, predict <ODD>. Finally, given <END> predict <ACCEPT> or <REJECT>.
    """
    src_encoded, tgt_encoded = dataloader.generate_batch(batch_size, same_lengths) # generates batch of shape (batch_size, max_seq_len)
    tgt_pred = torch.concat((src_encoded[:, 0].unsqueeze(1), tgt_encoded), dim=1)

    pred = None
    for i in range(0, src_encoded.size(dim=1), 2):
        # model should predict all the odd tokens, so append the even tokens from src
        next_src = src_encoded[:, i].unsqueeze(1)
        if pred==None:
            # take just the first token
            pred = next_src
        else:
            pred = torch.cat((pred, next_src), dim=1)

        # forward the model to get the logits for the index in the sequence
        logits, _ = model(pred)

        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        pred_next = torch.multinomial(probs, num_samples=1)
        # replace indices with prediction
        pred = torch.cat((pred, pred_next), dim=1)

    tgt_decoded = np.array([dataloader.decode(seq) for seq in tgt_pred])
    pred_decoded = np.array([dataloader.decode(seq) for seq in pred])

    return tgt_decoded, pred_decoded

def metrics_by_length(model, DFA, lengths, batch_size, pad_idx=0):
    macro_accuracies = []
    micro_accuracies = []
    aucs = []
    maps = []

    for length in lengths:
        dataloader = DFADataloader(DFA, length, pad_idx)
        # Perform inference
        y_true, y_pred = infer(model, dataloader, batch_size, same_lengths=True)

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

