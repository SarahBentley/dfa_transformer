import torch
import torch.nn.functional as F
import numpy as np
from .dataloader import DFADataloader
from ..utils.metrics import train_eval_metrics, micro_accuracy
import plotly.graph_objs as go
import wandb
from tqdm import tqdm
from .WGT import WGT
from .RWGT import RWGT

def highlight_mismatches(y_true_seq, y_pred_seq):
    highlighted_y_true = ["<p>"]
    highlighted_y_pred = ["<p>"]
    
    for true_elem, pred_elem in zip(y_true_seq, y_pred_seq):
        true_elem = true_elem.replace("<", "").replace(">", "")
        pred_elem = pred_elem.replace("<", "").replace(">", "")
        if true_elem != pred_elem:
            highlighted_y_true.append(f"<span style='color:red;'> {true_elem} </span>")
            highlighted_y_pred.append(f"<span style='color:red;'> {pred_elem} </span>")
        else:
            highlighted_y_true.append(f" {true_elem} ")
            highlighted_y_pred.append(f" {pred_elem} ")
    
    highlighted_y_true.append("</p>")
    highlighted_y_pred.append("</p>")
    
    return ''.join(highlighted_y_true), ''.join(highlighted_y_pred)


def infer(model, dataloader, batch_size, same_lengths=False, temperature=1.0, top_k=None):
    """
    Perform inference using the model, predicted only states and not input symbols, using the state_freq attribute of the dataloader.
    For example: given <START> predict <EVEN>, then given 1, predict <ODD>. Finally, given <END> predict <ACCEPT> or <REJECT>.
    """
    src_encoded, tgt_encoded = dataloader.generate_batch(batch_size, same_lengths) # generates batch of shape (batch_size, max_seq_len)
    tgt_pred = torch.concat((src_encoded[:, 0].unsqueeze(1), tgt_encoded), dim=1)
    pred = src_encoded[:, 0].unsqueeze(1) # add start token

    for i in range(tgt_encoded.size(1)):
        
        next_tokens = tgt_encoded[:, i]
        pad_mask = next_tokens == dataloader.pad_idx
        
        # Create pred_pad and pred_non_pad
        pred_pad = pred[pad_mask]
        pred_non_pad = pred[~pad_mask]

        if pad_mask.any():  # If there are pad tokens in this column
            # Get next source tokens for sequences with pad
            if i+1 >= src_encoded.size(1):
                next_src = next_tokens[pad_mask].unsqueeze(1)
            else:
                next_src = src_encoded[pad_mask][:, i+1].unsqueeze(1)
            # Append the next source token for sequences with pad
            pred_pad = torch.cat((pred_pad, next_src), dim=1)
        
        if (~pad_mask).any():  # If there are non-pad tokens to predict

            # Forward the model to get the logits for the sequences without pad
            if pred_non_pad.size(0) > 0:
                logits, _ = model(pred_non_pad)

                # Pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature

                # Optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)

                # Sample from the distribution
                pred_next = torch.multinomial(probs, num_samples=1)

                # Append predicted next token
                pred_non_pad = torch.cat((pred_non_pad, pred_next), dim=1)

        # Combine pred_pad and pred_non_pad back into pred
        pred_combined = torch.zeros((batch_size, pred.size(1) + 1), dtype=pred.dtype, device=pred.device)
        if pad_mask.any():
            pred_combined[pad_mask] = pred_pad
        if (~pad_mask).any():
            pred_combined[~pad_mask] = pred_non_pad
        pred = pred_combined

    tgt_decoded = np.array([dataloader.decode(seq) for seq in tgt_pred])
    pred_decoded = np.array([dataloader.decode(seq) for seq in pred])

    return tgt_decoded, pred_decoded


def metrics_by_length(model, DFA, lengths, batch_size, pad_idx=0):
    macro_accuracies = []
    micro_accuracies = []
    aucs = []
    maps = []
    seqs = []
    if isinstance(model, WGT) or isinstance(model, RWGT):
        state_freq = model.config.window_size - 1
    else:
        state_freq = 1

    for length in tqdm(lengths, desc="Metrics by length"):

        dataloader = DFADataloader(DFA, length, pad_idx, state_freq=state_freq)
        # Perform inference
        y_true, y_pred = infer(model, dataloader, batch_size, same_lengths=True)
        seqs.append((y_true, y_pred))

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

    # Sort micro accuracies and get the corresponding indices
    sorted_indices = sorted(range(len(micro_accuracies)), key=lambda i: micro_accuracies[i])

    # Extract the sequences corresponding to the top 10% worst micro accuracies
    topx = int(np.ceil(len(lengths)*.1))
    worst_seqs = [seqs[i] for i in sorted_indices[:topx]]
    best_seqs =  [seqs[i] for i in sorted_indices[-topx:]]

    # Create wandb tables for worst and best sequences
    table_worst = wandb.Table(columns=["Accuracy", "y_true", "y_pred"])
    table_best = wandb.Table(columns=["Accuracy", "y_true", "y_pred"])

    # Log worst sequences
    for y_true, y_pred in worst_seqs:
        mask = (y_true != '<pad>')
        y_true_masked = [true_seq[mask[i]] for i, true_seq in enumerate(y_true)]
        y_pred_masked = [pred_seq[mask[i]] for i, pred_seq in enumerate(y_pred)]

        accuracies = np.array([micro_accuracy(y_true_masked[i], y_pred_masked[i]) for i in range(len(y_true))])
        ind = np.argmin(accuracies)

        highlighted_y_true, highlighted_y_pred = highlight_mismatches(y_true_masked[ind], y_pred_masked[ind])
        table_worst.add_data(accuracies[ind], wandb.Html(highlighted_y_true), wandb.Html(highlighted_y_pred))

    # Log best sequences
    for y_true, y_pred in best_seqs:
        mask = (y_true != '<pad>')
        y_true_masked = [true_seq[mask[i]] for i, true_seq in enumerate(y_true)]
        y_pred_masked = [pred_seq[mask[i]] for i, pred_seq in enumerate(y_pred)]

        accuracies = np.array([micro_accuracy(y_true_masked[i], y_pred_masked[i]) for i in range(len(y_true))])
        ind = np.argmax(accuracies)

        highlighted_y_true, highlighted_y_pred = highlight_mismatches(y_true_masked[ind], y_pred_masked[ind])
        table_best.add_data(accuracies[ind], wandb.Html(highlighted_y_true), wandb.Html(highlighted_y_pred))


    return {'Performance metrics by length': fig, "Top 10% Best Accuracy Sequences": table_best, "Top 10% Worst Accuracy Sequences": table_worst}

