import matplotlib.pyplot as plt
from .metrics import *
import plotly.graph_objs as go

def plot_precision_recall_fscore(prf):
    # Extract class names and metrics
    classes = list(prf.keys())
    precision = [prf[c]['precision'] for c in classes]
    recall = [prf[c]['recall'] for c in classes]
    fscore = [prf[c]['fscore'] for c in classes]

    # Create the figure
    precision_recall_fscore_plot = go.Figure()

    # Add Precision trace
    precision_recall_fscore_plot.add_trace(go.Bar(
        x=classes, 
        y=precision, 
        name='Precision', 
        marker_color='blue'
    ))

    # Add Recall trace
    precision_recall_fscore_plot.add_trace(go.Bar(
        x=classes, 
        y=recall, 
        name='Recall', 
        marker_color='green'
    ))

    # Add F-score trace
    precision_recall_fscore_plot.add_trace(go.Bar(
        x=classes, 
        y=fscore, 
        name='F-score', 
        marker_color='orange'
    ))

    # Update layout
    precision_recall_fscore_plot.update_layout(
        title='Precision, Recall, and F-score by Class',
        xaxis_title='Classes',
        yaxis_title='Scores',
        barmode='group'  # Group bars together for each class
    )

    return precision_recall_fscore_plot

def plot_precision_recall_curve(precision_recall_curve):
     # Precision-Recall Curve using Plotly
    precision_recall_curve_plot = go.Figure()
    for c, (precision, recall) in precision_recall_curve.items():
        precision_recall_curve_plot.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'Class {c}'))

    precision_recall_curve_plot.update_layout(title='Precision-Recall Curve (after training)',
                                             xaxis_title='Recall', yaxis_title='Precision')
    
    return precision_recall_curve_plot

def visualize_final_metrics(true_states, predicted_states):
    # Evaluate metrics
    metrics = final_eval_metrics(true_states, predicted_states)

    # Initialize dictionary to store plots
    plots_dict = {}

    # Log Plotly plot to wandb
    plots_dict['precision_recall_fscore'] = plot_precision_recall_fscore(metrics['prf'])

    # Log Plotly plot to wandb
    plots_dict['precision_recall_curve'] = plot_precision_recall_curve(metrics['precision_recall_curve'])

    return plots_dict


if __name__ == "__main__":
    tgt = np.array([['<START>', '<even>', '<pad>', '<even>', '<pad>', '<odd>','<pad>', '<odd>', '<pad>', '<odd>', '<pad>', '<even>', '<pad>', '<odd>', '<pad>', '<even>', '<pad>', '<odd>', '<pad>', '<even>', '<pad>', '<ACCEPT>']])

    pred = np.array([['<START>', '<even>', '0', '<even>', '1', '<odd>', '0', '<odd>', '0', '<odd>', '1', '<even>', '1', '<even>', '1', '<even>', '1', '<odd>', '1', '<odd>', '<END>',
                    '<REJECT>']])

    print(visualize_final_metrics(tgt, pred))