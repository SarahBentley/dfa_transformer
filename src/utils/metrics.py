# Evaluation
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, average_precision_score, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

def macro_accuracy(true_states, predicted_states):
    accuracies = []
    classes = sorted(set(true_states) | set(predicted_states))
    for c in classes:
        class_indices = [i for i, state in enumerate(true_states) if state == c]
        class_true = [true_states[i] for i in class_indices]
        class_pred = [predicted_states[i] for i in class_indices if i < len(predicted_states)]
        if len(class_true) == len(class_pred):
            accuracies.append(accuracy_score(class_true, class_pred))
        else:
            # print(f"Warning: Class {c} has mismatched lengths for true and predicted states")
            continue
    return np.mean(accuracies) if accuracies else None

def micro_accuracy(true_states, predicted_states):
    return accuracy_score(true_states, predicted_states)

def precision_recall_fscore(true_states, predicted_states):
    classes = sorted(set(true_states) | set(predicted_states))  # Get unique class names
    precision, recall, fscore, _ = precision_recall_fscore_support(true_states, predicted_states, labels=classes, average=None)
    
    # Create a dictionary with class names as keys and metrics as values
    metrics = {cls: {'precision': p, 'recall': r, 'fscore': f} 
               for cls, p, r, f in zip(classes, precision, recall, fscore)}
    
    return metrics

def auc_score(true_states, predicted_states):
    classes = sorted(set(true_states) | set(predicted_states))
    true_binary = [[1 if s == c else 0 for c in classes] for s in true_states]
    pred_binary = [[1 if s == c else 0 for c in classes] for s in predicted_states]
    true_binary = np.array(true_binary)
    pred_binary = np.array(pred_binary)

    aucs = []
    for i in range(len(classes)):
        if len(np.unique(true_binary[:, i])) == 2:
            aucs.append(roc_auc_score(true_binary[:, i], pred_binary[:, i]))
        else:
            continue
            # print(f"Skipping AUC calculation for class {classes[i]} as it has only one class present in true labels.")

    return np.mean(aucs) if aucs else None

def mean_average_precision(true_states, predicted_states):
    classes = sorted(set(true_states) | set(predicted_states))
    true_binary = [[1 if s == c else 0 for c in classes] for s in true_states]
    pred_binary = [[1 if s == c else 0 for c in classes] for s in predicted_states]
    true_binary = np.array(true_binary)
    pred_binary = np.array(pred_binary)
    aps = []
    for i in range(len(classes)):
        aps.append(average_precision_score(true_binary[:, i], pred_binary[:, i]))
    return np.mean(aps)

def precision_recall_curve_metric(true_states, predicted_states):
    classes = sorted(set(true_states) | set(predicted_states))
    true_binary = [[1 if s == c else 0 for c in classes] for s in true_states]
    pred_binary = [[1 if s == c else 0 for c in classes] for s in predicted_states]
    true_binary = np.array(true_binary)
    pred_binary = np.array(pred_binary)

    precision_recall_curves = {}
    for i, c in enumerate(classes):
        precision, recall, _ = precision_recall_curve(true_binary[:, i], pred_binary[:, i])
        precision_recall_curves[c] = (precision, recall)
    return precision_recall_curves

def final_eval_metrics(true_states, predicted_states):
    mask = true_states != '<pad>'
    true_states = true_states[mask].flatten()
    predicted_states = predicted_states[mask].flatten()
    results = {}
    results['macro_accuracy'] = macro_accuracy(true_states, predicted_states)
    results['micro_accuracy'] = micro_accuracy(true_states, predicted_states)
    prf = precision_recall_fscore(true_states, predicted_states)
    results['prf'] = prf
    results['auc'] = auc_score(true_states, predicted_states)
    results['mAP'] = mean_average_precision(true_states, predicted_states)
    results['precision_recall_curve'] = precision_recall_curve_metric(true_states, predicted_states)
    return results

def train_eval_metrics(true_states, predicted_states):
    mask = true_states != '<pad>'
    true_states = true_states[mask].flatten()
    predicted_states = predicted_states[mask].flatten()
    results = {}
    results['macro_accuracy'] = macro_accuracy(true_states, predicted_states)
    results['micro_accuracy'] = micro_accuracy(true_states, predicted_states)
    results['auc'] = auc_score(true_states, predicted_states)
    results['mAP'] = mean_average_precision(true_states, predicted_states)
    return results


if __name__ == "__main__":
    tgt = np.array([['<START>', '<even>', '<pad>', '<even>', '<pad>', '<odd>','<pad>', '<odd>', '<pad>', '<odd>', '<pad>', '<even>', '<pad>', '<odd>', '<pad>', '<even>', '<pad>', '<odd>', '<pad>', '<even>', '<pad>', '<ACCEPT>']])

    pred = np.array([['<START>', '<even>', '0', '<even>', '1', '<odd>', '0', '<odd>', '0', '<odd>', '1', '<even>', '1', '<even>', '1', '<even>', '1', '<odd>', '1', '<odd>', '<END>','<REJECT>']])

    metrics = final_eval_metrics(tgt, pred)