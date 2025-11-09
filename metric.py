import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score

def calculate_tp_fp_fn_tn(y_true, y_pred):
    """Calculate TP, FP, FN, TN for each class"""

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    return tp, fp, fn, tn

def calculate_accuracy(tp, fp, fn, tn):
    """Calculate accuracy for each class"""
    if tp + fp + fn + tn == 0:
        return 0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return accuracy

def calculate_precision(y_true, y_pred):
    """Calculate precision for each class"""
    if np.sum(y_pred) == 0:
        return 0
    precision = precision_score(y_true, y_pred, average='binary')
    return precision

def calculate_recall(y_true, y_pred):
    """Calculate recall for each class"""
    if np.sum(y_true) == 0:
        return 0
    recall = recall_score(y_true, y_pred, average='binary')
    return recall

def calculate_f1(y_true, y_pred):
    """Calculate F1-score for each class"""
    if np.sum(y_true) == 0 or np.sum(y_pred) == 0:
        return 0
    if (recall_score(y_true, y_pred, average='binary') + precision_score(y_true, y_pred, average='binary')) == 0:
        return 0
    f1 = f1_score(y_true, y_pred, average='binary')
    return f1

def calculate_auc(y_true, y_pred):
    """Calculate AUC for each class"""
    try:
        auc = roc_auc_score(y_true, y_pred, average=None)
    except ValueError:
        return 0
    return auc

def hamming_accuracy(y_true, y_pred):
    """Hamming Accuracy"""
    return np.mean(np.equal(y_true, y_pred))

def example_f1(y_true, y_pred):
    """Example F1 score (averaged across samples)"""
    return f1_score(y_true, y_pred, average='samples')

def micro_f1(y_true, y_pred):
    """Micro F1 score (aggregated over all classes)"""
    return f1_score(y_true, y_pred, average='micro')

def macro_f1(y_true, y_pred):
    """Macro F1 score (averaged across classes)"""
    return f1_score(y_true, y_pred, average='macro')

def calculate_metrics(y_true, y_pred):
    """Calculate HA, example-F1, micro-F1, and macro-F1"""
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    ha = hamming_accuracy(y_true, y_pred)
    ex_f1 = example_f1(y_true, y_pred)
    micro_f1_score = micro_f1(y_true, y_pred)
    macro_f1_score = macro_f1(y_true, y_pred)
    
    return ha, ex_f1, micro_f1_score, macro_f1_score