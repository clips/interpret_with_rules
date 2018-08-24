from sklearn.metrics import f1_score

def get_f1_score(y_true, y_pred, avg = 'macro'):
    """
    Returns the F1-score of the predictions
    :param y_true: List of gold labels
    :param y_pred: List of predicted labels
    :param avg: (macro/micro/weighted/None) Averaging of F-score esp for multiclass setups.
    :return: F1 score
    """
    return f1_score(y_true = y_true, y_pred = y_pred, average=avg)