import numpy as np


def accuracy(pred, gt):
    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(gt, list):
        gt = np.array(gt)

    assert len(pred.shape) == 1 and len(gt.shape) == 1, "Please input a vector with only one dimension."
    assert pred.shape == gt.shape, "The lengths of 2 input vectors are not equal."

    return np.sum(pred == gt) / gt.shape[0]


def get_tp_fp_tn_fn(pred, gt):
    """
    Calculate True Positive, False Positive, True Negative and False Negative.

    Ref:
        https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)

    :param pred:
        The predictive results of binary classifier.
    :param gt:
        The ground truth of of binary classification.
    :return:
        The number of True Positive, False Positive, True Negative and False Negative.
    """
    if isinstance(pred, np.ndarray):
        pred = pred.tolist()
    if isinstance(pred, np.ndarray):
        gt = gt.tolist()

    assert len(pred) == len(gt), "The lengths of 2 input vectors are not equal."

    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1:
            tp += 1
        elif gt[i] == 0 and pred[i] == 0:
            tn += 1
        elif gt[i] == 1 and pred[i] == 0:
            fp += 1
        elif gt[i] == 0 and pred[i] == 1:
            fn += 1
        else:
            raise ValueError(f"Illegal gt[{i}] = {gt[i]} or pred[{i}] = {pred[i]}.")

    return tp, fp, tn, fn


def precision(pred, gt):
    """
    Calculate precision.

    :param pred:
        The predictive results of binary classifier.
    :param gt:
        The ground truth of of binary classification.
    """
    tp, fp, tn, fn = get_tp_fp_tn_fn(pred, gt)
    return tp / (tp + fp)


def recall(pred, gt):
    """
    Calculate recall.

    :param pred:
        The predictive results of binary classifier.
    :param gt:
        The ground truth of of binary classification.
    """
    tp, fp, tn, fn = get_tp_fp_tn_fn(pred, gt)
    return tp / (tp + fn)


if __name__ == '__main__':
    pass
