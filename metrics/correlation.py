import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


def plcc(x, y):
    """
    Pearson linear correlation coefficient (PLCC).

    :param x:
        Vector 1, a list or an array, with n values.
    :param y:
        Vector 2, a list or an array, with n values.
    :return:
        The PLCC of 2 input vectors (x and y).
    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    assert len(x.shape) == 1 and len(y.shape) == 1, "Please input a vector with only one dimension."
    assert x.shape == y.shape, "The lengths of 2 input vectors are not equal."

    x = x - np.average(x)
    y = y - np.average(y)
    numerator = np.dot(x, y)
    denominator = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))
    ours = numerator / denominator
    scis = pearsonr(x, y)[0]  # Use scipy to calculate PLCC again.

    if abs(ours - scis) > 1e-8:
        print(f"Warning: Our PLCC = {ours:.15f}, scipy PLCC = {scis:.15f}, please check the results!")

    return scis


def fitted_plcc(pred, mos):
    """
    Calculate PLCC with a nonlinear regression using third-order poly fitting.

    It follows the setting of PIPAL:
        Gu et al. PIPAL: A large-scale image quality assessment dataset for perceptual image restoration. In ECCV 2020.
        https://www.jasongt.com/projectpages/pipal.html.
        NTIRE 2021 Perceptual Image Quality Assessment Challenge.
    """
    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(mos, list):
        mos = np.array(mos)

    f = np.polyfit(pred, mos, deg=3)
    fitted_mos = np.polyval(f, pred)

    return plcc(fitted_mos, mos)


def srcc(x, y):
    """
    Spearman rank order correlation coefficient (SRCC).

    :param x:
        Vector 1, a list or an array, with n values.
    :param y:
        Vector 2, a list or an array, with n values.
    :return:
        The SRCC of 2 input vectors (x and y).
    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    assert len(x.shape) == 1 and len(y.shape) == 1, "Please input a vector with only one dimension."
    assert x.shape == y.shape, "The lengths of 2 input vectors are not equal."

    rank_x = x.argsort().argsort()
    rank_y = y.argsort().argsort()

    ours = plcc(rank_x, rank_y)
    scis = spearmanr(x, y)[0]  # Use scipy to calculate PLCC again.

    if abs(ours - scis) > 1e-5:
        print(f"Warning: Our SRCC = {ours:.15f}, scipy SRCC = {scis:.15f}, please check the results!")

    return scis


def krcc(x, y):
    """
    Kendall rank order correlation coefficient (KRCC).

    :param x:
        Vector 1, a list or an array, with n values.
    :param y:
        Vector 2, a list or an array, with n values.
    :return:
        The KRCC of 2 input vectors (x and y).
    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    assert len(x.shape) == 1 and len(y.shape) == 1, "Please input a vector with only one dimension."
    assert x.shape == y.shape, "The lengths of 2 input vectors are not equal."

    return kendalltau(x, y)[0]


if __name__ == "__main__":
    pass
