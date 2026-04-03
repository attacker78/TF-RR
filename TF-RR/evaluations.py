"""
All evaluation metric functions, including respiratory rate and respiratory signal
Respiratory rate: MAE MSE RMSE LOA
Respiratory signal: MAE PCC Cross-Correlation
MSE: from sklearn.metrics import mean_squared_error
"""
import numpy as np
from scipy.stats import stats
from math import sqrt
from sklearn.metrics import mean_squared_error

## MAE
def calculate_mae(predictions, true_values):
    if len(predictions) != len(true_values):
        raise ValueError("The lengths of the prediction list and true value list are not equal")

    n = len(predictions)
    absolute_diff = [abs(pred - true) for pred, true in zip(predictions, true_values)]
    mae = sum(absolute_diff) / n
    return mae

## RMSE
def calculate_rmse(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("The lengths of the two input lists are not equal")

    rmse = sqrt(mean_squared_error(list1, list2))

    return rmse

## LOA
def calculate_loa(predictions, true_values):
    if len(predictions) != len(true_values):
        raise ValueError("The lengths of the prediction list and true value list are not equal")

    n = len(predictions)
    diff = np.array(predictions) - np.array(true_values)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    loa = 1.96 * std_diff
    return loa

## PCC
def calculate_pearson_correlation(predictions, true_values):
    if len(predictions) != len(true_values):
        raise ValueError("The lengths of the two input lists are not equal")

    correlation, _ = stats.pearsonr(predictions, true_values)

    return correlation

## Cross-Correlation
def calculate_cross_correlation(predictions, true_values):
    if len(predictions) != len(true_values):
        raise ValueError("The lengths of the two input lists are not equal")

    # Calculate Cross-Correlation coefficient
    cross_corr = np.correlate(predictions, true_values, mode='full') / (np.std(predictions) * np.std(true_values) * len(predictions))
    return cross_corr

def cross_correlation(predictions, true_values):
    if len(predictions) != len(true_values):
        raise ValueError("Arrays must have the same length")

    # Calculate means
    mean_arr1 = sum(predictions) / len(predictions)
    mean_arr2 = sum(true_values) / len(true_values)

    # Calculate numerator of cross-correlation
    numerator = sum((x - mean_arr1) * (y - mean_arr2) for x, y in zip(predictions, true_values))

    # Calculate standard deviation for each array
    std_arr1 = (sum((x - mean_arr1) ** 2 for x in predictions) / len(predictions)) ** 0.5
    std_arr2 = (sum((y - mean_arr2) ** 2 for y in true_values) / len(true_values)) ** 0.5

    # Calculate denominator of cross-correlation
    denominator = std_arr1 * std_arr2

    # Avoid division by zero
    if denominator == 0:
        return 0

    # Calculate cross-correlation
    cross_corr = numerator / denominator

    return cross_corr


## Euclidean Distance
def euclidean_distance(predictions, true_values):
    if len(predictions) != len(true_values):
        raise ValueError("Arrays must have the same length")

    dist = np.sqrt(np.sum(np.square(np.array(predictions) - np.array(true_values))))

    return dist


## Cosine Similarity
def cosine_similarity(predictions, true_values):
    if len(predictions) != len(true_values):
        raise ValueError("Arrays must have the same length")

    cos_sim = np.dot(predictions, true_values) / (np.linalg.norm(predictions) * np.linalg.norm(true_values))

    return cos_sim

## Dynamic Time Warping (DTW)

def DTW(s1, s2):
    l1 = len(s1)
    l2 = len(s2)
    paths = np.full((l1 + 1, l2 + 1), np.inf)
    paths[0, 0] = 0
    for i in range(l1):
        for j in range(l2):
            d = s1[i] - s2[j]
            cost = d ** 2
            paths[i + 1, j + 1] = cost + min(paths[i, j + 1], paths[i + 1, j], paths[i, j])

    paths = np.sqrt(paths)
    s = paths[l1, l2]
    return s

from numpy import array, zeros, argmin, inf, equal, ndim
# from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import manhattan_distances


#def dynamic_time_warping(predictions, true_values):


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    data1 = np.array([25, 30, 35, 40, 45, 50, 55, 60])
    data2 = np.array([24, 32, 37, 42, 47, 52, 58, 61])

    diff = data1 - data2
    mean = np.mean(diff)
    loa = calculate_loa(data1, data2)

    dtw = DTW(data1, data2)
    cosine = cosine_similarity(data1, data2)

    pcc = calculate_pearson_correlation(data1, data2)
    pcc1, _ = stats.pearsonr(data1, data2)

    print(loa)

    plt.figure(figsize=(8, 6))
    plt.scatter(np.mean([data1, data2], axis=0), diff)

    plt.axhline(mean, color='black', linestyle='--')
    plt.axhline(mean + loa, color='red', linestyle='--')
    plt.axhline(mean - loa, color='red', linestyle='--')

    plt.show()