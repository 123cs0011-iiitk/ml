# Source: course assignment — Author: Ankit Kumar
# Written by Ankit — course implementation — do not modify

import numpy as np
import random
import operator
import math

def euclidean(x, y):
    """Calculate Euclidean distance between two points"""
    dist = 0.0
    for i in range(len(x) - 1):
        dist += (x[i] - y[i]) ** 2
    return math.sqrt(dist)


def get_neighbors(train, test, k):
    """Find k nearest neighbors for a test point"""
    # here test will be a row
    dist = []
    for i in train:
        dist.append((i[-1], euclidean(i, test)))  # i[-1] is label
    dist.sort(key=operator.itemgetter(1))
    neighbors = [dist[i] for i in range(k)]
    return neighbors


def get_votes(neighbors):
    """Get majority vote from neighbors"""
    class_votes = {}
    for x in range(len(neighbors)):
        res = neighbors[x][0]
        if res in class_votes:
            class_votes[res] += 1
        else:
            class_votes[res] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def get_accuracy(test_fold, pred):
    """Calculate accuracy of predictions"""
    correct = 0
    for i in range(len(test_fold)):
        if test_fold[i][-1] == pred[i]:  # here also full test_fold is there
            correct += 1
    return (correct / len(test_fold)) * 100


def knn_classifier(train_data, test_data, k=5):
    """
    K-Nearest Neighbors classifier implementation from scratch
    
    Args:
        train_data: Training data with features and labels
        test_data: Test data with features and labels
        k: Number of neighbors to consider
    
    Returns:
        predictions: List of predicted labels
        accuracy: Accuracy percentage
    """
    data = np.array(train_data, dtype='object')
    np.random.seed(42)
    np.random.shuffle(data)
    
    s = int(0.8 * len(data))
    train = data[:s]
    test = data[s:]
    folds = 5
    fold_size = len(train) // folds

    K = [1, 3, 5, 7, 9]
    k_acc = {}
    
    for k_val in K:
        fold_acc = []
        for i in range(folds):
            start_ind = i * fold_size
            end_ind = (i + 1) * fold_size
            test_fold = train[start_ind:end_ind]
            train_fold = np.concatenate((train[:start_ind], train[end_ind:]), axis=0)
            pred = []
            for x in range(len(test_fold)):
                neighbors = get_neighbors(train_fold, test_fold[x], k_val)
                result = get_votes(neighbors)
                pred.append(result)
            accuracy = get_accuracy(test_fold, pred)
            fold_acc.append(accuracy)
        mean_accuracy = np.mean(np.array(fold_acc), axis=0)  # here axis=0 keep in mind
        k_acc[k_val] = mean_accuracy
    
    k_acc = sorted(k_acc.items(), key=operator.itemgetter(1), reverse=True)
    k_best = k_acc[0][0]
    
    # Test on actual test set
    pred = [get_votes(get_neighbors(train, row, k_best)) for row in test]
    test_acc = get_accuracy(test, pred)
    
    return pred, test_acc, k_best


if __name__ == "__main__":
    # Example usage
    n = int(input())
    data = []
    for i in range(n):
        a, b, c, d, e = input().split()
        data.append([float(a), float(b), float(c), float(d), e])
    
    predictions, accuracy, best_k = knn_classifier(data, data)
    print(f"Best k: {best_k} | Test accuracy: {accuracy:.2f}")
