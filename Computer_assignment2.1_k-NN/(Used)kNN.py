"""
File: kNN.py
Author: Chayadon Lappabud
Date: September 15, 2023
Description: Building a k-Nearest-Neighbors and testing on iris dataset.
Version: 1.0
"""
# Import libraries
import numpy as np
import random as rd # Only used for shuffling dataset

# Create a function for reading the dataset.
def read_data(file_path:str)->list:
    data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        lines = lines[1:] # Skip the header line
        for line in lines:
            values = line.strip().split()
            data.append([float(val) for val in values])
    return data

# Function calculate Distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Function for sorting distances.
def argSort(distance:list)->list:
    sort_index = [i for i, x in sorted(enumerate(distance), key=lambda x: x[1])]
    return sort_index

# Function to find the most frequent element in a list.
def most_frequent(List):
    return max(set(List), key = List.count)

# k-Nearest Neighbors algorithm
def k_nearest_neighbors(X_train, y_train, x_query, k):
    distances = [euclidean_distance(x_query, x) for x in X_train]
    sorted_indices = argSort(distances)
    k_indices = sorted_indices[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = most_frequent(k_nearest_labels)
    return most_common

# Cross Validation
def cross_validation(X, y, k_folds, k):
    fold_size = len(X) // k_folds
    accuracy_scores = []
    
    for i in range(k_folds):
        start, end = i * fold_size, (i + 1) * fold_size
        X_val_fold = X[start:end]
        y_val_fold = y[start:end]
        X_train_fold = np.concatenate((X[:start], X[end:]), axis=0)
        y_train_fold = np.concatenate((y[:start], y[end:]), axis=0)

        correct_predictions = 0
        for j in range(len(X_val_fold)):
            predicted_label = k_nearest_neighbors(X_train_fold, y_train_fold, X_val_fold[j], k)
            if predicted_label == y_val_fold[j]:
                correct_predictions += 1

        accuracy = correct_predictions / len(X_val_fold)
        accuracy_scores.append(accuracy)

    avg_accuracy = np.mean(accuracy_scores)
    return avg_accuracy

# Split dataset into  training set and test set
def hold_out(train, test, num_train = 80):
    length = len(train)
    d = length*num_train // 100
    X_train = train[:d]
    X_test = test[:d]

    y_train = train[d:]
    y_test = test[d:]

    return X_train, X_test, y_train, y_test

# Shuffle the dataset
def fisher_yates_shuffle(arr):
    rd.shuffle(arr)

# Define a function train_test_split to split the data and labels.
def train_test_split(dataset:list)->list:
    data = [d[:-1] for d in dataset]
    labels = [int(label[-1]) for label in dataset]
    return data, labels


#-----------------------------MAIN CODE----------------------------------------------#
# Read the dataset from the specified file path
dataset = read_data('F:/Advanced_Pattern_Recognition/Computer_assignment2.1/dataset/iris.pat')

# Set a seed for random shuffling
rd.seed(1337)

# Shuffle the dataset using Fisher-Yates shuffle
fisher_yates_shuffle(dataset)

# Define the number of folds for cross-validation
k_folds = 10

# Split the dataset into data and labels
X, y = train_test_split(dataset)
X = np.array(X)
y = np.array(y)

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = hold_out(X, y)

# Perform k-fold cross-validation for different values of k
for k in range(1, 20, 1):
    avg_accuracy = cross_validation(np.array(X_train), np.array(X_test), k_folds, k)
    print(f"k = {k}, Average Accuracy: {avg_accuracy:.4f}")
