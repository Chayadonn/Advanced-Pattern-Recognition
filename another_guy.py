######################################################################
# Program: Minimum Risk Decision Theoretic Classifier
# Author: Chayadon Lappabud
# Date: July 26, 2023
# Description: So hard. T T
######################################################################
import math  as m
import another_guy as ag

transpost = lambda x: list(zip(*x))
print_matrix = lambda x: list(map(print, x))

# Create function reading dataset.
def read_data(file_path):
    data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        lines = lines[1:]
        for line in lines:
            values = line.strip().split()
            data.append([float(val) for val in values])
    return data


# Calculate mean vector.
def cal_mean_vector(data:list)->list:
    num_of_features = len(data[0])
    num_of_samples = len(data)
    mean_vector = []
    for i in range(num_of_features):
        sum = 0
        for j in range(num_of_samples):
            sum += data[j][i]
        mean_vector.append(round(sum / num_of_samples, 4))
        
    return mean_vector


# Define function cal_covariance_matrix.
def cal_covariance_matrix(mean_vec:list, data_class:list)->list:
    # print("-----Calculate Covarince Matrix-----")
    num_of_sample = len(data_class)
    cov = [[0 for _ in data_class[0]] for _ in data_class]
    i = 0
    for sample in data_class:
        j = 0
        for cols in sample:
            cov[i][j] = round(cols - mean_vec[j], 4)
            j += 1
        i += 1
    inv_cov = transpost(cov)
    covariance = [[round(sum(a * b for a, b in zip(inv_cova, cova))/num_of_sample,4)
                                    for cova in zip(*cov)] 
                                        for inv_cova in inv_cov]
    return covariance


# Define function train_test_split to split the data and labels.
def train_test_split(dataset:list)->list:
    data = [d[:-1] for d in dataset]
    labels = [int(label[-1]) for label in dataset]
    return data, labels


# Define function split_by_class.
def split_by_class(dataset:list, labels:list)->list:
    classes = list(set(labels))
    for c in range(len(classes)):
        yield [dataset[i] for i in range(len(dataset)) if labels[i] == classes[c]]


def multivariate_normal_pdf(x, mean, covariance_matrix):
    num_features = len(x)
    print(mean)
    determinant = 1.0
    inverse_cov_matrix = []

    for i in range(num_features):
        determinant *= covariance_matrix[i]
        
        inverse_row = [-c / covariance_matrix[i] for c in covariance_matrix]
        inverse_row[i] = -inverse_row[i]
        inverse_cov_matrix.append(inverse_row)
    prefactor = 1.0 / (m.pow((2 * m.pi), num_features / 2) * m.sqrt(determinant))
    exponent = 0.0
    for i in range(num_features):
        for j in range(num_features):
            exponent += (x[i] - mean) * inverse_cov_matrix[i][j] * (x[j] - mean)
    return prefactor * m.exp(-0.5 * exponent)


def predict(data, means, cov_matrices):
    predicted_labels = []
    for x in data:
        max_posterior_prob = -1
        predicted_label = None
        for class_label, mean, cov_matrix in zip([1, 2], means, cov_matrices):
            posterior_prob = multivariate_normal_pdf(x, mean, cov_matrix)
            if posterior_prob > max_posterior_prob:
                max_posterior_prob = posterior_prob
                predicted_label = class_label
        predicted_labels.append(predicted_label)
    return predicted_labels



# Define this function to train model.
def model_fit(i:int, test_data:list, test_label:list, *dataclass:list):
    vector_mean = cov_mat =  []
    print(f"------------------------------ folds {i+1} ------------------------------")
    for j in range(2):
        print(f"Class {j+1}: - Mean Vector")
        vector_mean = cal_mean_vector(dataclass[j])
        print(vector_mean)

        print(f"Class {j+1}: - Covariance Matrix")
        cov_mat = cal_covariance_matrix(vector_mean, dataclass[j])
        print_matrix(cov_mat)
        print()

    predicted = predict(test_data, vector_mean, cov_mat)
    print(predicted)
    print(test_label)
    correct = 0
    for true_label, predicted_label in zip(test_label, predicted):
        if true_label == predicted_label:
            correct += 1
    print("Accuracy = ", correct / len(test_label) * 100)
    print(f"-----------------------------------------------------------------------")



# Define function to do k-folds cross_validation .
def cross_validation(dataset:dict, folds = 10):
    print("_______________________")
    fold_size = len(dataset) // folds
    data, labels = train_test_split(dataset)
    for i in range(folds):
        idx_start = i * fold_size
        idx_end = fold_size * (i+1)

        X_train = data[:idx_start] + data[idx_end:]
        Y_train = labels[:idx_start] + labels[idx_end:]

        x_test = data[idx_start:idx_end]
        y_test = labels[idx_start:idx_end]
        
        class_1, class_2 = split_by_class(X_train, Y_train)
        model_fit(i, x_test, y_test, class_1, class_2)


# MAIN CODE.
if __name__ == "__main__":
    dataset = read_data('TWOCLASS.txt')
    cross_validation(dataset)
    
    

