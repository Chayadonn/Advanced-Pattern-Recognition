######################################################################
# Program: Minimum Risk Decision Theoretic Classifier
# Author: Chayadon Lappabud
# Date: July 26, 2023
# Description: very hard. T T
######################################################################
import math  as m

transpost = lambda x: list(zip(*x))
print_matrix = lambda x: list(map(print, x))

# Create function reading dataset.
def read_data(file_path:str)->list:
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
        mean_vector.append(round(sum / num_of_samples, 10))
    return mean_vector


# Define function cal_covariance_matrix.
def cal_covariance_matrix(mean_vec:list, data_class:list)->list:
    num_of_sample = len(data_class)
    cov = [[0 for y in data_class[0]] for _ in data_class]
    i = 0
    for sample in data_class:
        j = 0
        for cols in sample:
            cov[i][j] = round(cols - mean_vec[j], 10)
            j += 1
        i += 1
    inv_cov = transpost(cov)
    covariance = [[round(sum(a * b for a, b in zip(inv_cova, cova))/num_of_sample,10)
                                    for cova in zip(*cov)] 
                                        for inv_cova in inv_cov]
    return covariance


# Define function train_test_split to split the data and labels.
def train_test_split(dataset:list)->list:
    data = [d[:-1] for d in dataset]
    labels = [int(label[-1]) for label in dataset]
    return data, labels


# Define function split_by_class.
def split_by_class(dataset:list, labels:list, classes:int)->list:
        return [dataset[i] for i in range(len(dataset)) if labels[i] == classes]


# Define this function to calculate multivariate normal prob density function.
def multivariate_normal_prob_density_function(x:list, mean:list, covariance_matrix:list)->float:
    num_features = len(x)
    det = 1.0
    inverse_covariance_matrix = []

    for i in range(num_features):
        det *= covariance_matrix[i][i]
        inverse_row = [-c / covariance_matrix[i][i] for c in covariance_matrix[i]]
        inverse_row[i] = -inverse_row[i]
        inverse_covariance_matrix.append(inverse_row)

    prefactor = m.pow((2 * m.pi), -num_features/2) * det**(-1/2)
    exponent = 0.0

    for i in range(num_features):
        for j in range(num_features):
            exponent += (x[i] - mean[i]) * inverse_covariance_matrix[i][j] * (x[j] - mean[j])
    return prefactor * m.exp(-0.5 * exponent)


# Predict the class of given data.
def inference(data:list, mean1:list, covariance_matrices1:list, mean2:list, covariance_matrices2:list)->int:
    predicted_labels = []
    means = [mean1, mean2]
    covariance_mat = {1:covariance_matrices1, 2:covariance_matrices2}

    for sample in data:
        max_posterior_prob = 0
        predicted_label = None
        for mean,class_label in zip(means, [1,2]):
            posterior_prob = multivariate_normal_prob_density_function(sample, 
                                                                       mean, 
                                                                       covariance_mat[class_label])
            if posterior_prob > max_posterior_prob:
                max_posterior_prob = posterior_prob
                predicted_label = class_label
        predicted_labels.append(predicted_label)
    return predicted_labels


# Define this function to train model.
def model_fit(i:int, test_data:list, test_label:list, *dataclass:list)->int:
    print(f"------------------------------ folds {i+1} ------------------------------")
    for j in range(len(dataclass)):
        print(f"Class {j+1}: - Mean Vector")
        vector_mean = cal_mean_vector(dataclass[j])
        print(vector_mean)

        print(f"Class {j+1}: - Covariance Matrix")
        cov_mat = cal_covariance_matrix(vector_mean, dataclass[j])
        print_matrix(cov_mat)
        print()
    
    vector_mean = cal_mean_vector(dataclass[0])
    vector_mean2 = cal_mean_vector(dataclass[1])

    cov_mat = cal_covariance_matrix(vector_mean, dataclass[0])
    cov_mat2 = cal_covariance_matrix(vector_mean2, dataclass[1])

    predicted = inference(test_data, vector_mean, cov_mat, vector_mean2, cov_mat2)

    correct = 0
    for i in range(len(test_data)):
        if test_label[i] == predicted[i]:
            correct += 1
        else:
            print("worng inference : ", test_data[i])
    
    print("Accuracy = ", (correct/len(test_label))*100)
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(test_label, predicted)
    print_matrix(conf_matrix)
    print(f"-----------------------------------------------------------------------")
    return correct

# Calculate confusion matrix and return it
def confusion_matrix(actual_labels:list, predicted_labels:list)->list:
    confusion_matrix = [[0,0],[0,0]]
    for actual_label, predicted_label in zip(actual_labels, predicted_labels):
        actual_idx = int(actual_label) - 1
        predicted_idx = int(predicted_label) - 1
        confusion_matrix[actual_idx][predicted_idx] += 1
    return confusion_matrix


# Define function to do k-folds cross_validation.
def cross_validation(dataset:dict, folds = 10):
    fold_size = len(dataset) // folds
    data, labels = train_test_split(dataset)
    accuracy = 0
    for i in range(folds):
        idx_start = i * fold_size
        idx_end = fold_size * (i+1)

        X_train = data[:idx_start] + data[idx_end:]
        Y_train = labels[:idx_start] + labels[idx_end:]

        x_test = data[idx_start:idx_end]
        y_test = labels[idx_start:idx_end]
        
        class_1 = split_by_class(X_train, Y_train, 1)
        class_2 = split_by_class(X_train, Y_train, 2)
        
        accuracy += model_fit(i, x_test, y_test, class_1, class_2)

    print(f'Average accuracy = {round(accuracy/len(dataset)*100,4)}')
    print(f'Average error = {round((1-accuracy/len(dataset))*100,4)}')


# MAIN CODE.
if __name__ == "__main__":
    dataset = read_data('124TWOCLASS.txt')
    cross_validation(dataset)