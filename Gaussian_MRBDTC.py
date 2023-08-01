######################################################################
# Program: Minimum Risk Decision Theoretic Classifier
# Author: Chayadon Lappabud
# Date: July 26, 2023
# Description: So hard. T T
######################################################################
import math  as m

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
    num_of_sample = len(data_class)
    cov = [[0 for y in data_class[0]] for _ in data_class]
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
def split_by_class(dataset:list, labels:list, classes:int)->list:
        return [dataset[i] for i in range(len(dataset)) if labels[i] == classes]


# Define this function to train model.
def print_info(i:int, error:int, conf_mat:list, *dataclass:list):
    print(f"------------------------------ folds {i+1} ------------------------------")
    for j in range(2):
        print(f"Class {j+1}: - Mean Vector")
        vector_mean = cal_mean_vector(dataclass[j])
        print(vector_mean)

        print(f"Class {j+1}: - Covariance Matrix")
        cov_mat = cal_covariance_matrix(vector_mean, dataclass[j])
        print_matrix(cov_mat)
        print()
    print("Confusion Matrix")
    print_matrix(conf_mat)
    print(f"Error rate -> {error*100}%")
    print(f"----------------------------------------------------------------------")


# Define this function to calculate gaussian porb density function
def gaussian_prob_den_funct(data:list, mean:list, covariance:list):
    num_features = len(mean)
    determinant = 1
    
    for i in range(num_features):
        determinant *= covariance[i][i]
    inverse_covariance = [[0 for _ in range(num_features)] for _ in range(num_features)]
    
    for i in range(num_features):
        inverse_covariance[i][i] = 1 / covariance[i][i]
    exponent = 0
    for i in range(num_features):
        for j in range(num_features):
            exponent += (data[i] - mean[i]) * inverse_covariance[i][j] * (data[j] - mean[j])
    return (1 / ((2 * m.pi)**num_features * determinant))**0.5 * m.exp(-0.5 * exponent)


# define function to do classification 
def predict_sample(sample:list, class1_mean:list, class1_cov:list, class2_mean:list, class2_cov:list)->int:
    predicted_c1 = gaussian_prob_den_funct(sample, class1_mean, class1_cov)
    predicted_c2 = gaussian_prob_den_funct(sample, class2_mean, class2_cov)
    if predicted_c1 >= predicted_c2:
        return 1
    else:
        return 2


# Define function to do k-folds cross_validation .
def cross_validation(dataset:dict, folds:int = 10):
    print("_______________________")
    fold_size = len(dataset) // folds
    data, labels = train_test_split(dataset)
    total_error = 0

    for i in range(folds):
        idx_start = i * fold_size
        idx_end = fold_size * (i+1)

        X_train = data[:idx_start] + data[idx_end:]
        Y_train = labels[:idx_start] + labels[idx_end:]

        x_test = data[idx_start:idx_end]
        y_test = labels[idx_start:idx_end]

        class_1 = split_by_class(X_train, Y_train, 1)
        class_2 = split_by_class(X_train, Y_train, 2)

        class1_mean = cal_mean_vector(class_1)
        class1_cov = cal_covariance_matrix(class1_mean, class_1)

        class2_mean = cal_mean_vector(class_2)
        class2_cov = cal_covariance_matrix(class2_mean, class_2)

        predicted = []
        for sample in x_test:
            predicted.append(predict_sample(sample, class1_mean, class1_cov, class2_mean, class2_cov))

        # Calculate confusion matrix
        confusion_matrix = [[0, 0], 
                            [0, 0]]
        for k in range(len(y_test)):
            confusion_matrix[y_test[k] - 1][predicted[k] - 1] += 1
        error_rate = (confusion_matrix[0][1] + confusion_matrix[1][0]) / len(predicted)
        total_error += error_rate

        print_info(i, error_rate, confusion_matrix,class_1, class_2)
    
    print(f"Average Error -> {total_error/folds*100} %")
    print(f"Average Accuracy -> {(1-(total_error/folds))*100} %")

# main.
if __name__ == "__main__":
    dataset = read_data('2csTWOCLASS.txt')
    cross_validation(dataset, folds=10)
    
    

