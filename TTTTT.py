import math

def calculate_mean(data):
    num_samples = len(data)
    num_features = len(data[0])
    mean_vector = [0.0] * num_features

    for sample in data:
        for i in range(num_features):
            mean_vector[i] += sample[i]

    mean_vector = [x / num_samples for x in mean_vector]
    return mean_vector


def calculate_covariance(data, mean_vector):
    num_samples = len(data)
    num_features = len(data[0])
    covariance_matrix = [[0.0] * num_features for _ in range(num_features)]

    for sample in data:
        diff = [sample[i] - mean_vector[i] for i in range(num_features)]
        for i in range(num_features):
            for j in range(num_features):
                covariance_matrix[i][j] += diff[i] * diff[j]

    covariance_matrix = [[x / num_samples for x in row] for row in covariance_matrix]
    return covariance_matrix


def calculate_a_priori_probabilities(data):
    num_samples = len(data)
    num_classes = len(set([sample[-1] for sample in data]))
    a_priori_probs = [1.0 / num_classes] * num_classes
    return a_priori_probs


def calculate_01_loss(true_class, predicted_class):
    return 0 if true_class == predicted_class else 1


def classify_sample(sample, class_info):
    class_probs = []
    for class_index, (mean, covariance, a_priori_prob) in enumerate(class_info):
        exponent = calculate_exponent(sample, mean, covariance)
        class_probs.append(exponent + math.log(a_priori_prob))

    predicted_class = sorted(enumerate(class_probs), key=lambda x: x[1], reverse=True)[0][0] + 1
    return predicted_class


def calculate_exponent(sample, mean, covariance):
    num_features = len(sample)
    exponent = 0.0
    for i in range(num_features):
        for j in range(num_features):
            exponent += (sample[i] - mean[i]) * (sample[j] - mean[j]) * covariance[i][j]

    return exponent


def read_data_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split('\t')
            data.append([float(val) for val in values])
        # print(data)
    return data


def print_mean_and_covariance(class_info):
    for class_index, (mean, covariance, _) in enumerate(class_info, start=1):
        print(f"Class {class_index} - Mean Vector:")
        print(mean)
        print("Covariance Matrix:")
        for row in covariance:
            print(row)
        print()

def perform_cross_validation(data, num_folds):
    fold_size = len(data) // num_folds
    accuracy_sum = 0.0

    for fold in range(num_folds):
        test_data = data[fold * fold_size: (fold + 1) * fold_size]
        train_data = data[:fold * fold_size] + data[(fold + 1) * fold_size:]
        
        mean_vectors = []
        covariance_matrices = []
        class_info = []
        for class_index in set([sample[-1] for sample in train_data]):
            class_samples = [sample[:-1] for sample in train_data if sample[-1] == class_index]
            # print(class_samples)
            mean_vector = calculate_mean(class_samples)
            covariance_matrix = calculate_covariance(class_samples, mean_vector)
            a_priori_prob = calculate_a_priori_probabilities(train_data)[int(class_index) - 1]
            class_info.append((mean_vector, covariance_matrix, a_priori_prob))
            mean_vectors.append(mean_vector)
            covariance_matrices.append(covariance_matrix)

        print(f"Fold {fold + 1}:")
        print_mean_and_covariance(class_info)

        confusion_matrix = [[0] * len(class_info) for _ in range(len(class_info))]
        error_sum = 0

        for test_sample in test_data:
            true_class = test_sample[-1]
            predicted_class = classify_sample(test_sample[:-1], class_info)
            error_sum += calculate_01_loss(true_class, predicted_class)
            confusion_matrix[int(true_class) - 1][int(predicted_class) - 1] += 1

        accuracy = 1 - (error_sum / len(test_data))
        accuracy_sum += accuracy
        print(f"Fold {fold + 1} - Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:")
        for row in confusion_matrix:
            print(row)
        print()

    average_accuracy = accuracy_sum / num_folds
    print(f"Average Classification Accuracy: {average_accuracy:.2f}")


if __name__ == "__main__":
    file_path = "fake_TWOCLASS.txt"  # Replace this with the actual file path
    data = read_data_from_file(file_path)
    num_folds = 10

    perform_cross_validation(data, num_folds)