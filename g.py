import math

def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append([float(x) for x in line.strip().split()])
    return data

def mean_vector(data):
    num_features = len(data[0]) - 1  # Excluding the last column (target class)
    class_means = {}

    for row in data:
        class_label = row[-1]
        if class_label not in class_means:
            class_means[class_label] = [0.0] * num_features
        class_means[class_label] = [mean + value for mean, value in zip(class_means[class_label], row[:-1])]

    for class_label, mean in class_means.items():
        num_samples = len([row for row in data if row[-1] == class_label])
        class_means[class_label] = [mean_value / num_samples for mean_value in mean]

    return class_means

def covariance_matrix(data, class_means):
    num_features = len(data[0]) - 1  # Excluding the last column (target class)
    class_covariance_matrices = {}

    for class_label, mean in class_means.items():
        covariance_matrix = [[0.0] * num_features for _ in range(num_features)]
        num_samples = len([row for row in data if row[-1] == class_label])

        for row in data:
            if row[-1] == class_label:
                diff = [x - m for x, m in zip(row[:-1], mean)]
                outer_product = [[d * d2 for d2 in diff] for d in diff]
                covariance_matrix = [[c + o for c, o in zip(cov_row, outer_row)] for cov_row, outer_row in zip(covariance_matrix, outer_product)]

        covariance_matrix = [[cov_value / (num_samples - 1) for cov_value in cov_row] for cov_row in covariance_matrix]
        class_covariance_matrices[class_label] = covariance_matrix

    return class_covariance_matrices

def multivariate_normal_pdf(x, mean, covariance_matrix):
    num_features = len(x) #4
    # print(x)
    determinant = 1.0
    inverse_cov_matrix = []
 
    for i in range(num_features):

        determinant *= covariance_matrix[i][i]
        inverse_row = [-c / covariance_matrix[i][i] for c in covariance_matrix[i]]
        inverse_row[i] = -inverse_row[i]
        inverse_cov_matrix.append(inverse_row)

    prefactor = 1.0 / (math.pow((2 * math.pi), num_features / 2) * math.sqrt(determinant))
    exponent = 0.0

    for i in range(num_features):
        for j in range(num_features):
            exponent += (x[i] - mean[i]) * inverse_cov_matrix[i][j] * (x[j] - mean[j])
    return prefactor * math.exp(-0.5 * exponent)

def predict(data, class_means, class_covariance_matrices):
    predicted_labels = []
    # print(class_covariance_matrices)
    for row in data:
        # print('sample = ', row)
        max_posterior_prob = -1
        predicted_label = None
        # print('mean',class_means)
        for class_label, mean in class_means.items():
            print('class-label = ', class_label, mean)
            posterior_prob = multivariate_normal_pdf(row[:-1], mean, class_covariance_matrices[class_label])
            # print("Posterior",posterior_prob)
            if posterior_prob > max_posterior_prob:
                max_posterior_prob = posterior_prob
                predicted_label = class_label

        predicted_labels.append(predicted_label)

    return predicted_labels

def confusion_matrix(actual_labels, predicted_labels):
    unique_labels = sorted(set(actual_labels))
    num_classes = len(unique_labels)
    confusion_matrix = [[0,0],[0,0]]

    for actual_label, predicted_label in zip(actual_labels, predicted_labels):
        actual_idx = int(actual_label) - 1
        predicted_idx = int(predicted_label) - 1
        confusion_matrix[actual_idx][predicted_idx] += 1

    return confusion_matrix

def error_estimates(confusion_matrix):
    total_samples = sum(sum(row) for row in confusion_matrix)
    correct_predictions = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    accuracy = correct_predictions / total_samples
    error_rate = 1 - accuracy
    return accuracy, error_rate

def classification_information(actual_labels, predicted_labels):
    class_info = {}

    unique_labels = sorted(set(actual_labels))
    for label in unique_labels:
        class_info[label] = {
            'true_positive': 0,
            'true_negative': 0,
            'false_positive': 0,
            'false_negative': 0
        }

    for actual_label, predicted_label in zip(actual_labels, predicted_labels):
        for label in unique_labels:
            if actual_label == label and predicted_label == label:
                class_info[label]['true_positive'] += 1
            elif actual_label == label and predicted_label != label:
                class_info[label]['false_negative'] += 1
            elif actual_label != label and predicted_label == label:
                class_info[label]['false_positive'] += 1
            else:
                class_info[label]['true_negative'] += 1

    return class_info

def main():
    file_path = 'nohTWOCLASS.txt'  # Replace with the actual file path
    data = read_data(file_path)

    # 10% cross-validation
    num_samples = len(data)
    test_size = num_samples // 10
    for i in range(10):
        start_index = i * test_size
        end_index = (i + 1) * test_size
        test_data = data[start_index:end_index]
        train_data = data[:start_index] + data[end_index:]

        class_means = mean_vector(train_data)
        class_covariance_matrices = covariance_matrix(train_data, class_means)

        actual_labels = [row[-1] for row in test_data]
        predicted_labels = predict(test_data, class_means, class_covariance_matrices)
        print("PREDICTED : ",predicted_labels)
        print("ACTUAL :    ", actual_labels)
        print(f"Pass {i + 1}:")
        print("Mean Vectors:")
        for class_label, mean in class_means.items():
            print(f"Class {class_label}: {mean}")
        
        print("Covariance Matrices:")
        for class_label, cov_matrix in class_covariance_matrices.items():
            print(f"Class {class_label}:")
            for row in cov_matrix:
                print(row)
        
        print("Confusion Matrix:")
        conf_matrix = confusion_matrix(actual_labels, predicted_labels)
        for row in conf_matrix:
            print(row)
        
        accuracy, error_rate = error_estimates(conf_matrix)
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Error Rate: {error_rate:.2f}")

        class_info = classification_information(actual_labels, predicted_labels)
        print("Classification Information:")
        for class_label, info in class_info.items():
            print(f"Class {class_label}:")
            print(f"True Positive: {info['true_positive']}")
            print(f"True Negative: {info['true_negative']}")
            print(f"False Positive: {info['false_positive']}")
            print(f"False Negative: {info['false_negative']}\n")


if __name__ == "__main__":
    main()
