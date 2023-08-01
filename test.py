import math

# Function to read data from file
def read_data_from_file(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            values = line.strip().split()
            data.append([float(val) for val in values])
        # print(data)
    return data

# Function to separate data and labels from the given dataset
def separate_data_and_labels(dataset):
    data = [row[:-1] for row in dataset]
    labels = [int(row[-1]) for row in dataset]
    return data, labels

# Function to calculate mean vector from the data
def calculate_mean(data):
    num_features = len(data[0])
    mean = [0] * num_features
    num_samples = len(data)

    for sample in data:
        for i in range(num_features):
            mean[i] += sample[i]

    mean = [m / num_samples for m in mean]
    return mean

# Function to calculate covariance matrix from the data
def calculate_covariance(data, mean):
    num_features = len(data[0])
    num_samples = len(data)
    covariance = [[0 for _ in range(num_features)] for _ in range(num_features)]
    for sample in data:
        for i in range(num_features):
            for j in range(num_features):
                covariance[i][j] += (sample[i] - mean[i]) * (sample[j] - mean[j])

    covariance = [[cov / num_samples for cov in row] for row in covariance]
    return covariance

# Function to calculate Gaussian probability density function
def gaussian_pdf(x, mean, covariance):
    num_features = len(mean)
    determinant = 1

    for i in range(num_features):
        determinant *= covariance[i][i]

    inv_covariance = [[0 for _ in range(num_features)] for _ in range(num_features)]

    for i in range(num_features):
        inv_covariance[i][i] = 1 / covariance[i][i]

    exponent = 0
    for i in range(num_features):
        for j in range(num_features):
            exponent += (x[i] - mean[i]) * inv_covariance[i][j] * (x[j] - mean[j])

    return (1 / math.sqrt((2 * math.pi) ** num_features * determinant)) * math.exp(-0.5 * exponent)

# Function to perform classification using Bayes decision rule
def classify_sample(sample, class1_mean, class1_covariance, class2_mean, class2_covariance):
    pdf_class1 = gaussian_pdf(sample, class1_mean, class1_covariance)
    pdf_class2 = gaussian_pdf(sample, class2_mean, class2_covariance)
    # print(f"sample = {sample}")
    if pdf_class1 >= pdf_class2:
        return 1
    else:
        return 2

# Function to perform cross-validation and calculate average error rate
def cross_validation_and_test(dataset, num_folds=10):
    data, labels = separate_data_and_labels(dataset)
    fold_size = len(data) // num_folds
    total_error = 0

    for i in range(num_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size
        test_data = data[start_idx:end_idx]
        test_labels = labels[start_idx:end_idx]
        train_data = data[:start_idx] + data[end_idx:]
        train_labels = labels[:start_idx] + labels[end_idx:]

        class1_data = [train_data[j] for j in range(len(train_data)) if train_labels[j] == 1]
        class2_data = [train_data[j] for j in range(len(train_data)) if train_labels[j] == 2]
    
        class1_mean = calculate_mean(class1_data)
        class2_mean = calculate_mean(class2_data)

        class1_covariance = calculate_covariance(class1_data, class1_mean)
        class2_covariance = calculate_covariance(class2_data, class2_mean)

        predictions = [classify_sample(sample, class1_mean, class1_covariance, class2_mean, class2_covariance) for sample in test_data]

        confusion_matrix = [[0, 0], [0, 0]]
        for j in range(len(test_labels)):
            confusion_matrix[test_labels[j] - 1][predictions[j] - 1] += 1

        error_rate = (confusion_matrix[0][1] + confusion_matrix[1][0]) / len(predictions)
        total_error += error_rate

        print(f"Fold {i + 1}")
        print("Mean Vectors:")
        print("Class 1:", class1_mean)
        print("Class 2:", class2_mean)
        print("Covariance Matrices:")
        print("Class 1:")
        for row in class1_covariance:
            print(row)
        print("Class 2:")
        for row in class2_covariance:
            print(row)
        print("Confusion Matrix:")
        for row in confusion_matrix:
            print(row)
        print("Error Rate:", error_rate)
        print("-------------------------------------------------------")

    print("Average Error Rate:", total_error / num_folds)
    print("Average Accuracy Rate:", 1-(total_error / num_folds))


# Main function
if __name__ == "__main__":
    file_path = "fake_TWOCLASS.txt"
    dataset = read_data_from_file(file_path)
    cross_validation_and_test(dataset, num_folds=10)
