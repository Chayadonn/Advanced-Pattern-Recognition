import math

def multivariate_normal_pdf(x, mean, covariance_matrix):
    # Dimensionality
    d = len(mean)

    # Calculate the determinant of the covariance matrix
    det_covariance = covariance_matrix[0][0] * covariance_matrix[1][1] - covariance_matrix[0][1] * covariance_matrix[1][0]

    # Calculate the inverse of the covariance matrix
    inv_covariance = [
        [covariance_matrix[1][1] / det_covariance, -covariance_matrix[0][1] / det_covariance],
        [-covariance_matrix[1][0] / det_covariance, covariance_matrix[0][0] / det_covariance]
    ]

    # Calculate the difference vector (x - μ)
    diff_vector = [x[i] - mean[i] for i in range(d)]

    # Calculate (x - μ)^T * Σ^-1 * (x - μ)
    exp_term = diff_vector[0] * (inv_covariance[0][0] * diff_vector[0] + inv_covariance[0][1] * diff_vector[1]) + \
               diff_vector[1] * (inv_covariance[1][0] * diff_vector[0] + inv_covariance[1][1] * diff_vector[1])

    # Calculate the PDF value
    pdf = 1 / (2 * math.pi * math.sqrt(det_covariance)) * math.exp(-0.5 * exp_term)

    return pdf

# Example usage:
mean_vector = [2, 3]
cov_matrix = [[2, 0.5], [0.5, 1]]
data_point = [3, 4]

pdf_value = multivariate_normal_pdf(data_point, mean_vector, cov_matrix)
print("PDF value:", pdf_value)
