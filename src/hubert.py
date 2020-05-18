from scipy.spatial import distance
import math
import numpy as np

# def get_variance(matrix, mean):
#     N = len(matrix)
#     M = N * (N-1) / 2
#     mean_quad = mean*mean
#     var = 0
#
#     for i in range(N-1):
#         for j in range(i+1, N):
#             var += pow(matrix[i][j], 2) - mean_quad
#
#     return var / M


# Modified Hubert Г statistic
def calculate(points, markers):
    N = points.shape[1]

    # create distance matrix
    t_points = points.transpose((1,0))
    dist_mtrx = distance.cdist(t_points, t_points)

    # create connectivity matrix
    conn_mtrx = np.zeros((N,N))
    for i in range(0, N-1):
        for j in range(i+1, N):
            conn_mtrx[i][j] = markers[i] != markers[j]

    hubert_sum = np.sum(dist_mtrx * conn_mtrx)
    M = N * (N-1) / 2 
    
    return hubert_sum / M

# mean of top triangle
def get_mean(matrix):
    N = len(matrix)
    M = N * (N-1) / 2
    mean = 0

    for i in range(N-1):
        for j in range(i+1, N):
            mean += matrix[i][j]

    return mean / M

def calculate_norm(points, markers):
    N = points.shape[1]

    # create distance matrix
    t_points = points.transpose((1,0))
    dist_mtrx = distance.cdist(t_points, t_points)

    # create connectivity matrix
    conn_mtrx = np.zeros((N,N))
    for i in range(0, N):
        for j in range(0, N):
            conn_mtrx[i][j] = markers[i] != markers[j]

    # normalize matrices
    dist_mean = get_mean(dist_mtrx)
    conn_mean = get_mean(conn_mtrx)

    dist_mtrx_mean = (dist_mtrx - dist_mean)
    conn_mtrx_mean = (conn_mtrx - conn_mean)

    hubert_sum = 0
    sum_mtrx = dist_mtrx_mean * conn_mtrx_mean
    for i in range(0, N-1):
        for j in range(i+1, N):
            hubert_sum += sum_mtrx[i][j]

    std_dist = np.std(dist_mtrx)
    std_conn = np.std(conn_mtrx)
    M =  (N * (N-1) / 2)   
    
    return hubert_sum / (M * std_dist * std_conn)