from scipy.spatial import distance
import math
import numpy as np

# Если определять кластеры самому
# from sklearn.cluster import AgglomerativeClustering
# clustering = AgglomerativeClustering(linkage='average')
# clustering.fit(source_data.transpose((1,0)))
# clustering.labels_

# def get_norm_std(points):
#     for i in range(len(points)):
#         points[i] -= np.mean(points[i])
#     return np.std(points)


# Modified Hubert Г statistic
def calculate(points, markers):
    N = points.shape[1]

    # make distance matrix
    t_points = points.transpose((1,0))
    dist_mtrx = distance.cdist(t_points, t_points)

    # make connectivity matrix
    conn_mtrx = np.zeros((N,N))
    for i in range(0, N-1):
        for j in range(i+1, N):
            conn_mtrx[i][j] = markers[i] != markers[j]

    hubert_sum = np.sum(dist_mtrx * conn_mtrx)
    M = N * (N-1) / 2 
    
    return hubert_sum / M

# Normalized Hubert Г statistic
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
    dist_mtrx_mean = (dist_mtrx - np.mean(dist_mtrx))
    conn_mtrx_mean = (conn_mtrx - np.mean(conn_mtrx))

    hubert_sum = 0
    sum_mtrx = dist_mtrx_mean * conn_mtrx_mean
    for i in range(0, N-1):
        for j in range(i+1, N):
            hubert_sum += sum_mtrx[i][j]

    std_dist = np.std(dist_mtrx_mean)
    std_conn = np.std(conn_mtrx_mean)
    M =  (N * (N-1) / 2)   
      
    
    return abs(1/M * hubert_sum) / (std_dist * std_conn)