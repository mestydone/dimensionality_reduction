from scipy.spatial import distance
import math
import numpy as np

# Если определять кластеры самому
# from sklearn.cluster import AgglomerativeClustering
# clustering = AgglomerativeClustering(linkage='average')
# clustering.fit(source_data.transpose((1,0)))
# clustering.labels_

def calculate(points, markers):
    N = len(points[0])
    norm_points = points.copy()

    # Нормализация исходных данных
    for i in range(len(points)):
        norm_points[i] -= np.mean(norm_points[i])
    norm_points /= np.std(norm_points)

    t_points = norm_points.copy().transpose((1,0))
    dist_mtrx = distance.cdist(t_points, t_points)
    conn_mtrx = np.zeros((N,N))

    # Нормализация расстояний
    dist_mtrx /= np.max(dist_mtrx)

    for i in range(1, N-1):
        for j in range(i+1, N):
            conn_mtrx[i][j] = markers[i] == markers[j]

    hubert_sum = np.sum(dist_mtrx * conn_mtrx)
    
    return hubert_sum * 2 / (N*(N-1))