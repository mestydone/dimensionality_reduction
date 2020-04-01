from scipy.spatial import distance
import numpy as np

# Если определять кластеры самому
# from sklearn.cluster import AgglomerativeClustering
# clustering = AgglomerativeClustering(linkage='average')
# clustering.fit(source_data.transpose((1,0)))
# clustering.labels_


def calculate(points, markers):
    N = len(points[0])

    t_points = points.copy().transpose((1,0))
    dist_mtrx = distance.cdist(t_points, t_points)
    conn_mtrx = np.zeros((N,N))

    dist_mtrx -= dist_mtrx.min()
    dist_mtrx /= dist_mtrx.max()

    for i in range(N):
        for j in range(N):
            conn_mtrx[i][j] = markers[i] == markers[j]

    hubert_sum = 0
    for i in range(1, N-1):
        for j in range(i+1, N):
            hubert_sum += dist_mtrx[i][j] * conn_mtrx[i][j]

    return hubert_sum * 2 / (N*(N-1))