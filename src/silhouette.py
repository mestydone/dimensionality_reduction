from scipy.spatial import distance
import numpy as np

def calculate(points, markers):
    N = points.shape[1]

    # create distance matrix
    t_points = points.transpose((1,0))
    dist_mtrx = distance.cdist(t_points, t_points)
    
    # create connectivity matrix
    conn_mtrx = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            conn_mtrx[i][j] = markers[i] == markers[j]

    dist_mtrx_same = dist_mtrx * conn_mtrx  # in same cluster
    dist_mtrx_other = dist_mtrx * (conn_mtrx == 0) # in different clusters

    sil = 0
    for i in range(N):
        a = dist_mtrx_same[i].sum() / len(dist_mtrx_same[dist_mtrx_same != 0]) / N
        b = dist_mtrx_other[i].sum() / len(dist_mtrx_other[dist_mtrx_other != 0]) / N # without min() cause there are only 2 clusters
        
        sil += (b - a) / max(a, b)

    sil /= N
    return sil
