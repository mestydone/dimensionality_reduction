import numpy as np
import math

def calculate(points):
    cov_mat = np.cov(points)
    e_vals, e_vecs = np.linalg.eig(cov_mat)
    pca_points = np.dot(e_vecs.transpose((1,0)), points)
    return (pca_points, relative_error(pca_points, e_vals))

# Отношение остаточной дисперсии к объясненной 
def relative_error(points, e_vals):
    pca_error = []
    for i in range(len(points)):
        err = math.sqrt(abs(e_vals[i+1:].sum()/e_vals[:i+1].sum()))
        pca_error.append(err)
    return pca_error