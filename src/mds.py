import numpy as np
from scipy.spatial import distance as scipy_distance
import math

def init(points):
    return MDS_prepared(points)

class MDS_prepared:    
    def __init__(self, points):
        self.points_transposed = points.transpose((1,0)).copy()
        self.points_len = len(self.points_transposed)
        self.src_distances = scipy_distance.cdist(self.points_transposed, self.points_transposed)

    def calculate(self, dim, alpha=-1, eps=0.01):
        if (alpha == -1):
            alpha = 5e-5 / self.points_len
        res_proj, res_dist = self.scale(self.points_transposed[:,:dim], alpha, eps)
        return res_proj.transpose((1,0)), res_dist

    def scale(self, proj, alpha, eps):
        counter = 0
        prev_distance = 1e300
        distance = 0
        while(counter < 100):
            counter += 1

            # distance between points in source and projection (e_ij)
            proj_distance = scipy_distance.cdist(proj, proj)
            points_distance = proj_distance - self.src_distances

            distance = np.sqrt(np.power(points_distance, 2)).sum()
            # distance = np.power(points_distance, 2).sum() # without sqrt

            if (
                np.isnan(distance) 
                or distance == 0 
                or (prev_distance - distance) / pow(10, math.log10(distance)) < eps
                ):
                break

            # move points
            corr_vectors = []
            for k in range(0, self.points_len):
                point_vectors = (proj[k] - proj) * points_distance[:,k:k+1] # weighted vectors from k to all others
                corr_vectors.append(sum(point_vectors) * alpha)

            proj = proj - corr_vectors
            prev_distance = distance

        return proj, distance

