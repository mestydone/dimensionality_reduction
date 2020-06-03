import struct
import numpy as np
from scipy.spatial import distance as scipy_distance
from sklearn.neighbors import NearestNeighbors
import math

def load_data(path, take_every, old_format = False):
    with open(path, "rb") as f:
        if old_format:
            dim = struct.unpack('=i', f.read(4))[0] # data dimensionality
            count = struct.unpack('=i', f.read(4))[0] # points count
        else:
            count = struct.unpack('=i', f.read(4))[0] # points count
            dim = struct.unpack('=i', f.read(4))[0] # data dimensionality

        print('file dim:', dim)
        print('file len:', count)

        points = np.empty((dim, math.trunc(count/take_every)))
        markers = np.empty(math.trunc(count/take_every))

        print('real len:', len(points[0]))

        def skip():
            for _ in range(0, dim):
                f.read(8)
            f.read(4)
                
        for i in range(count):
            if (i % take_every != 0):
                skip()
            elif (math.trunc(i/take_every) == len(points[0])):
                break
            else:
                if not old_format:
                    markers[math.trunc(i/take_every)] = struct.unpack('=i', f.read(4))[0]
                    
                for d in range(0, dim):
                    points[d][math.trunc(i/take_every)] = struct.unpack('@d', f.read(8))[0]
                
                if old_format:
                    markers[math.trunc(i/take_every)] = struct.unpack('=i', f.read(4))[0]

    return (points, markers)


def load_noise(length, dim):
    points = np.random.uniform(0, 1, (dim,length))

    dim1_len = math.floor(length / 2)
    dim2_len = length - dim1_len
    markers = np.append(np.zeros(dim1_len), np.ones(dim2_len))

    return (points, markers)


def load_compact_clusters(length, dim, alpha=1):
    points = np.zeros((dim, length))

    dim1_len = math.floor(length / 2)
    dim2_len = length - dim1_len

    def generate_features():
        f1 = np.random.uniform(0,1,dim1_len) * alpha + np.random.random() * 10 
        f2 = np.random.uniform(0,1,dim2_len) * alpha + np.random.random() * 10
        return np.append(f1,f2)

    for d in range(dim):
        points[d] = generate_features()

    markers = np.append(np.zeros(dim1_len), np.ones(dim2_len))
    return (points, markers)





def sort(points, weights):
    for i in range(len(points)-1,0,-1):
        for j in range(i):
            if (weights[j] < weights[j+1]):
                weights[j], weights[j+1] = weights[j+1], weights[j]
                t = points[j].copy()
                points[j] = points[j+1]
                points[j+1] = t

def real_dimensionality(loss, epsilon=0.1):
    length = len(loss)
    est = np.empty(length)
    maxVal = max(loss)
    est[0] = 1#abs(loss[1] - 2*loss[0]) # first todo !!!!
    est[length-1] = 1#abs(-2*loss[length-1] + loss[length-2]) # last

    for i in range(1, length-1):
        est[i] = abs(loss[i-1] - loss[i+1]) / maxVal
        
    for i in range(length):
        if est[i] < epsilon:
            return i+1

    return length


def filter(points, markers, alpha):
    points_copy = points.copy()
    for i in range(len(points_copy)):
        points_copy[i] -= np.mean(points_copy[i])
    points_copy /= np.std(points_copy)

    length = len(points[0])
    dim = len(points)
    t_points = points.transpose((1,0))
    t_points_copy = points_copy.transpose((1,0))
    p_outliers = np.empty((length, dim))

    for i in range(length):
        p_outliers[i] = abs(t_points_copy[i]) < alpha

    p_outliers_sum = sum(p_outliers.transpose((1,0)))
    indices = p_outliers_sum == dim

    return t_points[indices].transpose((1,0)), markers[indices]


def stress(src_points, proj_points):
    src_t = src_points.transpose()
    proj_t = proj_points.transpose()

    src_dist = scipy_distance.cdist(src_t, src_t)
    proj_dist = scipy_distance.cdist(proj_t, proj_t)
    dist = np.sqrt(np.power(src_dist - proj_dist, 2)).sum()
    return dist


def points_neighbours(points, n=5):
    t_points = points.transpose()
    nbrs = NearestNeighbors(n_neighbors=n+1, algorithm='ball_tree', leaf_size=30).fit(t_points)
    _, indices = nbrs.kneighbors(t_points)

    indices = indices[:,1:]
    sets_list = []
    for idxs in indices:
        sets_list.append(set(idxs))

    return sets_list