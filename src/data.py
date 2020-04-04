import struct
import numpy as np
import math

def load_data(path, take_every, old_format = False):
    with open(path, "rb") as f:
        if old_format:
            dim = struct.unpack('=i', f.read(4))[0] # размерность
            count = struct.unpack('=i', f.read(4))[0] # количество точек
        else:
            count = struct.unpack('=i', f.read(4))[0] # количество точек
            dim = struct.unpack('=i', f.read(4))[0] # размерность

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
    points = np.random.rand(dim,length)
    markers = np.zeros(length)
    return (points, markers)


def sort(points, weights):
    for i in range(len(points)-1,0,-1):
        for j in range(i):
            if (weights[j] < weights[j+1]):
                weights[j], weights[j+1] = weights[j+1], weights[j]
                t = points[j].copy()
                points[j] = points[j+1]
                points[j+1] = t

def real_dimensionality(loss, epsilon):
    dim = 0
    for i in range(len(loss)-1):
        dim += 1
        der = loss[i+1] - loss[i]
        # todo: тут надо что-то придумать с производной
        if (abs(der) < 1e-2 and abs(loss[i]) <= epsilon or abs(loss[i]) <= epsilon):
            return dim
    return dim


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

    # todo фильтровать после нормализации
    for i in range(length):
        p_outliers[i] = abs(t_points_copy[i]) < alpha

    p_outliers_sum = sum(p_outliers.transpose((1,0)))
    indices = p_outliers_sum == dim

    return t_points[indices].transpose((1,0)), markers[indices]

