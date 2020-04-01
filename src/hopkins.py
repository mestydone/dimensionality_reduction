import math
import numpy as np
from sklearn.neighbors import NearestNeighbors

# def _generate_points(length, dim, amin, amax):
#     generated = np.empty(length * dim)
#     for i in range(length*dim):
#         generated[i] =  (2 * np.random.random_sample() - 1) * math.sqrt(3)
#     return np.reshape(generated, (dim,length))

def _generate_points(length, dim, amin, amax):
    a = math.sqrt(3)
    return np.random.uniform(-a, a, (length,dim)).reshape(dim,-1)


# def _generate_points(length, dim, amin, amax):
#     return np.random.uniform(amin, amax, (length,dim)).reshape(dim,-1)

# сумма растояний от каждой точки до ее ближайшего соседа
def _distance(points):
    t_points = points.transpose((1,0))
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(t_points)
    distances, _ = nbrs.kneighbors(t_points)
    return distances[:,1].sum()

def _std_dev(points):
    mean = points.mean()
    std = sum([pow(s - mean, 2) for s in points]) / len(points)
    return math.sqrt(std)

def _mean(points):
    return sum(points) / len(points)

def calculate(points):
    # print(points)
    length = len(points[0])
    dim = len(points)

    # Нормализация исходных данных
    real_norm_points = points.copy()
    for i in range(len(points)):
        real_norm_points[i] -= _mean(real_norm_points[i])
        real_norm_points[i] /= _std_dev(real_norm_points[i])
    
    #Генерация случайных данных
    generated_points = _generate_points(length, dim, np.amin(real_norm_points, axis=1), np.amax(real_norm_points, axis=1))

    # Считаем расстояния в исходных и сгенерированных
    dist_real = _distance(real_norm_points)
    dist_generated = _distance(generated_points)

    # print('var %.3f;' % np.var(real_norm_points), 'max %.3f;' % real_norm_points.max(), 
    # 'min %.3f;' % real_norm_points.min(), 'avg %.3f;' % real_norm_points.mean())
   
    # print('var %.3f;' % np.var(generated_points), 'max %.3f;' % generated_points.max(), 
    # 'min %.3f;' % generated_points.min(), 'avg %.3f;' % generated_points.mean())

    # print(dist_real, dist_generated)

    return dist_generated / (dist_real + dist_generated)