import math
import numpy as np
from sklearn.neighbors import NearestNeighbors


def _info(real_points, generated_points, real_dist, generated_dist):
    print('var %.3f;' % np.var(real_points), 'max %.3f;' % real_points.max(), 
    'min %.3f;' % real_points.min(), 'avg %.3f;' % real_points.mean())
   
    print('var %.3f;' % np.var(generated_points), 'max %.3f;' % generated_points.max(), 
    'min %.3f;' % generated_points.min(), 'avg %.3f;' % generated_points.mean())

    print(real_dist, generated_dist)

# def _generate_points(length, dim):
#     generated = np.empty(length * dim)
#     for i in range(length*dim):
#         generated[i] =  (2 * np.random.random_sample() - 1) * math.sqrt(3)
#     return np.reshape(generated, (dim,length))

def _generate_points(length, dim):
    a = math.sqrt(3)
    return np.random.uniform(-a, a, (length,dim)).reshape(dim,-1)

# сумма растояний от каждой точки до ее ближайшего соседа
def _distance(points):
    t_points = points.transpose((1,0))
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(t_points)
    distances, _ = nbrs.kneighbors(t_points)
    return distances[:,1].sum()

def calculate(points):
    length = len(points[0])
    dim = len(points)

    # Нормализация исходных данных
    real_norm_points = points.copy()
    for i in range(len(points)):
        real_norm_points[i] -= np.mean(real_norm_points[i])
    real_norm_points /= np.std(real_norm_points)

    #Генерация случайных данных
    generated_points = _generate_points(length, dim)

    # Считаем расстояния в исходных и сгенерированных
    dist_real = _distance(real_norm_points)
    dist_generated = _distance(generated_points)
    
    # Вывод информации о данных
    # _info(real_norm_points, generated_points, dist_real, dist_generated)

    return dist_generated / (dist_real + dist_generated)