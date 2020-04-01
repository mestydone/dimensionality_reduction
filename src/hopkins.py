import math
import numpy as np
from sklearn.neighbors import NearestNeighbors

def _generate_points(length, dim):
    generated = np.empty(length * dim)
    for i in range(length*dim):
        generated[i] =  (2 * np.random.random_sample() - 1) * math.sqrt(3)
    return np.reshape(generated, (length,dim))

# сумма растояний от каждой точки до ее ближайшего соседа
def _distance(points):
    t_points = points.transpose((1,0))
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(t_points)
    distances, _ = nbrs.kneighbors(t_points)
    return distances[:,1].sum()

def calculate(points):
    length = len(points)
    dim = len(points[0])

    # Нормализация исходных данных
    real_norm_points = points.copy()
    for i in range(len(points)):
        real_norm_points[i] -= real_norm_points[i].mean()
        real_norm_points[i] /= np.std(real_norm_points[i])
    
    #Генерация случайных данных
    generated_points = _generate_points(length, dim)

    # Считаем расстояния в исходных и сгенерированных
    dist_real = _distance(real_norm_points)
    dist_generated = _distance(generated_points)

    return dist_generated / (dist_real + dist_generated)