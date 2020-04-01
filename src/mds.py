import numpy as np
import math

# for perfomance estimation
import time


def init(points):
    return MDS_prepared(points)

def real_dimensionality(distances, epsilon):
    dim = 0
    for i in range(len(distances)-1):
        dim += 1
        der = distances[i+1] - distances[i]
        # todo: тут надо что-то придумать с производной
        if (abs(der) < 1e-2 and abs(distances[i]) <= epsilon or abs(distances[i]) <= epsilon):
            return dim
    return dim

class MDS_prepared:    
    def __init__(self, points):
        self.points_transposed = points.transpose((1,0)).copy()
        self.points_len = len(self.points_transposed)
        self.src_distances = np.empty((self.points_len, self.points_len))

        for i in range(self.points_len):
            self.src_distances[i] = self.get_distance(self.points_transposed, self.points_transposed[i])


    # возвращает расстояния от point до всех точек из matr. E/e ij
    def get_distance(self, points, to_point):
        dist = np.zeros(len(points))
        for i in range(0, len(points)):
            dist[i] = np.linalg.norm(to_point - points[i]) # длина вектора

        return dist

    def calculate(self, dim):
        alpha = 4.7e-5 / self.points_len
        res_proj, res_dist = self.scale(self.points_transposed[:,:dim], alpha)
        return res_proj.transpose((1,0)), res_dist

    def scale(self, proj, alpha):
        counter = 0
        prevDistance = 1e300
        distance = 0
        
        time_common = time.time()
        time_eij = 0
        time_L = 0
        time_scaling = 0


        while(counter < 100):
            counter+=1
            projDistance = []


            # рассчет e_ij
            start_time_eij = time.time()            
            for i in range(0, self.points_len):
                projDistance.append(self.get_distance(proj, proj[i]))
            time_eij += time.time() - start_time_eijW

            start_time_L = time.time()    
            distance = 0 #L
            for i in range(0, self.points_len):
                for j in range(0, self.points_len):
                    distance += pow(projDistance[j][i] - self.src_distances[j][i], 2)
            time_L += time.time() - start_time_L


            if (
                np.isnan(distance) 
                or distance == 0 
                or (prevDistance - distance) / pow(10, math.log10(distance)) < 0.01
                ):
                break

            # двигаем точки
            #alpha = 1.7e-10 / dataLen
            start_time_scaling = time.time()    
            newMatr = np.empty(self.points_len * len(proj[0])).reshape(len(proj), len(proj[0]))
            for k in range(0, self.points_len):
                point = proj[k]
                newMatr[k] = point.copy()
                vector = np.zeros(len(proj[0]))
                for i in range(0, self.points_len):
                    vector += (projDistance[i][k] - self.src_distances[i][k]) * (point - proj[i])
                
                vector *= alpha
                newMatr[k] -= vector

            time_scaling += time.time() - start_time_scaling

            proj = newMatr
            prevDistance = distance
           # projSteps.append(proj)

        print('t eij:', time_eij)
        print('t L:', time_L)
        print('t scaling:', time_scaling)
        print('t common:', time.time() - time_common)

        return proj, distance

