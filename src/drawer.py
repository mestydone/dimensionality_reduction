import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_2d(points, markers, x, y, label = ''):
    colors = ['red', 'green', 'blue', 'orange', 'black', 'yellow']
    def points_for_marker(idx, marker):
        result = []
        for i in range(0, len(points[0])):
            if (markers[i] == marker):
                result.append(points[idx][i])
        return result
        
    markers_set = set(markers)
    i = 0
    for m in markers_set:
        plt.scatter(
            points_for_marker(x, m), 
            points_for_marker(y, m), 
            c = colors[i])
        i += 1
        plt.xlabel(label)

    return plt


def plot_line(dots, top = -1, from_one = True, label = ''):
    _ , ax = plt.subplots(1)
    xdata = range(1 if from_one else 0, len(dots) + (1 if from_one else 0))
    ydata = dots
    ax.plot(xdata, ydata)
    ax.set_ylim(bottom=0)
    if (top > 0):
        ax.set_ylim(top=top)
    plt.xlabel(label)
    return plt