import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

def _line_top(dots):
    top = np.max(dots)
    top *= 1.3 if top > 0 else 0.7
    return top

def _line_bottom(dots):
    bottom = np.min(dots)
    bottom *= 1.3 if bottom < 0 else 0.7
    return bottom

def plot_line(dots, top = None, bottom = None, from_one = True, label = ''):
    if (top is None):
        top = _line_top(dots)

    if (bottom is None):
        bottom = _line_bottom(dots)

    _ , ax = plt.subplots(1)
    xdata = range(1 if from_one else 0, len(dots) + (1 if from_one else 0))
    ydata = dots
    ax.plot(xdata, ydata)
    ax.set_ylim(top=top, bottom=bottom)
    plt.xlabel(label)
    return plt

def plot_multiline(dots_list, top = None, bottom = None, from_one = True, label = '', legend = [], show_range=True):
    if (top is None):
        top = max( [_line_top(dots) for dots in dots_list] )

    if (bottom is None):
        bottom = min( [_line_bottom(dots) for dots in dots_list] )

    _ , ax = plt.subplots(1)
    xdata = range(1 if from_one else 0, len(dots_list[0]) + (1 if from_one else 0))

    for dots in dots_list:
        ax.plot(xdata, dots)   

    ax.set_ylim(top=top, bottom=bottom)

    if (show_range):
        for i in range(len(dots_list)):
            legend[i] += ' [' + '%.3f' % min(dots_list[i]) +'; ' + '%.3f' % max(dots_list[i]) + ']' 
    
    plt.legend(legend) 

    plt.xlabel(label)
    return plt