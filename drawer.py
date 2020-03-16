import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw2d(data, x, y, markers = [], label = ''):
    colors = ['red', 'green', 'blue', 'orange', 'black', 'yellow']
    for i in range(0, len(markers)):
        plt.scatter(data.axisM(x, markers[i]), data.axisM(y, markers[i]), c = colors[i])
        plt.xlabel(label)
    plt.show()


def drawLine(dots, top = -1, fromOne = True):
    #plt.plot(range(1, len(dots)+1), dots)

    f, ax = plt.subplots(1)

    xdata = range(1 if fromOne else 0, len(dots) + (1 if fromOne else 0))
    ydata = dots
    ax.plot(xdata, ydata)
    ax.set_ylim(bottom=0)
    if (top > 0):
        ax.set_ylim(top=top)
    plt.show(f)