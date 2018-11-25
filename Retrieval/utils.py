import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def draw_heatmap(data, xlabels, ylabels):
    cmap=cm.Blues
#     cmap = cm.get_cmap(plt.cm.gray, -100)
    figure = plt.figure(facecolor='w')
    ax = figure.add_subplot(1, 1, 1, position=[0.1, 0.15, 0.8, 0.8])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    vmax = data[0][0]
    vmin = data[0][0]
    for i in data:
        for j in i:
            if j > vmax:
                vmax = j
            if j < vmin:
                vmin = j
    print(vmax)
    print(vmin)
    print("heatmap")
    map = ax.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
    print("show")
    plt.show()

def testDraw():
    x_labels = ['1', '2', '3']
    y_labels = ['1', '2', '3']

    data = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            data[i][j] = i + j
    draw_heatmap(data, x_labels, y_labels)

def test():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([3, 5, 7, 6, 2, 6, 10, 15])
    plt.plot(x, y, 'r')  # 折线 1 x 2 y 3 color
    plt.plot(x, y, 'g', lw=10)  # 4 line w
    # 折线 饼状 柱状
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([13, 25, 17, 36, 21, 16, 10, 15])
    plt.bar(x, y, 0.2, alpha=1, color='b')  # 5 color 4 透明度 3 0.9
    plt.show()

if __name__ == '__main__':
    testDraw()
    pass