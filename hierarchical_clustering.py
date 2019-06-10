import numpy as np
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt


def calculate_distances(pi_n, po_data, po_centroids):
    distances = np.zeros([pi_n, pi_n])
    for i, p in enumerate(po_data):
        for j in range(i, pi_n):
            distances[i, j] = LA.norm(p - po_centroids[j])
            distances[j, i] = distances[i, j]
    return distances


if __name__ == '__main__':
    data = np.array([
        [5, 3],
        [10, 15],
        [15, 12],
        [24, 10],
        [30, 30],
        [85, 70],
        [71, 80],
        [60, 78],
        [70, 55],
        [80, 91],
    ])

    print('\n'.join([f'{i}: {coords}' for i, coords in enumerate(data)]))

    Z = shc.linkage(data, 'centroid')

    fig = plt.figure(figsize=(25, 10))
    ax = fig.add_subplot(111)
    plt.title("Dendogram")
    dn = shc.dendrogram(Z, ax=ax)

    fig = plt.figure(figsize=(25, 10))
    ax = fig.add_subplot(111)
    plt.title("Points")
    ax.scatter(Z[:, 0], Z[:, 1])
    plt.show()
