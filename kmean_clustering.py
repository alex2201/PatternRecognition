# Imports
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import scipy.linalg as la
import cv2


def kmeans(po_data, pi_k):
    n = len(po_data)
    picks = get_picks(n, pi_k)
    picked_data = [po_data[i] for i in picks]
    means = np.array(picked_data)  # Init means
    temp_clusters = dict()
    has_changed = True
    counter = 1

    print('BEGIN KMEANS ALGORITHM...')

    while has_changed:
        temp_clusters = {i: np.array([m]) for i, m in enumerate(picked_data)}  # Init clusters

        for d in po_data:
            cluster_index = min([[euclidean_distance(d, m), i] for i, m in enumerate(means)], key=lambda x: x[0])[1]
            cluster = temp_clusters[cluster_index]

            concatenate = True

            for c in cluster:
                if np.all(c == d):
                    concatenate = False
                    break

            if concatenate:
                temp_clusters[cluster_index] = np.concatenate([cluster, [d]])

        aux_means = np.array([temp_clusters[i].mean(0) for i in range(pi_k)])
        has_changed = not (means == aux_means).all()

        if has_changed:
            means = deepcopy(aux_means)

        counter += 1

    print(f'Iterations: {counter}')

    return means, temp_clusters


def euclidean_distance(a, b):
    return la.norm(a - b)


def get_picks(pi_n, pi_k):
    picks = []
    array = np.array(range(pi_n))

    while len(picks) != pi_k:
        choice = rand.choice(array)

        if choice not in picks:
            picks.append(choice)

    return picks


def load_image(path):
    return cv2.imread(path)


def parse_img(po_img):
    print(po_img[0, 0, :])


if __name__ == '__main__':

    # colors = ['blue', 'red', 'black']

    img_path = r'C:\Users\Alexander\PycharmProjects\PatternRecognition\mexico.png'
    img = load_image(img_path)

    blur = cv2.GaussianBlur(img, (11, 11), 0)

    data = blur.reshape((-1, 3))

    # convert to np.float32
    data = np.float32(data)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, centroids = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    centroids = np.uint8(centroids)
    clusters = centroids[label.flatten()]
    cluster_img = clusters.reshape(img.shape)

    final_img = cluster_img[:, :, ::-1]  # Get original colors

    print('CENTROIDS:')

    for i, c in enumerate(centroids):
        print(f'{i} -> {c}')

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(222)
    ax1.imshow(cluster_img)
    ax2 = fig1.add_subplot(221)
    ax2.imshow(img[:, :, ::-1])
    ax3 = fig1.add_subplot(223)
    ax3.imshow(cluster_img[:, :, ::-1])
    plt.show()
