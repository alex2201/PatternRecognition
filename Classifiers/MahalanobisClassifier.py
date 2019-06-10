import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA


def malahanobis_distance(cls, c):
    members_no = cls.members.shape[0]
    mean = cls.members.mean(0)

    d1 = c - mean

    v = np.array([cls.members[i] - mean for i in range(members_no)])

    sigma = 1 / members_no * np.transpose(v).dot(v)

    distance = np.transpose(d1).dot(LA.inv(sigma).dot(d1)) ** 0.5

    return distance


class MahalanobisClassifier:
    def __init__(self, classes):
        self.classes = classes

    def evaluate_vector(self, coords):
        vector = np.array(coords)

        distances = []
        distances_desc = []
        figure = plt.figure(figsize=(4, 4), dpi=100)
        ax = figure.add_subplot(111)

        for i, cls in enumerate(self.classes):
            class_name = f'class {i + 1}'

            c = np.array(coords)

            distance = malahanobis_distance(cls, c)

            distances.append(distance)
            distance_desc = f'{class_name} -> {distance:.4}'
            distances_desc.append(distance_desc)

            ax.scatter(cls.members[:, 0], cls.members[:, 1], label=distance_desc)

        ax.scatter(coords[0], coords[1], label='vector')

        result = f'\nVector belongs to class {distances.index(min(distances)) + 1}'

        figure.suptitle('Mahalanobis Classifier')
        ax.annotate(result, xy=[coords[0], coords[1]])
        ax.legend()
        ax.grid(True)
        ax.axhline(linewidth=2.0, color="black", label='x')
        ax.axvline(linewidth=2.0, color="black", label='y')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.show()
