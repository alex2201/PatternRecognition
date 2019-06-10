import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


def malahanobis_distance(cls, c):
    members_no = cls.members.shape[0]
    mean = cls.members.mean(0)

    d1 = c - mean

    v = np.array([cls.members[i] - mean for i in range(members_no)])

    sigma = 1 / members_no * np.transpose(v).dot(v)

    distance = np.transpose(d1).dot(LA.inv(sigma).dot(d1)) ** 0.5

    return distance, sigma


def calculate_probability(cls, coords):
    malahanobis, sigma = malahanobis_distance(cls, coords)

    res = np.exp(-0.5 * malahanobis) / (2 * np.pi * (LA.det(sigma) ** -0.5))

    return res


class BayesianClassifier:
    def __init__(self, classes):
        self.classes = classes

    def evaluate_vector(self, coords):
        vector = np.array(coords)

        figure = plt.figure(figsize=(4, 4), dpi=100)
        ax = figure.add_subplot(111)

        probabilities = [calculate_probability(cls, coords) for cls in self.classes]

        ax.scatter(coords[0], coords[1], label='vector')

        probabilities_sum = sum(probabilities)
        probabilities_normalized = [p / probabilities_sum * 100 for p in probabilities]

        probabilities_desc = [f'class{i + 1} -> {pr:.4}%' for i, pr in enumerate(probabilities_normalized)]

        [ax.scatter(cls.members[:, 0], cls.members[:, 1], label=probabilities_desc[i]) for i, cls in
         enumerate(self.classes)]

        result = f'\nVector belongs to class {probabilities_normalized.index(max(probabilities_normalized)) + 1}'

        figure.suptitle('Bayesian Classifier')
        ax.annotate(result, xy=[coords[0], coords[1]])
        ax.legend()
        ax.grid(True)
        ax.axhline(linewidth=2.0, color="black", label='x')
        ax.axvline(linewidth=2.0, color="black", label='y')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.show()
