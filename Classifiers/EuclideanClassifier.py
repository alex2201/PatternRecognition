import random

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA


class EuclideanClassifier:
    def __init__(self, classes):
        self.classes = classes

    def evaluate_vector(self, coords, plot=True):
        vector = np.array(coords)

        distances = []
        distances_desc = []
        # figure = Figure((4, 4), dpi=100)
        figure = plt.figure(figsize=(4, 4), dpi=100)
        ax = figure.add_subplot(111)

        for i, cls in enumerate(self.classes):
            class_name = f'class {i + 1}'
            centroid = cls.members.mean(0)
            distance = LA.norm(vector - centroid)
            distances.append(distance)
            distance_desc = f'{class_name} -> {distance:.4}'
            distances_desc.append(distance_desc)

            ax.scatter(cls.members[:, 0], cls.members[:, 1], label=distance_desc)

        ax.scatter(coords[0], coords[1], label='vector')

        r = distances.index(min(distances))
        result = f'\nVector belongs to class {r + 1}'

        figure.suptitle('Euclidean Classifier')
        ax.annotate(result, xy=[coords[0], coords[1]])
        ax.legend()
        ax.grid(True)
        ax.axhline(linewidth=2.0, color="black", label='x')
        ax.axvline(linewidth=2.0, color="black", label='y')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.show()

        return r

    def evaluate_vector_r(self, coords):
        vector = np.array(coords)

        # figure = plt.figure(figsize=(4, 4), dpi=100)
        # ax = figure.add_subplot(111)

        distances = []

        for i, cls in enumerate(self.classes):
            centroid = cls.members.mean(0)
            distance = LA.norm(vector - centroid)
            distances.append(distance)

            class_name = f'class {i + 1}'
            distance_desc = f'{class_name} -> {distance:.4}'

        #     ax.scatter(cls.members[:, 0], cls.members[:, 1], label=distance_desc)
        #
        # ax.scatter(coords[0], coords[1], label='vector')

        r = distances.index(min(distances))
        result = f'\nVector belongs to class {r + 1}'

        # figure.suptitle('Euclidean Classifier')
        # ax.annotate(result, xy=[coords[0], coords[1]])
        # ax.legend()
        # ax.grid(True)
        # ax.axhline(linewidth=2.0, color="black", label='x')
        # ax.axvline(linewidth=2.0, color="black", label='y')
        # ax.set_xlabel('x1')
        # ax.set_ylabel('x2')
        # plt.show()

        return r

    def evaluate_vector_r_leave_one_out(self, sel_cls):
        # figure = plt.figure(figsize=(4, 4), dpi=100)
        # ax = figure.add_subplot(111)

        result = []

        for vector in sel_cls.members:
            members_aux = list(sel_cls.members[:])

            # Delete member
            members_aux = np.setdiff1d(members_aux, vector)

            distances = []

            for i, cls in enumerate(self.classes):
                if cls is sel_cls:
                    centroid = members_aux.mean(0)
                else:
                    centroid = cls.members.mean(0)

                distance = LA.norm(vector - centroid)
                distances.append(distance)

                class_name = f'class {i + 1}'
                distance_desc = f'{class_name} -> {distance:.4}'

            #     ax.scatter(cls.members[:, 0], cls.members[:, 1], label=distance_desc)
            #
            # ax.scatter(vector[0], vector[1], label='vector')

            r = distances.index(min(distances))

            result.append(r)

            # figure.suptitle('Euclidean Classifier')
            # ax.annotate(result, xy=[vector[0], vector[1]])
            # ax.legend()
            # ax.grid(True)
            # ax.axhline(linewidth=2.0, color="black", label='x')
            # ax.axvline(linewidth=2.0, color="black", label='y')
            # ax.set_xlabel('x1')
            # ax.set_ylabel('x2')
            # plt.show()

        return result
