import matplotlib.lines as lines
import matplotlib.pyplot as plt
from numpy import linalg as LA


def cls_neighbors(i, neighbors):
    neighbors = len(list(filter(lambda x: x[1] == i, neighbors)))
    return neighbors


class KNNClassifier:
    def __init__(self, classes):
        self.classes = classes

    def calculate_distances(self, v):
        points = [((LA.norm(m - v)), i, m) for i, c in enumerate(self.classes) for m in c.members]
        points.sort(key=lambda x: x[0])

        return points

    def evaluate_vector(self, v, k):
        neighbors = self.calculate_distances(v)[:k]

        cls_no = len(self.classes)

        probabilities = [(cls_neighbors(i, neighbors) / k, i) for i in range(cls_no)]
        r = max(probabilities, key=lambda x: x[0])

        self.plot_points(probabilities, v, r[1], neighbors)

    def plot_points(self, prob, v, r, neighbors):
        cls_no = len(self.classes)

        figure = plt.figure(figsize=(4, 4), dpi=100)
        ax = figure.add_subplot(111)

        [ax.scatter(cls.members[:, 0], cls.members[:, 1], label=f'{p[0] * 100}%') for cls, p in zip(self.classes, prob)]
        [ax.add_line(lines.Line2D([n[2][0], v[0]], [n[2][1], v[1]])) for n in neighbors]

        ax.scatter(v[0], v[1], label='vector')

        result = f'\nVector belongs to class {r + 1}'

        figure.suptitle('KNN Classifier')
        ax.annotate(result, xy=[v[0], v[1]])
        ax.legend()
        ax.grid(True)
        ax.axhline(linewidth=2.0, color="black", label='x')
        ax.axvline(linewidth=2.0, color="black", label='y')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.show()
