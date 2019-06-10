from copy import deepcopy
import cv2
import matplotlib.pyplot as plt

import numpy as np

from Classifiers.ClassifierClass import ClassifierClass
from Classifiers.EuclideanClassifier import EuclideanClassifier


def plot_classes(po_cls):
    figure = plt.figure(figsize=(4, 4), dpi=100)
    ax = figure.add_subplot(111)

    for i, c in enumerate(po_cls):
        class_name = f'class {i + 1}'
        ax.scatter(c.members[:, 0], c.members[:, 1], label=class_name)

    figure.suptitle('Classes')
    ax.legend()
    ax.grid(True)
    ax.axhline(linewidth=2.0, color="black", label='x')
    ax.axvline(linewidth=2.0, color="black", label='y')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')


def leave_one_out(po_cls, po_classifier):
    n = len(po_cls)
    confusion_matrix = np.zeros([n, n])
    sum_ = 0

    figure = plt.figure(figsize=(4, 4), dpi=100)
    ax = figure.add_subplot(111)

    plot_data = []

    global members_no

    iterations = members_no
    # for x in range(iterations):
    for i, c in enumerate(po_cls):
        result = po_classifier.evaluate_vector_r_leave_one_out(c)
        for r in set(result):
            confusion_matrix[i, r] += result.count(i)

    for x in range(n):
        plot_data.append(confusion_matrix[x, x] / iterations * 100)
        sum_ += confusion_matrix[x, x]

    x_pos = list(range(n))

    ax.bar(x=x_pos, height=plot_data)

    figure.suptitle('Leave one out')

    return sum_ / (n * iterations) * 100


def re_substitution(po_cls, po_classifier, po_members_no):
    n = len(po_cls)
    confusion_matrix = np.zeros([n, n])
    sum_ = 0

    figure = plt.figure(figsize=(4, 4), dpi=100)
    ax = figure.add_subplot(111)

    plot_data = []

    for i, c in enumerate(po_cls):
        for m in c.members:
            r = po_classifier.evaluate_vector_r(m)
            confusion_matrix[i, r] += 1

        sum_ += confusion_matrix[i, i]

    for x in range(n):
        plot_data.append(confusion_matrix[x, x] / po_members_no * 100)

    x_pos = list(range(n))

    ax.bar(x=x_pos, height=plot_data)

    figure.suptitle('Re-Substitution')

    return sum_ / po_members_no / n * 100


def cross_validation(po_cls, po_classifier, po_members_no):
    members = [deepcopy(c.members) for c in po_cls]
    values = []
    n = len(po_cls)
    half = po_members_no // 2

    figure = plt.figure(figsize=(4, 4), dpi=100)
    ax = figure.add_subplot(111)

    plot_data = [0 for x in range(n)]

    iterations = 20
    for x in range(iterations):
        test = []
        sum_ = 0

        for i, c in enumerate(po_cls):
            aux = deepcopy(members[i])
            c.members = aux[:half]
            test.append(aux[half:])

        confusion_matrix = np.zeros([n, n])

        for i in range(len(po_cls)):
            ms = test[i]

            for member in ms:
                r = po_classifier.evaluate_vector_r(member)
                confusion_matrix[i, r] += 1

            sum_ += confusion_matrix[i, i]
            plot_data[i] += confusion_matrix[i, i]

        efficiency = sum_ / half / n
        values.append(efficiency)

    plot_data = list(map(lambda x: x / half / iterations * 100, plot_data))

    x_pos = list(range(n))
    ax.bar(x=x_pos, height=plot_data)

    figure.suptitle('Cross Validation')

    ret = sum(values) / iterations * 100

    return ret


def load_image(path):
    return cv2.imread(path)


if __name__ == '__main__':
    c1 = ClassifierClass(268, 212, 20, 20)
    c2 = ClassifierClass(274, 286, 40, 40)
    c3 = ClassifierClass(416, 208, 20, 20)
    c4 = ClassifierClass(468, 284, 60, 60)

    members_no = 30

    c1.generate_members(members_no)
    c2.generate_members(members_no)
    c3.generate_members(members_no)
    c4.generate_members(members_no)

    img_path = r'C:\Users\Alexander\PycharmProjects\PatternRecognition\peppers.png'
    img = load_image(img_path)

    cls = [c1, c2, c3, c4]

    classifier = EuclideanClassifier(cls)

    # plot_classes(cls)

    print(f'Re Substitution Efficiency: {re_substitution(cls, classifier, members_no)}%')
    cross_validation = cross_validation(deepcopy(cls), classifier, members_no)
    leave_one_validation = leave_one_out(cls, classifier)
    print(f'Cross Validation Efficiency: {min(cross_validation, leave_one_validation)}%')
    print(f'Leave One Out Efficiency: {max(cross_validation, leave_one_validation)}%')

    figure = plt.figure(figsize=(4, 4), dpi=100)
    ax = figure.add_subplot(111)

    ax.imshow(img[:, :, ::-1])

    for c in cls:
        points = c.members
        ax.scatter(points[:, 0], points[:, 1])

    plt.show()
