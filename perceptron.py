import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np

from Classifiers.ClassifierClass import ClassifierClass


def plot_classes(po_cls, po_classifier):
    figure = plt.figure(figsize=(4, 4), dpi=100)
    ax = figure.add_subplot(111)

    x = [0, 1, 0, 1]
    y = [0, 0, 1, 1]

    # for i, c in enumerate(po_cls):
    #     class_name = f'class {i + 1}'
    #     ax.scatter(c.members[:, 0], c.members[:, 1], label=class_name)

    ax.scatter(x, y)

    figure.suptitle('Perceptron')

    # Plot function
    formula = lambda x: (x * -1 * po_classifier[0] - po_classifier[2]) / po_classifier[1]
    x = np.array(range(-2, 4))
    y = formula(x)
    ax.plot(x, y, label='function')

    ax.legend()
    ax.grid(True)
    ax.axhline(linewidth=2.0, color="black", label='x')
    ax.axvline(linewidth=2.0, color="black", label='y')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')


def predict(po_sample, po_weights):
    activation = po_sample.dot(po_weights)

    return 1 if activation >= 0 else 0


def train_classifier(po_samples, po_threshold):
    dim = len(po_samples[0]) - 1
    weights = np.ones(dim + 1)

    repeat = True
    no_iterations = 0

    while repeat:
        repeat = False
        no_iterations += 1

        print(weights)

        for s in po_samples:
            expected = s[-1]
            sample = np.array(list(s[:dim]) + [1])
            result = predict(sample, weights)

            if result == 1 and expected == 0:
                weights = weights - po_threshold * sample
                repeat = True
            elif result == 0 and expected == 1:
                weights = weights + po_threshold * sample
                repeat = True

    print(f'Iterations: {no_iterations}')

    return weights


if __name__ == '__main__':
    c1 = ClassifierClass(-20, 30, 20, 20)
    c2 = ClassifierClass(80, 90, 20, 40)

    cls = [c1, c2]

    for c in cls:
        c.generate_members(50)

    # x1 = np.array([[1, 1, 1]])
    samples = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
        [1, 1, 1]
    ])

    # [x, y, class]
    # samples = [
    #     list(m) + [i]
    #     for i, c in enumerate(cls)
    #     for m in c.members
    # ]

    threshold = 2

    classifier = train_classifier(samples, threshold)

    plot_classes(cls, classifier)
    plt.show()
