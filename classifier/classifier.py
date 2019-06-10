import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


def generate_class_members(px, py, pr):
    global representatives_no
    x1 = np.random.uniform(px - pr, px + pr, representatives_no + 1)
    x2 = np.random.uniform(py - pr, py + pr, representatives_no + 1)

    return np.array([x1, x2])


if __name__ == '__main__':
    class_no = int(input('Introduce number of classes: ').strip())
    representatives_no = int(input('Introduce number of representatives for each class: ').strip())

    positions = []

    for i in range(class_no):
        current_class_no = i + 1
        x, y, r = list(map(float, input(
            f'Introduce coordinates and radio for class {current_class_no}(x1,x2,r): ').strip().split(',')))
        positions.append((x, y, r))

    class_members = []

    print('\nGENERATING MEMBERS...\n')

    for x, y, r in positions:
        class_members.append(generate_class_members(x, y, r))

    vx, vy = list(map(float, input('Introduce vector coordinates(x1,x2): ').strip().split(',')))

    vector = np.array([vx, vy])

    distances = []

    fig, ax = plt.subplots(1, 1)

    print('\nCALCULATING DISTANCES...\n')

    for i, class_member in enumerate(class_members):
        class_name = f'c{i + 1}'
        mean = class_member.mean(1)
        distance = LA.norm(vector - mean)
        distances.append(distance)

        print(f'{class_name} center: {mean} -> Distance from input vector: {distance}')

        ax.scatter(class_member[0], class_member[1], label=class_name)

    ax.scatter(vx, vy, label='vector')

    result = f'\nVector belongs to class {distances.index(min(distances)) + 1}'

    print('\nPLOTTING...\n')

    fig.suptitle('2D Classifier')
    ax.annotate(result, xy=[vx, vy])
    ax.legend()
    ax.grid(True)
    ax.axhline(linewidth=2.0, color="black", label='x1')
    ax.axvline(linewidth=2.0, color="black", label='x2')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.show()
