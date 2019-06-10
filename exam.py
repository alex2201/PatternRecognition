import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import math


def is_prime(n):
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))


p1 = list(filter(is_prime, range(1, 1001)))
p2 = list(filter(is_prime, range(3000, 4001)))
p3 = list(filter(is_prime, range(6000, 7001)))

c1 = np.array([
    p1, p1, p1
]).transpose()

c2 = np.array([
    p2, p2, p2
]).transpose()

c3 = np.array([
    p3, p3, p3
]).transpose()

m1 = c1.mean(0)
m2 = c2.mean(0)
m3 = c3.mean(0)

v = [311, 311, 311]

for n in v:
    if not is_prime(n):
        print('El vector no es primo')
        exit()

d1 = linalg.norm(m1 - v)
d2 = linalg.norm(m2 - v)
d3 = linalg.norm(m3 - v)

distances = [d1, d2, d3]


result = f'Vector belongs to class{distances.index(min(distances)) + 1}'

figure = plt.figure(figsize=(4, 4), dpi=100)
ax = figure.add_subplot(111)
ax.annotate(result, xy=[v[0], v[1]])
ax.annotate('media 1', xy=[m1[0], m1[1]])
ax.annotate('media 2', xy=[m2[0], m2[1]])
ax.annotate('media 3', xy=[m3[0], m3[1]])
ax.scatter(c1[:, 0], c1[:, 1], label=f'Class 1 dist: {d1}')
ax.scatter(c2[:, 0], c2[:, 1], label=f'Class 2 dist: {d2}')
ax.scatter(c3[:, 0], c3[:, 1], label=f'Class 3 dist: {d3}')
ax.scatter(m1[0], m1[1])
ax.scatter(m2[0], m2[1])
ax.scatter(m3[0], m3[1])
ax.scatter(v[0], v[1], label='vector')
ax.legend()
ax.grid(True)
ax.axhline(linewidth=2.0, color="black", label='x')
ax.axvline(linewidth=2.0, color="black", label='y')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()

