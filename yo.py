import numpy as np
a = np.arange(10)
print(a)

id = np.random.randint(2, size=[10])

X = np.random.randint(10, size=[10,3])

print(id)

indices = np.where(id == 0)

print(indices)

b = X[indices]

print(b)