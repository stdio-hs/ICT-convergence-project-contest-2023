import numpy as np

x = np.random.rand(3,2,3)

print(x)

x = x[:, :, 0]

print(x)