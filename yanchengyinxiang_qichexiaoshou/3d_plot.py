from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

x = np.arange(-20, 20, 0.25)
y = np.arange(-20, 20, 0.25)

x, y = np.meshgrid(x, y)
z = (np.power(x, 2) + 2 * x * y) / 240
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')

plt.show()
