import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## WORKS! 

# Parameters for the torus
R = 3 # Distance from the center of the hole to the center of the tube
r = 0.5 # Radius of the tube

# Parametric variables
theta = np.linspace(0, 2*np.pi, 100)


phi = np.linspace(0, 2*np.pi, 100)
#phi = np.array([1])

v = np.linspace(0, 1, 100)

print("v´s", v)
print("phis ", phi  )
print("thetas ", theta)
#v = np.array([0])
# Creating a grid of parameters
theta, phi, v = np.meshgrid(theta, phi, v)

# Parametric equations for the filled torus
x = (R + v * r * np.cos(phi)) * np.cos(theta)
y = (R + v * r * np.cos(phi)) * np.sin(theta)
z = v * r * np.sin(phi)




# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plotting a subset of points for clarity
ax.scatter(x[::5, ::5, ::5], y[::5, ::5, ::5], z[::5, ::5, ::5], color='b', s=1)

# Find the maximum and minimum bounds across all dimensions
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

# Get the mid points in each dimension
mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5

# Set the limits
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.title('Filled Torus')
plt.show()
