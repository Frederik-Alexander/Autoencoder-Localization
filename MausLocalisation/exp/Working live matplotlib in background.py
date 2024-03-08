import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt
import time
import random


def init_live_plot():
    plt.ion()
    fig, ax = plt.subplots()
    sc = ax.scatter([], [])
    ax.set_xlim(0, 10)  # Adjust these limits based on your expected data range
    ax.set_ylim(0, 10)
    return ax, sc 


def update_plot(ax, sc ,new_points):
    x, y = zip(*new_points)
    sc.set_offsets(list(zip(x, y)))
    ax.relim()  # Recalculate limits
    ax.autoscale_view()  # Auto-adjust to new data
    plt.draw()
    plt.pause(0.1)

try:
    new_points = []
    ax, sc =init_live_plot()
    while True:
        # Example: generate a new point
        new_x = random.uniform(0, 10)
        new_y = random.uniform(0, 10)
        new_points.append((new_x, new_y))

        update_plot(ax, sc,new_points )
        time.sleep(1)  # You can adjust the sleep time as needed
except KeyboardInterrupt:
    pass
