#Live 3d Torus 

import runguicust_QUEUE_ONLY_POS
import time
import math 

def run_main_task(queue):
    runguicust_QUEUE_ONLY_POS.main(queue)

#if __name__ == '__main__':
    #inside_func()


import runguicust_QUEUE_ONLY_POS
import time
import numpy as np  # Import numpy

def run_main_task(queue):
    runguicust_QUEUE_ONLY_POS.main(queue)


def inside_func():
    from multiprocessing import Process, Queue

    queue = Queue()
    main_task_process = Process(target=run_main_task, args=(queue,))
    main_task_process.start()

    counter = 0

    positions = []
    angles = []


    last_save_time = time.time()
    while True:
        env_value = queue.get()
        #print(np.sum(env_value))
        print(env_value)
        positions.append(env_value[0])
        angles.append(env_value[1])

        # Save each array in a separate .npy file
        current_time = time.time()
        if current_time - last_save_time >= 2000: #20
            main_task_process.join()
            #break
            #print(np.sum(env_value)) 
        sleep_time = 0.01
        time.sleep(sleep_time)

    return positions,angles

def direction_to_angle(dx,dy):
    #, dy = direction  # Assuming direction is a tuple (dx, dy)
    angle_radians = math.atan2(dy, dx)
    #angle_degrees = math.degrees(angle_radians)
    return angle_radians

def xy_to_polar (x,y): # VON MIR 

    radius = math.sqrt(x**2 + y**2)
    angle = math.atan2(y, x)  # This returns the angle in radians
    return radius, angle 
        

def pos_dir_to_torus(pos_dir):
    ## Converts an x,y and direction coordiante to a torus coordinate
    R = 1.2 # Distance from the center of the hole to the center of the tube
    r = 0.5 # Radius of the tube
    print(pos_dir)
    x = (pos_dir[0]  - 4.5) /4.5
    y  = (pos_dir[1]  - 4.5) /4.5

    # x and y are on the unit circle, convert them to polar coordinates
    v, theta = xy_to_polar(x,y)

    phi = pos_dir[2]
    print("v´s", v)
    print("phis ", phi  )
    print("thetas ", theta)


    x = (R + v * r * np.cos(phi)) * np.cos(theta)
    y = (R + v * r * np.cos(phi)) * np.sin(theta)
    z = v * r * np.sin(phi)

    return x,y,z

def Torus_vis(data):

        coords = data[0]
        angle = [direction_to_angle(data[1][i][0],data[1][i][1]) for i in range(len(data[1]))]
        print(coords)
        print(angle)
        x = []
        y = []
        z = []
        for i in range(len(coords)):
            x_returned, y_returned, z_returned = pos_dir_to_torus([coords[i][0],coords[i][1],angle[i]])
            x.append(x_returned)
            y.append(y_returned)
            z.append(z_returned)


        print("x",x)
        print("y",y)
        print("z",z)

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        # Plotting
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plotting a subset of points for clarity
        ax.scatter(x,y,z, color='b', s=10)

        # Find the maximum and minimum bounds across all dimensions
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

        # Get the mid points in each dimension
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5

        # Set the limits
        #ax.set_xlim(mid_x - max_range, mid_x + max_range)
        #ax.set_ylim(mid_y - max_range, mid_y + max_range)
        #ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)



        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.title('Filled Torus')
        plt.show()

import numpy as np





#pos_dir = np.concatenate((images, positions), axis=0)



if __name__ == '__main__':
    data = inside_func()
    #print(data)
    # Testing with random data: 
   
    #data = ([array([6.5, 7.5]), array([6.5, 7.5]), array([6.5, 7.5]), array([6.5, 7.5]), array([6.5, 7.5]), array([6.5, 7.5])], [array([ 0.18859416, -0.98205511]), array([ 0.18859416, -0.98205511]), array([ 0.18859416, -0.98205511]), array([ 0.18859416, -0.98205511]), array([ 0.18859416, -0.98205511]), array([ 0.18859416, -0.98205511])])
    #data = generate_random_data(60)
    print(data) # expected structure: tuple of two lists, in each list eintreis are arrays with 2 elemts 
    #print(data[1].type)
    #print("data 1: " ,data[1])
    Torus_vis(data)

    # Rungs the game for 15 seconds and records the x,y positions and angles 
    #Then visulaises them in a torus: 

