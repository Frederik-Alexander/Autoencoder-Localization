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
    images = []
    positions = []
    directions = []
    layout= []

    last_save_time = time.time()
    while True:
        env_value = queue.get()
        #print(np.sum(env_value))
        #print(env_value)


        # Save each array in a separate .npy file
        current_time = time.time()
        if current_time - last_save_time >= 30: #20
            print("Saving data")
            #main_task_process.join()
            break
            #print(np.sum(env_value)) 
        sleep_time = 0.01
        time.sleep(sleep_time)

        positions.append(env_value[0])
        directions.append(env_value[1])
        images.append(env_value[2])
        layout.append(env_value[3])
 

        # Assuming images, positions, and directions are your lists of data
    print(len(directions))
    #print(directions)

    #remove the first 20 datapoints 
    images = images[20:]
    positions = positions[20:]
    directions = directions[20:]
    layout = layout[20:]

    data = {
        'images': images,
        'positions': positions,
        'directions': directions,
        'maze_layout': layout
    }

        

    return data

def direction_to_angle(dx,dy):
    #, dy = direction  # Assuming direction is a tuple (dx, dy)
    angle_radians = math.atan2(dy, dx)
    #angle_degrees = math.degrees(angle_radians)
    return angle_radians

def xy_to_polar (x,y): # VON MIR 

    radius = math.sqrt(x**2 + y**2)
    angle = math.atan2(y, x)  # This returns the angle in radians
    return radius, angle 
        



import numpy as np





#pos_dir = np.concatenate((images, positions), axis=0)



if __name__ == '__main__':
    data = inside_func()
    

    #print(data)
    import pickle
    file_name = 'only_circle_data.pkl'
    # Save data to a file
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    print("done saving")