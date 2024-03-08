import runguicust
import time

def run_main_task(queue):
    runguicust.main(queue)

def inside_func_old():
    from multiprocessing import Process, Queue
    import time
    
    queue = Queue()
    main_task_process = Process(target=run_main_task, args=(queue,))
    main_task_process.start()

    try:
        with open("data_log.txt", "a") as file:  # Open file in append mode
            while True:
                #print(str(env_value))
                env_value = queue.get()
                print(env_value)
                print(env_value.dtype)
                env_value_list = [i for i in env_value]
                env_value = str(env_value)
                file.write(f"{env_value}\n")
                file.flush()  # Ensure data is written to the file
                time.sleep(1)  # Update once per second
    except KeyboardInterrupt:
        pass

    main_task_process.join()

#if __name__ == '__main__':
    #inside_func()


import runguicust
import time
import numpy as np  # Import numpy

def run_main_task(queue):
    runguicust.main(queue)

def inside_funcold():
    from multiprocessing import Process, Queue
    
    queue = Queue()
    main_task_process = Process(target=run_main_task, args=(queue,))
    main_task_process.start()

    try:
        with open("data_log.txt", "a") as file:  # Open file in append mode
            while True:
                env_value = queue.get()
                print(env_value.shape)
                #print(sum(env_value))
                #print(env_value[-1].dtype)

                # Use numpy.savetxt to write the array to the file
                np.savetxt(file, env_value, fmt='%s')

                file.flush()  # Ensure data is written to the file
                time.sleep(1)  # Update once per second
    except KeyboardInterrupt:
        pass

    main_task_process.join()


def save(env_value, counter):
    #np.save(f"/Users/frederikalexander/My Drive/Computer Science/Bachelorarbeit Maus Topo/aVis_data/data_log_{counter}.npy", env_value)
    np.save(f"/Users/frederikalexander/My Drive/Computer Science/maus/data_for_live_demo/data_log_{counter}.npy", env_value)
    
    counter += 1


def delete_files():
    import glob
    import os

    files = glob.glob("/Users/frederikalexander/My Drive/Computer Science/maus/data_for_live_demo/data_log_*.npy")
    for file in files:
        os.remove(file)

def inside_func():
    from multiprocessing import Process, Queue

    queue = Queue()
    main_task_process = Process(target=run_main_task, args=(queue,))
    main_task_process.start()
    delete_files()
    counter = 0
    try:
        last_save_time = time.time()
        while True:
            env_value = queue.get()
            #print(np.sum(env_value))
            # Get the shape of env_value
            print(env_value.shape)
            #print(env_value)

            # Save each array in a separate .npy file
            current_time = time.time()
            if current_time - last_save_time >= 0.2:
                #print(np.sum(env_value)) 
                save(env_value, counter)
                counter += 1
                last_save_time = current_time

            
    except KeyboardInterrupt:
        pass    

    main_task_process.join()

if __name__ == '__main__':
    inside_func()


