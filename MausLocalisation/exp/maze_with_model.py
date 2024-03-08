   

from mazelive import runguicust
def run_main_task(queue):
    runguicust.main(queue)

def inside_func():
    from multiprocessing import Process, Queue
    
    
    queue = Queue()

    # Start run_gui_cust.main in a separate process
    main_task_process = Process(target=run_main_task, args=(queue,))
    main_task_process.start()

    model = Process(target=run_main_task, args=(queue,))
    main_task_process.start()


    # Periodically read from the queue in your main script
    try:
        while True:
            # This will block until there's something in the queue
            env_value = queue.get()
            print("Received from queue:", env_value)

            # Implement your logic for using env_value

    except KeyboardInterrupt:
        # Handle any cleanup here
        pass

    # Wait for the process to finish, if needed
    main_task_process.join()

if __name__ == '__main__':
    inside_func()
      