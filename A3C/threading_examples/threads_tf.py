import numpy as np
import itertools
import time
import threading
import multiprocessing
import tensorflow as tf
import random

global_step = 0
global_counter = itertools.count()


class Worker:
    def __init__(self, id_):
        self.id = id_
        self.local_counter = itertools.count()

    def run(self):
        global global_step

        while True:
            time.sleep(random.choice(range(1, 4)) * 2)
            local_step = next(self.local_counter)
            global_step = next(global_counter)
            print(f"Worker({self.id}): {local_step}")
            if global_step > 20:
                break


num_workers = multiprocessing.cpu_count()
print(f"Num cpu cores: {num_workers}")

# Create the workers
workers = []
for worker_id in range(num_workers):
    worker = Worker(worker_id)
    workers.append(worker)

# Threads coordinator
coordinator = tf.train.Coordinator()

# Start the threads
worker_threads = []
for worker in workers:
    worker_fn = lambda: worker.run()
    t = threading.Thread(target=worker_fn)
    t.start()
    worker_threads.append(t)


while not coordinator.should_stop():
    if global_step == 20:
        coordinator.request_stop()
coordinator.join(worker_threads, stop_grace_period_secs=10)

print("Done")
